"""
Continuous batching: requests join and leave a running batch between two
decode steps (Yu et al., 2022, OSDI: iteration-level scheduling).

stream() in minllm/inference.py generates for a fixed set of prompts. The
batch starts together and ends when its longest reply ends, and a row that
finished early still takes part in every step until then. A server cannot
work that way: requests arrive at any time, and a short question should not
wait for a long answer to someone else.

Here the batch is a fixed number of slots. A request waiting for a free slot
is prefilled alone, its keys and values are copied into the slot, and from
the next decode step on it is one row of the batch. When it ends, its slot
is free for the next request. Every slot holds a sequence of its own length,
so every row has its own position. See docs/04c-continuous-batching.md.
"""

from collections import deque

import torch

from minllm.inference import KVCache, next_logits, sample


class SlotLayer:
    """The keys and values of one attention layer, for every slot."""

    def __init__(self, cache):
        self.cache = cache
        self.k = self.v = None  # (slots, kv heads, max_len, head_dim)

    def put(self, slot, k, v):
        """Write the keys and values of one prefilled sequence, (1, kv heads, length, head_dim), into a slot."""
        if self.k is None:
            _, H, _, D = k.shape
            self.k = k.new_zeros(self.cache.slots, H, self.cache.max_len, D)
            self.v = v.new_zeros(self.cache.slots, H, self.cache.max_len, v.shape[-1])
        self.k[slot, :, : k.shape[2]] = k[0]
        self.v[slot, :, : v.shape[2]] = v[0]

    def append(self, k, v):
        """
        Store one new token per slot, each at its slot's own position, and
        return the keys and values up to the longest slot. Shorter slots are
        masked by the caller.
        """
        assert k.shape[2] == 1, "a slot takes one token per step; a prompt goes in by put"
        rows = torch.arange(self.cache.slots, device=k.device)
        self.k[rows, :, self.cache.pos] = k[:, :, 0]
        self.v[rows, :, self.cache.pos] = v[:, :, 0]
        return self.k[:, :, : self.cache.width], self.v[:, :, : self.cache.width]


class SlotCache:
    """A cache of `slots` rows, each a sequence with its own length, which is the position of its next token."""

    def __init__(self, n_layers, slots, max_len, device):
        self.slots, self.max_len = slots, max_len
        self.pos = torch.zeros(slots, dtype=torch.long, device=device)  # the lengths, as the model reads them
        self.width = 0  # the keys every row attends over in the next step: up to the longest slot
        self.layers = [SlotLayer(self) for _ in range(n_layers)]

    @property
    def nbytes(self):
        return sum(t.numel() * t.element_size() for layer in self.layers for t in (layer.k, layer.v) if t is not None)


class Batcher:
    """
    A running batch of `slots` rows. add() queues a request; every call to
    step() first moves waiting requests into free slots, then decodes one
    token for every running request, and returns what came out.
    """

    def __init__(self, model, slots, stop=(), temperature=0.0, top_k=None, top_p=None):
        self.model, self.slots = model, slots
        self.sampling = dict(temperature=temperature, top_k=top_k, top_p=top_p)
        self.stop = set(stop)
        self.device = next(model.parameters()).device
        self.cache = SlotCache(model.cfg.n_layers, slots, model.cfg.max_seq_len, self.device)
        self.waiting = deque()
        # Per slot, kept on the CPU and sent to the GPU once per step: the
        # request in it (None when free), its length, and the token it reads next.
        self.running, self.lengths, self.next_ids = [None] * slots, [0] * slots, [0] * slots
        self.count = 0

    def add(self, prompt, max_new_tokens):
        """Queue a request: a list of token ids. Returns its id."""
        self.count += 1
        self.waiting.append({"id": self.count, "prompt": prompt, "left": max_new_tokens})
        return self.count

    def busy(self):
        return bool(self.waiting) or any(r is not None for r in self.running)

    def emit(self, slot, token, out):
        """Hand out a new token. A request ends at a stop token, at its limit, or when the context is full."""
        request = self.running[slot]
        request["left"] -= 1
        stop = token in self.stop
        done = stop or request["left"] == 0 or self.lengths[slot] + 1 >= self.cache.max_len
        out.append((request["id"], None if stop else token, done))
        if done:
            # A free slot computes along with the others, harmlessly, at position 0.
            self.running[slot], self.lengths[slot] = None, 0
        else:
            self.next_ids[slot] = token

    @torch.no_grad()
    def step(self):
        """
        Returns (request id, token, finished) for every request that moved:
        a token of None means the request ended at a stop token.
        """
        out = []
        # Admit: every waiting request that finds a free slot is prefilled
        # alone, as stream() would do, and its cache copied into the slot.
        # Its first token comes out of the prefill.
        for slot in range(self.slots):
            if not self.waiting:
                break
            if self.running[slot] is not None:
                continue
            request = self.waiting.popleft()
            prompt = request["prompt"]
            assert len(prompt) < self.cache.max_len, "the prompt leaves no room in the context"
            alone = KVCache(self.model.cfg.n_layers, len(prompt))
            logits = next_logits(self.model, torch.tensor([prompt], device=self.device), alone, None)
            for layer, single in zip(self.cache.layers, alone.layers):
                layer.put(slot, single.k, single.v)
            self.running[slot], self.lengths[slot] = request, len(prompt)
            self.emit(slot, sample(logits[:, -1], **self.sampling).item(), out)

        # Decode: one token for every slot at once. A row sees the keys of
        # its own sequence, up to and including the token it reads now.
        active = [slot for slot in range(self.slots) if self.running[slot] is not None]
        if not active:
            return out
        self.cache.pos = torch.tensor(self.lengths, device=self.device)
        self.cache.width = max(self.lengths) + 1
        mask = torch.arange(self.cache.width, device=self.device) <= self.cache.pos[:, None]
        idx = torch.tensor(self.next_ids, device=self.device)[:, None]
        tokens = sample(next_logits(self.model, idx, self.cache, mask[:, None, None])[:, -1], **self.sampling).tolist()
        for slot in active:
            self.lengths[slot] += 1
            self.emit(slot, tokens[slot], out)
        return out


def generate(model, prompts, max_new_tokens, slots, **sampling):
    """Every prompt's continuation, as generate() in minllm/inference.py gives it, through a running batch of `slots` rows."""
    batcher = Batcher(model, slots, **sampling)
    ids = {batcher.add(p, max_new_tokens): i for i, p in enumerate(prompts)}
    out = [[] for _ in prompts]
    while batcher.busy():
        for request, token, _ in batcher.step():
            if token is not None:
                out[ids[request]].append(token)
    return out
