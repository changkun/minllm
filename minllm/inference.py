"""
Generation: sampling text from the model, without redoing work.

The sampler in model.py reruns the whole sequence for every new token, so
writing n tokens costs on the order of n^2. That is wasteful, because the
model is causal: what it computed for a token cannot change when later tokens
arrive. The only thing a new token needs from the earlier ones is their keys
and values, in every attention layer. Keeping those in a cache makes each new
token cost one forward pass over a single position.

See docs/04-inference.md.
"""

import torch
import torch.nn.functional as F


class LayerCache:
    """The keys and values of one attention layer, for every token so far."""

    def __init__(self, max_len):
        self.max_len = max_len
        self.k = self.v = None  # (batch, kv heads, max_len, head_dim)
        self.len = 0

    def append(self, k, v):
        """Store the keys and values of the new tokens, return those of all tokens."""
        if self.k is None:
            # One allocation for the whole generation. Growing the tensors by
            # concatenation instead would copy the entire cache at every step.
            B, H, _, D = k.shape
            self.k, self.v = k.new_empty(B, H, self.max_len, D), v.new_empty(B, H, self.max_len, v.shape[-1])
        T = k.shape[2]
        self.k[:, :, self.len : self.len + T] = k
        self.v[:, :, self.len : self.len + T] = v
        self.len += T
        return self.k[:, :, : self.len], self.v[:, :, : self.len]


class KVCache:
    def __init__(self, n_layers, max_len):
        self.layers = [LayerCache(max_len) for _ in range(n_layers)]

    @property
    def pos(self):
        """How many tokens are cached, which is the position of the next one."""
        return self.layers[-1].len

    @property
    def nbytes(self):
        return sum(t.numel() * t.element_size() for layer in self.layers for t in (layer.k, layer.v))

    def crop(self, length):
        """Forget every token after the first `length`. Their entries are overwritten by the next ones appended."""
        for layer in self.layers:
            layer.len = min(layer.len, length)


def sample(logits, temperature=1.0, top_k=None, top_p=None):
    """
    Choose the next token for every row of logits (batch, vocab).

    temperature rescales the logits before the softmax: below 1 sharpens the
    distribution, above 1 flattens it, and 0 always takes the likeliest token.
    top_k samples only among the k likeliest tokens. top_p samples among the
    smallest set of likeliest tokens whose probabilities sum to at least p, so
    the set is small when the model is sure and large when it is not.
    """
    if temperature == 0:
        return logits.argmax(dim=-1)
    logits = logits / temperature
    if top_k is not None:
        kth = torch.topk(logits, min(top_k, logits.size(-1))).values[:, [-1]]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    probs = F.softmax(logits.float(), dim=-1)
    if top_p is not None:
        ranked, order = probs.sort(dim=-1, descending=True)
        before = ranked.cumsum(dim=-1) - ranked  # probability of all likelier tokens
        ranked = ranked.masked_fill(before >= top_p, 0.0)
        probs = torch.zeros_like(probs).scatter(-1, order, ranked)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


def next_logits(model, idx, cache, mask):
    """The logits at the last position only, (batch, 1, vocab)."""
    h = model.hidden(idx, cache=cache, mask=mask)
    return model.lm_head(h[:, -1:])


@torch.no_grad()
def stream(model, prompts, max_new_tokens, temperature=1.0, top_k=None, top_p=None, stop=(), allow=None):
    """
    Generate a continuation for every prompt at once, yielding one (batch,)
    tensor of new token ids per step. A row is finished once it samples a
    token in `stop`; from then on its entry is -1. Generation ends when every
    row is finished, after max_new_tokens, or when the context is full.

    prompts is a list of token id lists. They may differ in length.

    allow, if given, restricts what may be generated. Before every step it
    is called with the tokens picked in the step before, a (batch,) tensor,
    or None before the first step, and returns which tokens may come next:
    booleans of shape (batch, vocab). Whatever it does not allow gets a
    probability of zero.
    """
    device = next(model.parameters()).device
    B, longest = len(prompts), max(len(p) for p in prompts)
    n_steps = min(max_new_tokens, model.cfg.max_seq_len - longest)
    assert n_steps > 0, "the prompt leaves no room in the context"

    # Rows of a batch must have one length, so shorter prompts are padded on
    # the left: every row then ends in its last real token, and the model's
    # final position predicts the next token for all of them. `real` marks the
    # positions that hold tokens; padding must never be attended to. Rotary
    # embeddings only see distances, so shifting a prompt right changes nothing.
    idx = torch.zeros(B, longest, dtype=torch.long, device=device)
    real = torch.ones(B, longest + n_steps, dtype=torch.bool, device=device)
    for i, prompt in enumerate(prompts):
        idx[i, longest - len(prompt) :] = torch.tensor(prompt, device=device)
        real[i, : longest - len(prompt)] = False
    padded = not bool(real.all())

    mask = None
    if padded:
        causal = torch.ones(longest, longest, dtype=torch.bool, device=device).tril()
        # The diagonal stays on: a padding position has to attend to something,
        # or its softmax runs over an empty set.
        mask = (causal & real[:, None, :longest]) | torch.eye(longest, dtype=torch.bool, device=device)
        mask = mask[:, None]  # (B, 1, T, T), shared by all heads

    cache = KVCache(model.cfg.n_layers, longest + n_steps)
    stop = torch.tensor(list(stop), dtype=torch.long, device=device)
    done = torch.zeros(B, dtype=torch.bool, device=device)

    # Prefill: the whole prompt in one pass. Only its last position predicts
    # anything, so the output layer is applied there alone: over a long prompt
    # the logits of every position are batch x length x vocabulary numbers.
    logits = next_logits(model, idx, cache, mask)
    next_id = None
    for step in range(n_steps):
        scores = logits[:, -1]
        if allow is not None:
            scores = scores.masked_fill(~allow(next_id).to(device), float("-inf"))
        next_id = sample(scores, temperature, top_k, top_p)
        done = done | (next_id[:, None] == stop).any(dim=-1)
        yield next_id.masked_fill(done, -1)
        if done.all() or step == n_steps - 1:
            return
        # Decode: one new position per row, attending to everything cached.
        mask = real[:, None, None, : cache.pos + 1] if padded else None
        logits = next_logits(model, next_id[:, None], cache, mask)


def generate(model, prompts, max_new_tokens, **sampling):
    """Like stream, but returns the finished continuations: one list of token ids per prompt."""
    steps = torch.stack(list(stream(model, prompts, max_new_tokens, **sampling)), dim=1).tolist()
    return [row[: row.index(-1)] if -1 in row else row for row in steps]


def stream_text(model, tok, prompt, max_new_tokens, **sampling):
    """
    Generate a continuation of one prompt and yield it as text, piece by
    piece. A token can end in the middle of a multi-byte character, so a
    piece is held back until the bytes so far decode cleanly.
    """
    ids, shown = [], ""
    for step in stream(model, [prompt], max_new_tokens, **sampling):
        if step[0] < 0:
            return
        ids.append(step[0].item())
        text = tok.decode(ids)
        if not text.endswith("\ufffd"):
            yield text[len(shown) :]
            shown = text
