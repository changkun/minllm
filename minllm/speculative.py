"""
Speculative decoding: a cheap guess at the next few tokens, checked by the
model in one pass.

Generating one sequence, a model does one forward pass per token, and on a
laptop that pass is limited by reading the weights and by the overhead of
launching its many small operations, not by arithmetic (docs/04-inference.md).
A pass over five new tokens costs little more than a pass over one. So a
draft proposes k tokens, the model computes its own distribution at all k
positions in one pass, and keeps the longest prefix of the draft that it
agrees with, plus one token of its own. Every pass then yields between 1
and k + 1 tokens.

What comes out is exactly what the model alone would write: the same
tokens when decoding greedily, and the same distribution when sampling,
by the rule of Leviathan et al. (2022, arXiv:2211.17192) and Chen et al.
(2023, arXiv:2302.01318) below. The draft decides only the speed.

Two drafts are here: a smaller model with the same tokenizer, and no model
at all, the continuation of the last place where the text so far repeats
its final few tokens (prompt lookup). See docs/04b-speculative.md.
"""

import torch
import torch.nn.functional as F

from minllm.inference import KVCache


def probabilities(logits, temperature):
    """The distribution that sampling at this temperature draws from; greedy decoding is the one-hot of the likeliest token."""
    if temperature == 0:
        return F.one_hot(logits.argmax(-1), logits.size(-1)).float()
    return F.softmax(logits.float() / temperature, dim=-1)


class Cached:
    """
    A model with a KV cache over a growing sequence, fed only the tokens
    it has not seen. Between calls the cache holds at most every token of
    the sequence but the last, whose next-token distribution is still to
    be computed.
    """

    def __init__(self, model, max_len):
        self.model, self.cache = model, KVCache(model.cfg.n_layers, max_len)

    def logits(self, seq, extra=()):
        """
        In one pass, the next-token logits after seq and after each token of
        extra that follows it: (1 + len(extra), vocab).
        """
        new = (list(seq) + list(extra))[self.cache.pos :]
        idx = torch.tensor([new], device=next(self.model.parameters()).device)
        h = self.model.hidden(idx, cache=self.cache)
        return self.model.lm_head(h[0, -(1 + len(extra)) :]).float()

    def keep(self, length):
        self.cache.crop(length)


class ModelDraft:
    """Draft with a smaller model: k tokens, one cached pass each."""

    def __init__(self, model, max_len):
        self.cached = Cached(model, max_len)

    def propose(self, seq, k, temperature):
        tokens, q = [], []
        for _ in range(k):
            p = probabilities(self.cached.logits(seq + tokens), temperature)[-1]
            token = int(torch.multinomial(p, 1)) if temperature else int(p.argmax())
            tokens.append(token)
            q.append(p)
        return tokens, torch.stack(q) if q else None

    def keep(self, length):
        # The draft has read every proposed token but the last; what it read beyond length was rejected.
        self.cached.keep(length)


class LookupDraft:
    """
    Draft without a model: find the last earlier place where the final n
    tokens of the text occur, longest n first, and propose the k tokens
    that followed them there. A reply that copies from its prompt, as a
    solution copies the numbers of its problem, is drafted well. The
    proposal is certain, a one-hot distribution.
    """

    def __init__(self, vocab_size, ngram=3):
        self.vocab_size, self.ngram = vocab_size, ngram

    def propose(self, seq, k, temperature):
        tokens = []
        for n in range(min(self.ngram, len(seq) - 1), 0, -1):
            tail = seq[-n:]
            for i in range(len(seq) - n - 1, -1, -1):
                if seq[i : i + n] == tail:
                    tokens = seq[i + n : i + n + k]
                    break
            if tokens:
                break
        return tokens, F.one_hot(torch.tensor(tokens, dtype=torch.long), self.vocab_size).float() if tokens else None

    def keep(self, length):
        pass


def accept(p, q, tokens, temperature, generator=None):
    """
    Which draft tokens the model keeps, and the token it adds. p is the
    model's distribution at every draft position and one more, (k + 1,
    vocab); q the draft's at every draft position, (k, vocab). Draft token
    i is kept with probability min(1, p_i / q_i) of that token, which is 1
    when the model would have picked it greedily. At the first one
    rejected, the model's token is drawn from what p has beyond q,
    max(0, p - q) normalized: the probability that the draft took from a
    token and the check gave back. If all are kept, it is drawn from the
    last row of p. Over the draft's own randomness, every token then has
    exactly its probability under p.
    """
    for i, token in enumerate(tokens):
        ratio = p[i, token] / q[i, token]
        if (temperature == 0 and p[i].argmax() == token) or (temperature and torch.rand((), generator=generator) < ratio):
            continue
        if temperature == 0:
            return tokens[:i], int(p[i].argmax())
        rest = (p[i] - q[i]).clamp(min=0)
        return tokens[:i], int(torch.multinomial(rest / rest.sum(), 1, generator=generator))
    last = p[len(tokens)]
    return tokens, int(last.argmax()) if temperature == 0 else int(torch.multinomial(last, 1, generator=generator))


@torch.no_grad()
def generate(model, draft, prompt, max_new_tokens, k=4, temperature=0.0, stop=()):
    """
    Generate a continuation of one prompt with a draft. Returns the new
    tokens and how it went: the passes of the model, the draft tokens
    proposed and those kept.
    """
    target = Cached(model, len(prompt) + max_new_tokens + k + 1)
    seq, new = list(prompt), []
    stats = {"passes": 0, "proposed": 0, "kept": 0}
    while len(new) < max_new_tokens:
        tokens, q = draft.propose(seq, min(k, max_new_tokens - len(new) - 1), temperature)
        # One pass of the model: its distribution after the text, and after every draft token.
        p = probabilities(target.logits(seq, tokens), temperature)
        if tokens:
            q = q.to(p.device)
            q = F.pad(q, (0, p.size(-1) - q.size(-1))) if q.size(-1) < p.size(-1) else q[:, : p.size(-1)]
        kept, extra = accept(p, q, tokens, temperature)
        stats["passes"] += 1
        stats["proposed"] += len(tokens)
        stats["kept"] += len(kept)
        # Both caches keep the text and the kept tokens; the token added last is read in the next pass.
        target.keep(len(seq) + len(kept))
        draft.keep(len(seq) + len(kept))
        for token in kept + [extra]:
            new.append(token)
            seq.append(token)
            if token in stop or len(new) == max_new_tokens:
                return new, stats
    return new, stats
