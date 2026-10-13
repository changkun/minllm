"""
Longer context: using a model on more tokens than it was trained on.

The model of stage 3 was trained on windows of 256 tokens. Nothing in its
weights depends on that length: attention takes any number of tokens, and
the rotary table of minllm/model.py can be computed for any position. What
the model has never seen is the angles that the slow pairs of the table
reach beyond 256, and the question is what it makes of them, and how to
stretch the table so that it never has to. See docs/02b-context.md.

Two measurements. The loss at every position of long documents says
whether the model still predicts well far into a window. It says little
about whether the far tokens help, which depends on the document. The
copy probe does: a passage, other documents, and the passage again. A
model that can reach back to the first copy predicts the second far
better, at any distance it can reach.
"""

import torch
import torch.nn.functional as F

from minllm.model import rope_table


def extend(model, max_seq_len, kind=None, factor=None):
    """
    Let a trained model read max_seq_len tokens, with its rotary table
    stretched by `factor` in the way `kind` says (see rope_table), or
    computed further as it is when kind is None. Changes the model in place.
    """
    cfg = model.cfg
    # the length the model was trained on, kept across extensions
    original = getattr(model, "trained_len", None) or (cfg.rope_scaling["original"] if cfg.rope_scaling else cfg.max_seq_len)
    model.trained_len = original
    cfg.max_seq_len = max_seq_len
    cfg.rope_scaling = {"type": kind, "factor": factor, "original": original} if kind else None
    cos, sin = rope_table(cfg.dim // cfg.n_heads, max_seq_len, cfg.rope_theta, cfg.rope_scaling)
    device = model.rope_cos.device
    model.rope_cos, model.rope_sin = cos.to(device), sin.to(device)
    return model


@torch.no_grad()
def loss_by_position(model, tokens, length, starts, batch_size=8):
    """
    The mean loss at every position of windows of `length` tokens that
    begin at `starts` in tokens: a (length - 1,) tensor. Windows that begin
    where documents begin show how the loss falls as more of the document
    has been read.
    """
    device = next(model.parameters()).device
    total = torch.zeros(length - 1, device=device)
    for i in range(0, len(starts), batch_size):
        x = torch.stack([torch.from_numpy(tokens[o : o + length].astype("int64")) for o in starts[i : i + batch_size]]).to(device)
        logits, _ = model(x[:, :-1])
        total += F.cross_entropy(logits.float().transpose(1, 2), x[:, 1:], reduction="none").sum(0)
    return (total / len(starts)).cpu()


def copy_sequences(documents, distance, separator, passage=48, offset=40, n=64, control=False):
    """
    n sequences of a passage from inside a document, then other documents
    until `distance` tokens separate the passage from its copy, then the
    passage again. With control, the first passage is from another
    document, so that the second is read for the first time: the difference
    in loss is what the first sighting was worth. documents is a list of
    token lists. Returns the sequences and where each copy starts.
    """
    usable = [d for d in documents if len(d) >= offset + passage]
    sequences, starts = [], []
    for i in range(n):
        second = usable[i][offset : offset + passage]
        first = usable[(i + n) % len(usable)][offset : offset + passage] if control else second
        filler, j = [], 2 * n
        while len(filler) < distance - passage:
            filler += [separator] + usable[(j + 7 * i) % len(usable)]
            j += 1
        sequences.append([separator] + first + filler[: distance - passage] + second)
        starts.append(1 + distance)
    return sequences, starts


@torch.no_grad()
def copy_loss(model, sequences, starts, batch_size=8):
    """The mean loss on the copies, from their second token on: low when the model finds the first copy."""
    device = next(model.parameters()).device
    total, count = 0.0, 0
    for i in range(0, len(sequences), batch_size):
        x = torch.tensor(sequences[i : i + batch_size], device=device)
        logits, _ = model(x[:, :-1])
        loss = F.cross_entropy(logits.float().transpose(1, 2), x[:, 1:], reduction="none")
        for row, start in enumerate(starts[i : i + batch_size]):
            # position t predicts token t + 1: the copy's tokens after its first are predicted from start on
            total += loss[row, start:].sum().item()
            count += loss.shape[1] - start
    return total / count
