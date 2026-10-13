"""
Scaling laws: how the loss falls as the model and its data grow.

Training a model of N parameters on D tokens costs about C = 6 N D
floating-point operations: 2 N per token for the forward pass and 4 N for
the backward one. For a fixed C, a larger model sees fewer tokens. Which
split gives the lowest loss?

Hoffmann et al. (2022, arXiv:2203.15556) answered it by training many
models at a few budgets each. At every budget the final loss against log N
is a valley, whose floor is the best size for that budget, and the best
sizes of the budgets lie on a power law, N_opt = k C^a. They also fitted
one formula to every run,

    L(N, D) = E + A / N^alpha + B / D^beta

where E is the loss that no model reaches, the second term what a model of
finite size costs, and the third what finite data costs. This file holds
the family of models that the runs are drawn from and both fits. See
docs/03c-scaling.md.
"""

import math

import numpy as np
import torch

from minllm.model import Config

# The family: the model of stage 3 (width 384, 8 blocks) and models narrower
# and wider, with depth growing with width so that their shape stays about
# the same. Heads are 64 wide, and the MLP is 8/3 as wide as the stream.
LAYERS = {64: 2, 128: 3, 192: 4, 256: 5, 320: 6, 384: 8, 448: 9, 512: 10}


def family(dim, vocab_size=4096, seq_len=256):
    """The configuration of the family's model of width dim."""
    heads = dim // 64
    # key/value heads: the divisor of the heads closest to a third of them, as in stage 3's 6 and 2
    kv = min((d for d in range(1, heads + 1) if heads % d == 0), key=lambda d: abs(d - heads / 3))
    return Config(vocab_size=vocab_size, max_seq_len=seq_len, dim=dim, n_layers=LAYERS[dim], n_heads=heads, n_kv_heads=kv, mlp_dim=round(dim * 8 / 3 / 32) * 32)


def flops(n_params, tokens):
    """The compute of a training run, 6 N D. Attention over the window adds a few percent that this leaves out."""
    return 6 * n_params * tokens


def valley(n_params, losses):
    """
    The floor of one budget's valley: a parabola fitted to the loss against
    log N, and the N and loss at its lowest point. The points have to
    reach up on both sides, or the floor is a guess.
    """
    x = np.log(np.asarray(n_params, dtype=float))
    a, b, c = np.polyfit(x, np.asarray(losses, dtype=float), 2)
    assert a > 0, "the losses do not form a valley"
    best = -b / (2 * a)
    return float(math.exp(best)), float(c - b * b / (4 * a))


def power_law(x, y):
    """k and a in y = k x^a, fitted as a straight line through log y against log x."""
    a, log_k = np.polyfit(np.log(np.asarray(x, dtype=float)), np.log(np.asarray(y, dtype=float)), 1)
    return float(math.exp(log_k)), float(a)


def loss_law(n_params, tokens, losses, delta=1e-3, starts=None):
    """
    E, A, alpha, B and beta of L(N, D) = E + A / N^alpha + B / D^beta, fitted
    to every run as Hoffmann et al. fitted theirs: the Huber loss of the
    difference in log loss, minimized by L-BFGS from a grid of starting
    points, keeping the best. The Huber loss counts a run that is far off
    less than a square would, so that one bad run cannot pull the fit.
    """
    N = torch.tensor(n_params, dtype=torch.float64)
    D = torch.tensor(tokens, dtype=torch.float64)
    L = torch.tensor(losses, dtype=torch.float64)

    def predicted(p):
        # the three terms in log space, combined with logsumexp so that nothing overflows
        log_a, log_b, log_e, alpha, beta = p
        return torch.logsumexp(torch.stack([log_a - alpha * N.log(), log_b - beta * D.log(), log_e.expand_as(N)]), dim=0)

    best, best_value = None, math.inf
    starts = starts or [(a, b, e, al, be) for a in (0.0, 5.0, 10.0) for b in (0.0, 5.0, 10.0) for e in (-1.0, 0.0, 0.5) for al in (0.2, 0.5) for be in (0.2, 0.5)]
    for start in starts:
        p = torch.tensor(start, dtype=torch.float64, requires_grad=True)
        optimizer = torch.optim.LBFGS([p], lr=1.0, max_iter=500, line_search_fn="strong_wolfe")

        def closure():
            optimizer.zero_grad()
            value = torch.nn.functional.huber_loss(predicted(p), L.log(), delta=delta, reduction="sum")
            value.backward()
            return value

        optimizer.step(closure)
        with torch.no_grad():
            value = torch.nn.functional.huber_loss(predicted(p), L.log(), delta=delta, reduction="sum").item()
        if math.isfinite(value) and value < best_value:
            best, best_value = p.detach(), value
    log_a, log_b, log_e, alpha, beta = best.tolist()
    return {"E": math.exp(log_e), "A": math.exp(log_a), "alpha": alpha, "B": math.exp(log_b), "beta": beta}


def law(fit, n_params, tokens):
    """The loss that a fitted L(N, D) predicts."""
    return fit["E"] + fit["A"] / n_params ** fit["alpha"] + fit["B"] / tokens ** fit["beta"]


def optimal_split(fit, compute):
    """
    The N and D that minimize the fitted L(N, D) at a given compute, from
    setting its derivative along 6 N D = C to zero: the split at which one
    more parameter and one more token are worth the same.
    """
    a, b = fit["alpha"], fit["beta"]
    G = (a * fit["A"] / (b * fit["B"])) ** (1 / (a + b))
    n = G * (compute / 6) ** (b / (a + b))
    return n, compute / (6 * n)
