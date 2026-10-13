import math

import numpy as np

from minllm import scaling
from minllm.model import Config, Transformer


def test_the_family_contains_the_model_of_stage_3():
    cfg = scaling.family(384)
    assert (cfg.dim, cfg.n_layers, cfg.n_heads, cfg.n_kv_heads, cfg.mlp_dim) == (384, 8, 6, 2, 1024)
    sizes = [Transformer(scaling.family(d)).num_params() for d in sorted(scaling.LAYERS)]
    assert sizes == sorted(sizes) and sizes[0] < 0.5e6 and sizes[-1] > 25e6
    assert all(c.n_heads % c.n_kv_heads == 0 for c in map(scaling.family, scaling.LAYERS))
    assert scaling.family(384) == Config(vocab_size=4096, max_seq_len=256, dim=384, n_layers=8, n_heads=6, n_kv_heads=2, mlp_dim=1024)


def test_valley_finds_the_floor():
    n = np.array([1e6, 2e6, 4e6, 8e6, 16e6])
    losses = 2.0 + 0.3 * (np.log(n) - math.log(5e6)) ** 2
    best_n, best_loss = scaling.valley(n, losses)
    assert abs(best_n / 5e6 - 1) < 1e-6 and abs(best_loss - 2.0) < 1e-9


def test_power_law():
    x = np.array([1e14, 1e15, 1e16])
    k, a = scaling.power_law(x, 3.0 * x**0.5)
    assert abs(k - 3.0) < 1e-6 and abs(a - 0.5) < 1e-9


def test_loss_law_recovers_a_known_law():
    true = {"E": 1.2, "A": 400.0, "alpha": 0.34, "B": 410.0, "beta": 0.28}
    n, d = np.meshgrid([3e5, 1e6, 3e6, 1e7, 3e7], [1e7, 3e7, 1e8, 3e8])
    n, d = n.ravel(), d.ravel()
    fit = scaling.loss_law(n.tolist(), d.tolist(), [scaling.law(true, a, b) for a, b in zip(n, d)])
    for a, b in zip(n, d):
        assert abs(scaling.law(fit, a, b) / scaling.law(true, a, b) - 1) < 1e-3
    assert abs(fit["E"] - true["E"]) < 0.05


def test_optimal_split_is_the_lowest_loss_at_that_compute():
    fit = {"E": 1.2, "A": 400.0, "alpha": 0.34, "B": 410.0, "beta": 0.28}
    compute = 1e16
    n, d = scaling.optimal_split(fit, compute)
    assert abs(scaling.flops(n, d) / compute - 1) < 1e-9
    grid = np.exp(np.linspace(math.log(n) - 2, math.log(n) + 2, 2001))
    on_budget = [scaling.law(fit, m, compute / (6 * m)) for m in grid]
    assert abs(grid[int(np.argmin(on_budget))] / n - 1) < 0.01
