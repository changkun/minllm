import math

import numpy as np
import torch

from minllm import context
from minllm.model import Config, Transformer, rope_table


def test_no_scaling_is_the_plain_table():
    cos, sin = rope_table(64, 100, 10000.0)
    for scaling in (None, {}):
        c, s = rope_table(64, 100, 10000.0, scaling)
        assert torch.equal(c, cos) and torch.equal(s, sin)


def test_interpolation_turns_every_pair_slower():
    cos, _ = rope_table(64, 1024, 10000.0)
    stretched, _ = rope_table(64, 1024, 10000.0, {"type": "linear", "factor": 4, "original": 256})
    assert torch.allclose(stretched[4 * 100], cos[100], atol=1e-5)  # position 400 looks as position 100 did


def test_ntk_keeps_the_fastest_pair_and_slows_the_slowest():
    angle = lambda c, s: torch.atan2(s, c)
    cos, sin = rope_table(64, 2, 10000.0)
    c, s = rope_table(64, 2, 10000.0, {"type": "ntk", "factor": 4, "original": 256})
    assert abs(angle(c, s)[1, 0] - angle(cos, sin)[1, 0]) < 1e-6
    slowest = 10000.0 ** (-62 / 64)
    assert abs(angle(c, s)[1, -1].item() - slowest / 4) < 1e-9 + 1e-3 * slowest


def test_yarn_interpolates_only_the_slow_pairs_and_sharpens():
    cos, sin = rope_table(64, 300, 10000.0)
    c, s = rope_table(64, 300, 10000.0, {"type": "yarn", "factor": 4, "original": 256})
    gain = 0.1 * math.log(4) + 1
    assert torch.allclose(c.pow(2) + s.pow(2), torch.full_like(c, gain**2), atol=1e-4)
    assert torch.allclose(c[:, 0] / gain, cos[:, 0], atol=1e-5)  # the fastest pair turns as before
    linear, _ = rope_table(64, 300, 10000.0, {"type": "linear", "factor": 4, "original": 256})
    assert torch.allclose(c[:, -1] / gain, linear[:, -1], atol=1e-5)  # the slowest as interpolated


def test_extend_and_measure():
    torch.manual_seed(0)
    model = Transformer(Config(vocab_size=32, max_seq_len=16, dim=32, n_layers=1, n_heads=2, n_kv_heads=1, mlp_dim=64)).eval()
    x = torch.randint(0, 32, (1, 16))
    before = model(x)[0]
    context.extend(model, 64)  # the same table, computed further
    assert torch.allclose(model(x)[0], before, atol=1e-5) and model.cfg.max_seq_len == 64
    context.extend(model, 64, "yarn", 4)
    assert model.cfg.rope_scaling == {"type": "yarn", "factor": 4, "original": 16}
    tokens = np.random.default_rng(0).integers(0, 32, 1000).astype(np.uint16)
    per_position = context.loss_by_position(model, tokens, 64, [0, 100, 300, 500], batch_size=2)
    assert per_position.shape == (63,) and per_position.min() > 0


def test_copy_probe():
    documents = [[i] * 50 + list(range(40)) for i in range(1, 40)]
    sequences, starts = context.copy_sequences(documents, distance=60, separator=0, passage=12, offset=40, n=3)
    controls, _ = context.copy_sequences(documents, distance=60, separator=0, passage=12, offset=40, n=3, control=True)
    for seq, control, start in zip(sequences, controls, starts):
        assert len(seq) == 1 + 60 + 12 and seq[1:13] == seq[start : start + 12] == control[start : start + 12]
        assert seq[0] == 0 and control[1:13] != seq[1:13]
    torch.manual_seed(0)
    model = Transformer(Config(vocab_size=64, max_seq_len=80, dim=32, n_layers=1, n_heads=2, n_kv_heads=1, mlp_dim=64)).eval()
    assert context.copy_loss(model, sequences, starts) > 0
