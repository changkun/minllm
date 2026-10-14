import math

import torch

from minllm import dpo


def test_loss():
    zero = torch.zeros(3)
    # Before training every gain is 0, and the loss is that of a coin: log 2.
    assert math.isclose(dpo.loss(zero, zero, beta=0.1).item(), math.log(2), rel_tol=1e-6)
    # Only the difference counts: both replies falling, the other one faster, lowers the loss too.
    up = dpo.loss(torch.tensor([2.0]), torch.tensor([0.0]), beta=0.5)
    down = dpo.loss(torch.tensor([-10.0]), torch.tensor([-12.0]), beta=0.5)
    assert math.isclose(up.item(), down.item(), rel_tol=1e-6) and up.item() < math.log(2)
    # The gradient raises the preferred reply's gain and lowers the other's.
    preferred, other = torch.zeros(1, requires_grad=True), torch.zeros(1, requires_grad=True)
    dpo.loss(preferred, other, beta=0.1).backward()
    assert preferred.grad.item() < 0 < other.grad.item()


def test_pairs():
    # three prompts with four replies each
    texts = ["a", "b", "c", "d"] + ["e", "e", "f", "g"] + ["h", "i", "j", "k"]
    right = [True, False, True, False] + [True, True, True, True] + [False, False, True, False]
    # right ones first in each pair, and each reply used once
    assert dpo.pairs(texts, right, samples=4) == [(0, 1), (2, 3), (10, 8)]
    assert dpo.pairs(texts, right, samples=4, per_prompt=1) == [(0, 1), (10, 8)]
    # a prompt whose replies are all right gives nothing to prefer
    assert not [p for p in dpo.pairs(texts, right, samples=4) if 4 <= p[0] < 8]
    # the same text counts once: here only one wrong text is left to pair with
    assert dpo.pairs(["x", "y", "y", "z"], [True, False, False, True], samples=4) == [(0, 1)]
