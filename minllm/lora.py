"""
LoRA: low-rank adaptation (Hu et al., 2021, arXiv:2106.09685).

Fine-tuning changes a weight matrix W to W + dW. LoRA bets that dW does not
need all the freedom of a full matrix, and writes it as the product of two
thin ones, dW = B A, with A of shape (r, in) and B of shape (out, r). W stays
frozen and only A and B are trained: for r = 16 that is about 2% of the
parameters of the model here. See docs/06-finetuning.md.
"""

import torch
import torch.nn as nn

# The linear layers of a block, by their names in minllm/model.py.
LINEARS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


class LoRALinear(nn.Module):
    """A frozen linear layer plus a trainable low-rank update: W x + (alpha / r) B A x."""

    def __init__(self, base, r, alpha):
        super().__init__()
        self.base = base.requires_grad_(False)
        weight = base.weight
        # A starts random and B at zero, so B A = 0 and training starts from
        # exactly the pretrained model. A is scaled so that A x has about the
        # size of x: much larger, and the first steps on B overshoot.
        self.A = nn.Parameter(torch.randn(r, base.in_features, device=weight.device, dtype=weight.dtype) / base.in_features**0.5)
        self.B = nn.Parameter(torch.zeros(base.out_features, r, device=weight.device, dtype=weight.dtype))
        self.scale = alpha / r
        self.enabled = True  # off: the layer computes what the pretrained model computes

    def forward(self, x):
        y = self.base(x)
        if self.enabled:
            y = y + (x @ self.A.T) @ self.B.T * self.scale
        return y

    def merged(self):
        """The plain linear layer that computes the same thing: W + (alpha / r) B A."""
        self.base.weight.data += (self.B @ self.A * self.scale).to(self.base.weight.dtype)
        return self.base


def modules(model):
    return [m for m in model.modules() if isinstance(m, LoRALinear)]


def add(model, r=16, alpha=32, names=LINEARS):
    """Freeze the model and wrap the named linear layers of every block. Returns the new parameters."""
    model.requires_grad_(False)
    for layer in model.layers:
        for part in (layer.attn, layer.mlp):
            for name in names:
                if hasattr(part, name):
                    setattr(part, name, LoRALinear(getattr(part, name), r, alpha))
    return [p for m in modules(model) for p in (m.A, m.B)]


def enable(model, on=True):
    """Switch every adapter on or off. Off, the model is the pretrained one again."""
    for m in modules(model):
        m.enabled = on


def merge(model):
    """Fold every adapter into its weight matrix, leaving an ordinary model."""
    for layer in model.layers:
        for part in (layer.attn, layer.mlp):
            for name, child in list(part.named_children()):
                if isinstance(child, LoRALinear):
                    setattr(part, name, child.merged())
    return model


def state(model):
    """The adapter weights alone, which are all that training changed."""
    return {name: p.detach() for name, p in model.named_parameters() if name.endswith((".A", ".B"))}
