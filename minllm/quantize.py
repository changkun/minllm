"""
Quantization: a weight in a few bits instead of sixteen.

A trained weight matrix is stored as small whole numbers, the codes, together
with the two numbers that turn a code back into a weight:

    weight = code * scale + offset

With 4 bits a code is one of 16 values. The weights of a row are cut into
groups, and every group has a scale and an offset of its own, chosen so that
its 16 values reach from its smallest weight to its largest. Nothing is
trained: the model is rounded after the fact. See docs/10a-quantization.md.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def quantize(w, bits, group=0, dtype=torch.float32):
    """
    Round a matrix (rows, columns) to 2^bits values per group of `group`
    consecutive weights in a row, or per whole row if group is 0. Returns the
    codes, integers from 0 to 2^bits - 1 in the shape of w, and each group's
    scale and offset, both (rows, columns / group) and of the given dtype.
    """
    groups = w.detach().float().cpu().reshape(w.shape[0], -1, group or w.shape[1])
    low, high = groups.amin(-1, keepdim=True), groups.amax(-1, keepdim=True)
    scale = (high - low) / (2**bits - 1)
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)  # a group of equal weights: any scale will do
    # The codes are chosen against the scale and offset as they will be stored.
    scale, low = scale.to(dtype), low.to(dtype)
    codes = ((groups - low.float()) / scale.float()).round().clamp(0, 2**bits - 1).to(torch.uint8)
    return codes.reshape(w.shape), scale.squeeze(-1), low.squeeze(-1)


def dequantize(codes, scale, offset):
    """The weights that the codes stand for, in the dtype of the scale."""
    groups = codes.reshape(*scale.shape, -1).to(scale.dtype)
    return (groups * scale.unsqueeze(-1) + offset.unsqueeze(-1)).reshape(codes.shape)


def pack(codes, bits):
    """
    Store 32 / bits codes in every 32-bit word, the first code in the lowest
    bits. This is the layout MLX reads. The words are unsigned, and held here
    as int32, which has the same bits and which PyTorch can compute with.
    """
    assert 32 % bits == 0 and codes.shape[-1] % (32 // bits) == 0
    per_word = 32 // bits
    parts = codes.reshape(*codes.shape[:-1], -1, per_word).to(torch.int64)
    shifts = bits * torch.arange(per_word, device=codes.device)
    return (parts << shifts).sum(-1).to(torch.int32)  # the conversion wraps a set top bit into the sign


def unpack(words, bits):
    shifts = bits * torch.arange(32 // bits, device=words.device, dtype=torch.int32)
    return ((words.unsqueeze(-1) >> shifts) & (2**bits - 1)).flatten(-2)


class Quantized(nn.Module):
    """A matrix kept as packed codes, with the scale and offset of every group."""

    def __init__(self, shape, bits, group, dtype=torch.float16):
        super().__init__()
        self.bits, self.group = bits, group or shape[1]
        self.register_buffer("words", torch.zeros(shape[0], shape[1] * bits // 32, dtype=torch.int32))
        self.register_buffer("scale", torch.ones(shape[0], shape[1] // self.group, dtype=dtype))
        self.register_buffer("offset", torch.zeros(shape[0], shape[1] // self.group, dtype=dtype))

    def set(self, w):
        codes, scale, offset = quantize(w, self.bits, self.group, self.scale.dtype)
        self.words.copy_(pack(codes, self.bits))
        self.scale.copy_(scale)
        self.offset.copy_(offset)
        return self

    @property
    def weight(self):
        """The whole matrix, expanded. It exists only for as long as the caller keeps it."""
        return dequantize(unpack(self.words, self.bits), self.scale, self.offset)


class QuantLinear(Quantized):
    """A linear layer that expands its weight for every forward pass."""

    def __init__(self, linear, bits, group):
        super().__init__(linear.weight.shape, bits, group)
        self.out_features, self.in_features = linear.weight.shape
        self.bias = linear.bias

    def forward(self, x):
        return F.linear(x, self.weight.to(x.dtype), self.bias)


class QuantEmbedding(Quantized):
    """An embedding table that expands only the rows that are looked up."""

    def forward(self, idx):
        return dequantize(unpack(self.words[idx], self.bits), self.scale[idx], self.offset[idx])


class TiedHead(nn.Module):
    """The output layer of a model with tied embeddings: the quantized table again, used as a linear layer."""

    def __init__(self, embed):
        super().__init__()
        self.__dict__["embed"] = embed  # a reference and not a submodule, so that the table is stored and moved once

    @property
    def weight(self):
        return self.embed.weight

    def forward(self, x):
        return F.linear(x, self.weight.to(x.dtype))


def apply(model, bits, group=64, embed_bits=None, fill=True):
    """
    Replace the model's matrices by quantized ones, in place: every linear
    layer of the blocks at `bits`, and the embedding table with the output
    layer at `embed_bits`, which defaults to `bits`. With embed_bits 0 the
    table stays as it is. fill=False leaves the codes empty, to be loaded.
    """
    embed_bits = bits if embed_bits is None else embed_bits
    for layer in model.layers:
        for module in list(layer.modules()):
            for name, child in list(module.named_children()):
                if isinstance(child, nn.Linear):
                    quantized = QuantLinear(child, bits, group)
                    setattr(module, name, quantized.set(child.weight.data) if fill else quantized)
    if embed_bits:
        assert model.cfg.tie_embeddings, "only a tied embedding table is handled"
        table = model.embed.weight.data
        model.embed = QuantEmbedding(table.shape, embed_bits, group)
        if fill:
            model.embed.set(table)
        model.lm_head = TiedHead(model.embed)
    return model


def rounded(model, bits, group=64, embed_bits=None):
    """
    What apply does to the weights, without the packing: every matrix is
    replaced by its rounded values, at full size and in place. This measures
    what a bit width costs, also for the widths that do not pack evenly into
    32-bit words. bits 0 leaves the blocks as they are.
    """
    embed_bits = bits if embed_bits is None else embed_bits
    for name, p in model.named_parameters():
        width = embed_bits if name.startswith(("embed", "lm_head")) else bits
        if p.dim() == 2 and width:
            codes, scale, offset = quantize(p.data, width, group, torch.float16)
            p.data.copy_(dequantize(codes, scale.float(), offset.float()))
    return model


def expand(model):
    """
    Put ordinary layers back, holding the rounded weights at full size. The
    model computes what the quantized one computes, as fast as before and in
    as much memory as before: the form to evaluate in.
    """
    for module in list(model.modules()):
        for name, child in list(module.named_children()):
            if isinstance(child, QuantLinear):
                linear = nn.Linear(child.in_features, child.out_features, bias=child.bias is not None, device=child.words.device)
                linear.weight.data, linear.bias = child.weight.float(), child.bias
                setattr(module, name, linear)
    if isinstance(model.embed, QuantEmbedding):
        table = model.embed.weight.float()
        model.embed = nn.Embedding(*table.shape, device=table.device)
        model.lm_head = nn.Linear(table.shape[1], table.shape[0], bias=False, device=table.device)
        model.embed.weight.data = table
        model.lm_head.weight = model.embed.weight
    return model.requires_grad_(False)


def nbytes(model):
    """The bytes that the model's parameters and buffers take, a tied matrix counted once."""
    tensors = {id(t): t for t in list(model.parameters()) + list(model.buffers())}
    return sum(t.numel() * t.element_size() for t in tensors.values())
