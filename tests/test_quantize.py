import torch
import torch.nn as nn

from minllm import lora, quantize
from minllm.model import Config, Transformer

TINY = Config(vocab_size=64, max_seq_len=32, dim=64, n_layers=2, n_heads=4, n_kv_heads=2, mlp_dim=128)


def test_error_is_at_most_half_a_step():
    # Every group's values reach from its smallest weight to its largest, so
    # rounding moves a weight by at most half the distance between two values.
    torch.manual_seed(0)
    w = torch.randn(8, 128)
    for bits in (2, 4, 8):
        for group in (0, 64, 16):
            codes, scale, offset = quantize.quantize(w, bits, group)
            assert codes.max() == 2**bits - 1 and codes.min() == 0
            error = (quantize.dequantize(codes, scale, offset) - w).abs().reshape(*scale.shape, -1)
            assert (error <= scale.unsqueeze(-1) / 2 + 1e-6).all()
            # the smallest and the largest weight of a group are kept exactly
            ends = w.reshape(*scale.shape, -1)
            assert torch.allclose(quantize.dequantize(codes, scale, offset).reshape(*scale.shape, -1).amax(-1), ends.amax(-1), atol=1e-5)


def test_more_bits_and_smaller_groups_lose_less():
    torch.manual_seed(0)
    w = torch.randn(16, 256)
    error = lambda bits, group: (quantize.dequantize(*quantize.quantize(w, bits, group)) - w).pow(2).mean().item()
    assert error(8, 64) < error(4, 64) < error(2, 64)
    assert error(4, 16) < error(4, 64) < error(4, 0)


def test_a_group_of_equal_weights():
    w = torch.cat([torch.zeros(2, 64), torch.full((2, 64), 0.3)], dim=1)
    codes, scale, offset = quantize.quantize(w, 4, 64)
    assert torch.allclose(quantize.dequantize(codes, scale, offset), w)


def test_pack_and_unpack():
    torch.manual_seed(0)
    for bits in (2, 4, 8):
        codes = torch.randint(0, 2**bits, (5, 64), dtype=torch.uint8)
        words = quantize.pack(codes, bits)
        assert words.dtype == torch.int32 and words.shape == (5, 64 * bits // 32)
        assert torch.equal(quantize.unpack(words, bits), codes.int())
    # the first code sits in the lowest bits, and a set top bit survives as the sign
    assert quantize.pack(torch.tensor([[1, 2, 3, 4, 5, 6, 7, 15]], dtype=torch.uint8), 4).item() == 0xF7654321 - (1 << 32)


def test_quantized_model():
    torch.manual_seed(0)
    model = Transformer(TINY).eval()
    idx = torch.randint(0, TINY.vocab_size, (2, 16))
    reference = model(idx)[0]
    size = quantize.nbytes(model)

    eight = quantize.apply(Transformer(TINY).eval(), 8)
    eight.load_state_dict(quantize.apply(model, 8).state_dict())  # the state alone rebuilds it
    assert not any(isinstance(m, (nn.Linear, nn.Embedding)) for m in eight.modules())
    assert "lm_head.words" not in eight.state_dict()  # the tied table is stored once
    close = (eight(idx)[0] - reference).abs().max()
    assert close < 0.02 * reference.abs().max()
    assert quantize.nbytes(eight) < 0.33 * size

    torch.manual_seed(0)
    four = quantize.apply(Transformer(TINY).eval(), 4)
    assert close < (four(idx)[0] - reference).abs().max() < 0.3 * reference.abs().max()
    assert quantize.nbytes(four) < 0.2 * size

    # Expanded, it computes the same with ordinary layers.
    logits = four(idx)[0]
    expanded = quantize.expand(four)
    assert isinstance(expanded.lm_head, nn.Linear) and expanded.lm_head.weight is expanded.embed.weight
    assert torch.allclose(expanded(idx)[0], logits, atol=1e-3)


def test_embedding_can_stay():
    torch.manual_seed(0)
    model = quantize.apply(Transformer(TINY), 4, embed_bits=0)
    assert isinstance(model.embed, nn.Embedding) and model.lm_head.weight is model.embed.weight
    assert isinstance(model.layers[0].attn.q_proj, quantize.QuantLinear)


def test_adapters_on_a_quantized_model():
    # LoRA on top of quantized weights: the codes stay fixed and the adapters learn.
    torch.manual_seed(0)
    model = quantize.apply(Transformer(TINY), 4)
    trainable = lora.add(model, r=4, alpha=8)
    idx = torch.randint(0, TINY.vocab_size, (2, 16))
    model(idx, idx)[1].backward()
    assert all(p.grad is not None for p in trainable) and len(trainable) == 2 * 7 * TINY.n_layers


def test_rounded_is_the_quantized_model_at_full_size():
    torch.manual_seed(0)
    model = Transformer(TINY).eval()
    idx = torch.randint(0, TINY.vocab_size, (2, 16))
    state = {k: v.clone() for k, v in model.state_dict().items()}
    packed = quantize.apply(model, 4)(idx)[0]
    plain = Transformer(TINY).eval()
    plain.load_state_dict(state)
    assert torch.allclose(quantize.rounded(plain, 4)(idx)[0], packed, atol=1e-3)
    assert plain.lm_head.weight is plain.embed.weight
