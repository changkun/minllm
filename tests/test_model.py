import math

import torch
import torch.nn.functional as F

from minllm.model import Attention, Config, RMSNorm, Transformer, apply_rope, rope_table

TINY = Config(vocab_size=64, max_seq_len=32, dim=48, n_layers=2, n_heads=4, n_kv_heads=2, mlp_dim=96)


def test_shapes_and_params():
    model = Transformer(TINY)
    idx = torch.randint(0, TINY.vocab_size, (3, 10))
    logits, loss = model(idx)
    assert logits.shape == (3, 10, TINY.vocab_size) and loss is None

    # Count the parameters by hand.
    head_dim = TINY.dim // TINY.n_heads
    attn = TINY.dim * head_dim * (2 * TINY.n_heads + 2 * TINY.n_kv_heads)  # q, o and k, v
    mlp = 3 * TINY.dim * TINY.mlp_dim
    norms = 2 * TINY.dim
    expected = TINY.vocab_size * TINY.dim + TINY.n_layers * (attn + mlp + norms) + TINY.dim
    assert model.num_params() == expected


def test_untrained_model_is_uniform():
    # Before any training, every token is about equally likely, so the loss
    # is close to that of a uniform guess, ln(vocab_size).
    torch.manual_seed(0)
    model = Transformer(TINY)
    idx = torch.randint(0, TINY.vocab_size, (8, 32))
    targets = torch.randint(0, TINY.vocab_size, (8, 32))
    _, loss = model(idx, targets)
    assert abs(loss.item() - math.log(TINY.vocab_size)) < 0.05


def test_ignored_targets():
    torch.manual_seed(0)
    model = Transformer(TINY)
    idx = torch.randint(0, TINY.vocab_size, (2, 8))
    targets = torch.randint(0, TINY.vocab_size, (2, 8))
    masked = targets.clone()
    masked[:, :4] = -100
    logits, loss = model(idx, masked)
    expected = F.cross_entropy(logits[:, 4:].reshape(-1, TINY.vocab_size), targets[:, 4:].reshape(-1))
    assert torch.allclose(loss, expected)


def test_causal():
    # Changing a token must not change the predictions made before it.
    torch.manual_seed(0)
    model = Transformer(TINY).eval()
    a = torch.randint(0, TINY.vocab_size, (1, 16))
    b = a.clone()
    b[0, 10] = (b[0, 10] + 1) % TINY.vocab_size
    la, lb = model(a)[0], model(b)[0]
    assert torch.allclose(la[0, :10], lb[0, :10], atol=1e-6)
    assert not torch.allclose(la[0, 10:], lb[0, 10:], atol=1e-6)


def test_rmsnorm():
    x = torch.randn(4, 16) * 7
    y = RMSNorm(16, eps=1e-6)(x)
    assert torch.allclose(y.pow(2).mean(-1).sqrt(), torch.ones(4), atol=1e-4)


def test_rope_is_relative():
    # Rotations keep lengths, and the score between a query at position m and
    # a key at position n depends only on m - n.
    cos, sin = rope_table(head_dim=16, seq_len=64, theta=10000.0)
    q, k = torch.randn(16), torch.randn(16)
    at = lambda x, pos: apply_rope(x, cos[pos], sin[pos])
    assert torch.allclose(at(q, 7).norm(), q.norm(), atol=1e-5)
    assert torch.allclose(at(q, 0), q)
    assert torch.allclose(at(q, 5) @ at(k, 2), at(q, 45) @ at(k, 42), atol=1e-4)
    assert not torch.allclose(at(q, 5) @ at(k, 2), at(q, 5) @ at(k, 3), atol=1e-4)


def test_attention_matches_the_formula():
    # The same computation as Attention.forward, spelled out without the
    # library call: softmax(q k^T / sqrt(d) + mask) v.
    torch.manual_seed(0)
    attn = Attention(TINY)
    B, T, H, KV, D = 2, 9, TINY.n_heads, TINY.n_kv_heads, TINY.dim // TINY.n_heads
    x = torch.randn(B, T, TINY.dim)
    cos, sin = rope_table(D, T, TINY.rope_theta)

    q = apply_rope(attn.q_proj(x).view(B, T, H, D).transpose(1, 2), cos, sin)
    k = apply_rope(attn.k_proj(x).view(B, T, KV, D).transpose(1, 2), cos, sin)
    v = attn.v_proj(x).view(B, T, KV, D).transpose(1, 2)
    out = torch.zeros(B, H, T, D)
    for h in range(H):
        g = h // (H // KV)  # the key/value head that query head h shares
        scores = q[:, h] @ k[:, g].transpose(-2, -1) / math.sqrt(D)  # (B, T, T)
        future = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
        weights = F.softmax(scores.masked_fill(future, float("-inf")), dim=-1)
        out[:, h] = weights @ v[:, g]
    expected = attn.o_proj(out.transpose(1, 2).reshape(B, T, -1))

    assert torch.allclose(attn(x, cos, sin), expected, atol=1e-5)


def test_overfits_one_batch():
    # The first thing to check about any model and training loop: it can
    # memorize a handful of examples.
    torch.manual_seed(0)
    model = Transformer(TINY)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    data = torch.randint(0, TINY.vocab_size, (4, 33))
    x, y = data[:, :-1], data[:, 1:]
    for _ in range(200):
        _, loss = model(x, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.1


def test_generate():
    torch.manual_seed(0)
    model = Transformer(TINY).eval()
    prompt = torch.randint(0, TINY.vocab_size, (2, 5))
    out = model.generate(prompt, max_new_tokens=40, top_k=10)  # runs past max_seq_len
    assert out.shape == (2, 45) and torch.equal(out[:, :5], prompt)
    greedy = model.generate(prompt, max_new_tokens=8, temperature=0)
    assert torch.equal(greedy, model.generate(prompt, max_new_tokens=8, temperature=0))
