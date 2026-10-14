import torch

from minllm.inference import KVCache, generate
from minllm.model import Config, LatentAttention, Transformer

TINY = Config(vocab_size=64, max_seq_len=48, dim=48, n_layers=2, n_heads=4, n_kv_heads=2, mlp_dim=96, kv_latent=16, rope_dim=8)


def model_and_tokens(seed=0, length=24):
    torch.manual_seed(seed)
    model = Transformer(TINY).eval()
    # Trained weights are not this small. Larger ones make the tests below
    # sensitive to a wrong position or a wrong projection.
    with torch.no_grad():
        for p in model.parameters():
            if p.dim() == 2:
                p.mul_(4)
    return model, torch.randint(0, TINY.vocab_size, (2, length))


def test_every_block_has_latent_attention():
    model, _ = model_and_tokens()
    assert all(isinstance(layer.attn, LatentAttention) for layer in model.layers)
    attn = model.layers[0].attn
    head_dim = TINY.dim // TINY.n_heads
    assert attn.q_proj.weight.shape == (TINY.n_heads * (head_dim + TINY.rope_dim), TINY.dim)
    assert attn.kv_down.weight.shape == (TINY.kv_latent + TINY.rope_dim, TINY.dim)
    assert attn.k_up.weight.shape == attn.v_up.weight.shape == (TINY.n_heads * head_dim, TINY.kv_latent)
    assert model.rope_cos.shape == (TINY.max_seq_len, TINY.rope_dim // 2)


def test_causal():
    model, idx = model_and_tokens()
    changed = idx.clone()
    changed[:, 15:] = (changed[:, 15:] + 1) % TINY.vocab_size
    a, b = model(idx)[0], model(changed)[0]
    assert torch.allclose(a[:, :15], b[:, :15], atol=1e-5) and not torch.allclose(a[:, 15:], b[:, 15:], atol=1e-3)


def test_cache_holds_the_latent_and_the_rotary_key():
    model, idx = model_and_tokens()
    cache = KVCache(TINY.n_layers, max_len=32)
    model(idx, cache=cache)
    assert cache.layers[0].k.shape == (2, 1, 32, TINY.kv_latent)  # the latent, one per token, for all heads
    assert cache.layers[0].v.shape == (2, 1, 32, TINY.rope_dim)  # the rotary key, shared too
    assert cache.nbytes == TINY.n_layers * 2 * 32 * (TINY.kv_latent + TINY.rope_dim) * 4


def test_cache_gives_the_same_logits():
    # Prefill, then one token at a time: decoding against the latent
    # itself gives the logits of a single pass, which projects it up.
    model, idx = model_and_tokens()
    full, _ = model(idx)
    for absorb in (True, False):
        for layer in model.layers:
            layer.attn.absorb = absorb
        cache = KVCache(TINY.n_layers, max_len=idx.shape[1])
        logits = [model(idx[:, :10], cache=cache)[0]]
        for t in range(10, idx.shape[1]):
            logits.append(model(idx[:, [t]], cache=cache)[0])
        assert torch.allclose(torch.cat(logits, dim=1), full, atol=1e-4)


def test_cache_takes_several_tokens_at_a_time():
    model, idx = model_and_tokens()
    full, _ = model(idx)
    cache = KVCache(TINY.n_layers, max_len=idx.shape[1])
    chunks = [model(chunk, cache=cache)[0] for chunk in idx.split([7, 1, 9, 7], dim=1)]
    assert torch.allclose(torch.cat(chunks, dim=1), full, atol=1e-4)


def test_prompts_of_different_lengths():
    model, idx = model_and_tokens()
    prompts = [idx[0, :3].tolist(), idx[1, :11].tolist(), idx[0, :7].tolist()]
    together = generate(model, prompts, max_new_tokens=20, temperature=0)
    alone = [generate(model, [p], max_new_tokens=20, temperature=0)[0] for p in prompts]
    assert together == alone


def test_overfits_one_batch():
    torch.manual_seed(0)
    model = Transformer(TINY)
    idx = torch.randint(0, TINY.vocab_size, (4, 32))
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    for _ in range(150):
        _, loss = model(idx[:, :-1], idx[:, 1:])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    assert loss.item() < 0.1
