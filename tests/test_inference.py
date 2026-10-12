import torch

from minllm.inference import KVCache, generate, sample, stream
from minllm.model import Config, Transformer

TINY = Config(vocab_size=64, max_seq_len=48, dim=48, n_layers=2, n_heads=4, n_kv_heads=2, mlp_dim=96)


def model_and_tokens(seed=0, length=24):
    torch.manual_seed(seed)
    return Transformer(TINY).eval(), torch.randint(0, TINY.vocab_size, (2, length))


def test_cache_gives_the_same_logits():
    # Feeding the prompt and then one token at a time through the cache gives
    # the logits of a single pass over the whole sequence.
    model, idx = model_and_tokens()
    full, _ = model(idx)

    cache = KVCache(TINY.n_layers, max_len=idx.shape[1])
    logits = [model(idx[:, :10], cache=cache)[0]]
    for t in range(10, idx.shape[1]):
        logits.append(model(idx[:, [t]], cache=cache)[0])
    assert cache.pos == idx.shape[1]
    assert torch.allclose(torch.cat(logits, dim=1), full, atol=1e-5)


def test_cache_takes_several_tokens_at_a_time():
    model, idx = model_and_tokens()
    full, _ = model(idx)
    cache = KVCache(TINY.n_layers, max_len=idx.shape[1])
    chunks = [model(chunk, cache=cache)[0] for chunk in idx.split([7, 1, 9, 7], dim=1)]
    assert torch.allclose(torch.cat(chunks, dim=1), full, atol=1e-5)


def test_cache_stores_the_shared_heads_only():
    # With grouped-query attention the cache holds n_kv_heads heads, not n_heads.
    model, idx = model_and_tokens()
    cache = KVCache(TINY.n_layers, max_len=32)
    model(idx, cache=cache)
    head_dim = TINY.dim // TINY.n_heads
    assert cache.layers[0].k.shape == (2, TINY.n_kv_heads, 32, head_dim)
    assert cache.nbytes == TINY.n_layers * 2 * (2 * TINY.n_kv_heads * 32 * head_dim) * 4


def test_generate_matches_the_uncached_sampler():
    model, idx = model_and_tokens(length=6)
    expected = model.generate(idx, max_new_tokens=30, temperature=0)[:, 6:]
    assert generate(model, idx.tolist(), max_new_tokens=30, temperature=0) == expected.tolist()


def test_prompts_of_different_lengths():
    # A batch of unequal prompts is padded on the left. Every row must come
    # out as if it had been generated alone.
    model, idx = model_and_tokens()
    prompts = [idx[0, :3].tolist(), idx[1, :11].tolist(), idx[0, :7].tolist()]
    together = generate(model, prompts, max_new_tokens=20, temperature=0)
    alone = [generate(model, [p], max_new_tokens=20, temperature=0)[0] for p in prompts]
    assert together == alone


def test_stop_tokens():
    model, idx = model_and_tokens(length=4)
    free = generate(model, idx.tolist(), max_new_tokens=30, temperature=0)
    stop = free[0][5]  # a token that row 0 is known to produce
    cut = generate(model, idx.tolist(), max_new_tokens=30, temperature=0, stop=[stop])
    for row_free, row_cut in zip(free, cut):
        end = row_free.index(stop) if stop in row_free else len(row_free)
        assert row_cut == row_free[:end]
    assert len(cut[0]) <= 5

    # The stream marks a finished row with -1 and ends once all rows are done.
    steps = list(stream(model, [idx[0].tolist()], max_new_tokens=30, temperature=0, stop=[stop]))
    assert steps[-1].tolist() == [-1] and len(steps) == len(cut[0]) + 1


def test_generation_stops_when_the_context_is_full():
    model, idx = model_and_tokens(length=40)
    out = generate(model, idx.tolist(), max_new_tokens=100, temperature=0)
    assert [len(row) for row in out] == [TINY.max_seq_len - 40] * 2


def test_sample_temperature_zero_is_argmax():
    logits = torch.randn(5, 64)
    assert torch.equal(sample(logits, temperature=0), logits.argmax(-1))


def test_sample_top_k_and_top_p():
    torch.manual_seed(0)
    probs = torch.tensor([0.05, 0.5, 0.15, 0.3]).repeat(2000, 1)
    logits = probs.log()

    assert set(sample(logits).tolist()) == {0, 1, 2, 3}
    assert set(sample(logits, top_k=2).tolist()) == {1, 3}
    # The two likeliest tokens sum to 0.8: enough for p = 0.79, not for p = 0.81.
    assert set(sample(logits, top_p=0.79).tolist()) == {1, 3}
    assert set(sample(logits, top_p=0.81).tolist()) == {1, 2, 3}
    assert set(sample(logits, top_p=0.01).tolist()) == {1}  # the likeliest token always stays

    # A low temperature concentrates the samples on the likeliest token.
    assert (sample(logits, temperature=0.2) == 1).float().mean() > 0.9


def test_allow_restricts_every_step_and_sees_the_last_pick():
    model, idx = model_and_tokens()
    prompts = [idx[0, :5].tolist(), idx[1, :8].tolist()]
    seen = []

    def allow(previous):
        # even tokens for the first row, odd ones for the second, and never a token twice in a row
        seen.append(None if previous is None else previous.tolist())
        ok = torch.zeros(2, TINY.vocab_size, dtype=torch.bool)
        ok[0, 0::2] = ok[1, 1::2] = True
        if previous is not None:
            ok[torch.arange(2), previous.cpu()] = False
        return ok

    out = generate(model, prompts, 6, temperature=1.0, allow=allow)
    assert all(t % 2 == 0 for t in out[0]) and all(t % 2 == 1 for t in out[1])
    assert all(a != b for row in out for a, b in zip(row, row[1:]))
    assert seen[0] is None and seen[1] == [out[0][0], out[1][0]]  # each call gets the step before it
    # with everything allowed, generation is what it was
    free = lambda previous: torch.ones(2, TINY.vocab_size, dtype=torch.bool)
    assert generate(model, prompts, 6, temperature=0, allow=free) == generate(model, prompts, 6, temperature=0)
