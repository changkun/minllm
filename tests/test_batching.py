import torch

from minllm import batching, inference
from minllm.model import Config, Transformer

TINY = Config(vocab_size=64, max_seq_len=48, dim=48, n_layers=2, n_heads=4, n_kv_heads=2, mlp_dim=96)


def setup(cfg=TINY, seed=0):
    torch.manual_seed(seed)
    model = Transformer(cfg).eval()
    prompts = [torch.randint(0, cfg.vocab_size, (n,)).tolist() for n in (3, 11, 7, 5, 9)]
    return model, prompts


def alone(model, prompt, n, **sampling):
    return inference.generate(model, [prompt], n, temperature=0, **sampling)[0]


def test_a_running_batch_gives_what_each_request_gives_alone():
    # Five requests through two slots: three have to wait for a slot, and
    # join a batch that is already running, at positions of their own.
    model, prompts = setup()
    assert batching.generate(model, prompts, 20, slots=2) == [alone(model, p, 20) for p in prompts]
    assert batching.generate(model, prompts, 20, slots=8) == [alone(model, p, 20) for p in prompts]


def test_requests_of_different_lengths_leave_at_different_steps():
    model, prompts = setup()
    batcher = batching.Batcher(model, slots=2)
    limits = [4, 15, 9, 2, 12]
    ids = [batcher.add(p, n) for p, n in zip(prompts, limits)]
    out, finished, steps = {i: [] for i in ids}, [], 0
    while batcher.busy():
        for request, token, done in batcher.step():
            out[request].append(token)
            if done:
                finished.append(request)
        assert sum(r is not None for r in batcher.running) <= 2
        steps += 1
    assert [out[i] for i in ids] == [alone(model, p, n) for p, n in zip(prompts, limits)]
    # The first request is short and leaves first; the third joins in its slot and leaves before the second, which is long.
    assert finished[0] == ids[0] and finished.index(ids[2]) < finished.index(ids[1])
    assert steps < sum(limits)  # requests ran side by side


def test_stop_tokens_free_the_slot():
    model, prompts = setup()
    free = alone(model, prompts[0], 30)
    stop = free[5]  # a token that the first request is known to produce
    out = batching.generate(model, prompts, 30, slots=2, stop=[stop])
    for p, row in zip(prompts, out):
        expected = alone(model, p, 30, stop=[stop])
        assert row == expected and stop not in row


def test_the_context_bounds_every_request():
    model, _ = setup()
    long = torch.randint(0, TINY.vocab_size, (40,)).tolist()
    short = [1, 2, 3]
    out = batching.generate(model, [long, short], 100, slots=2)
    assert out == [alone(model, long, 100), alone(model, short, 100)]
    assert len(out[0]) == TINY.max_seq_len - 40


def test_latent_attention_in_a_running_batch():
    cfg = Config(**vars(TINY) | {"kv_latent": 16, "rope_dim": 8})
    model, prompts = setup(cfg)
    assert batching.generate(model, prompts, 20, slots=2) == [alone(model, p, 20) for p in prompts]


def test_every_request_samples_at_its_own_temperature():
    model, prompts = setup()
    batcher = batching.Batcher(model, slots=3, temperature=0.0)
    greedy = batcher.add(prompts[0], 20)
    hot = batcher.add(prompts[0], 20, temperature=5.0)
    nucleus = batcher.add(prompts[0], 20, temperature=1.0, top_p=1e-6)  # only the likeliest token is left
    out = {i: [] for i in (greedy, hot, nucleus)}
    torch.manual_seed(0)
    while batcher.busy():
        for request, token, _ in batcher.step():
            out[request].append(token)
    assert out[greedy] == out[nucleus] == alone(model, prompts[0], 20)
    assert out[hot] != out[greedy]
