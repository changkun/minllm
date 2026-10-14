import torch

from minllm import inference, speculative
from minllm.model import Config, Transformer


def models():
    torch.manual_seed(0)
    cfg = Config(vocab_size=64, max_seq_len=128, dim=32, n_layers=2, n_heads=2, n_kv_heads=1, mlp_dim=64)
    target = Transformer(cfg).eval()
    torch.manual_seed(1)
    draft = Transformer(Config(**{**vars(cfg), "dim": 16, "n_layers": 1, "mlp_dim": 32})).eval()
    # Larger weights make the random models sure of themselves, so that greedy decoding repeats and drafts are kept.
    with torch.no_grad():
        for m in (target, draft):
            m.embed.weight.mul_(20)
    return target, draft


def test_crop_forgets_and_appends_again():
    target, _ = models()
    tokens = torch.tensor([[5, 9, 2, 7, 1, 3]])
    cache = inference.KVCache(target.cfg.n_layers, 16)
    target.hidden(tokens[:, :5], cache=cache)
    cache.crop(3)
    assert cache.pos == 3
    h = target.hidden(tokens[:, 3:], cache=cache)  # the last three again, after the first three
    assert torch.allclose(h[0, -1], target.hidden(tokens)[0, -1], atol=1e-5)


def test_greedy_speculation_writes_what_the_model_writes():
    target, small = models()
    prompt = [3, 8, 1, 4, 3, 8, 1]
    alone = inference.generate(target, [prompt], 40, temperature=0.0)[0]
    for k in (1, 2, 4, 7):
        for draft in (speculative.ModelDraft(small, 128), speculative.LookupDraft(64)):
            new, stats = speculative.generate(target, draft, prompt, 40, k=k)
            assert new == alone, (k, type(draft).__name__)
            assert stats["passes"] + stats["kept"] == 40 and stats["kept"] <= stats["proposed"]


def test_a_draft_that_is_the_model_is_always_kept():
    target, _ = models()
    new, stats = speculative.generate(target, speculative.ModelDraft(target, 128), [1, 2, 3], 30, k=4)
    assert stats["kept"] == stats["proposed"] and stats["passes"] == 6  # 5 tokens a pass: 4 drafted, 1 added
    assert new == inference.generate(target, [[1, 2, 3]], 30, temperature=0.0)[0]


def test_stop_ends_the_continuation():
    target, small = models()
    alone = inference.generate(target, [[3, 8, 1]], 30, temperature=0.0)[0]
    stop = alone[5]
    new, _ = speculative.generate(target, speculative.ModelDraft(small, 128), [3, 8, 1], 30, k=4, stop=[stop])
    assert new == alone[: alone.index(stop)]
    assert new == inference.generate(target, [[3, 8, 1]], 30, temperature=0.0, stop=[stop])[0]  # the model alone, stopped the same way


def test_lookup_proposes_what_followed_the_last_repeat():
    draft = speculative.LookupDraft(64, ngram=2)
    tokens, q = draft.propose([5, 6, 7, 8, 9, 1, 5, 6], 3, 0.0)
    assert tokens == [7, 8, 9] and q.argmax(-1).tolist() == [7, 8, 9]
    assert draft.propose([1, 2, 3], 3, 0.0) == ([], None)  # nothing repeats


def test_sampling_keeps_the_models_distribution():
    # Draft tokens drawn from q, checked against p: the first token that comes
    # out must be distributed as p's first row, and the token after a fully
    # kept draft as its last row, whatever q is.
    g = torch.Generator().manual_seed(0)
    p = torch.softmax(torch.randn(3, 5, generator=g) * 2, -1)
    q = torch.softmax(torch.randn(2, 5, generator=g) * 2, -1)
    first, trials = torch.zeros(5), 40_000
    for _ in range(trials):
        tokens = [int(torch.multinomial(q[i], 1, generator=g)) for i in range(2)]
        kept, extra = speculative.accept(p, q, tokens, 1.0, g)
        first[(kept + [extra])[0]] += 1
    assert (first / trials - p[0]).abs().max() < 0.01
