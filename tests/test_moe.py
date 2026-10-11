import math
from dataclasses import replace

import torch

from minllm.inference import KVCache
from minllm.model import Config, MoE, Transformer

TINY = Config(vocab_size=64, max_seq_len=32, dim=48, n_layers=2, n_heads=4, n_kv_heads=2, mlp_dim=40, n_experts=4, n_active=2)


def by_the_formula(moe, x, weights):
    """Every expert runs on every token, and the results are added by weight."""
    return sum(weights[:, [i]] * expert(x) for i, expert in enumerate(moe.experts))


def moe_and_tokens(tokens=60):
    torch.manual_seed(0)
    return MoE(TINY), torch.randn(tokens, TINY.dim)


def test_parameters():
    dense, model = Transformer(replace(TINY, n_experts=0)), Transformer(TINY)
    mlp = 3 * TINY.dim * TINY.mlp_dim
    router = TINY.dim * TINY.n_experts
    assert model.num_params() == dense.num_params() + TINY.n_layers * ((TINY.n_experts - 1) * mlp + router)
    assert not dense.mixtures() and len(model.mixtures()) == TINY.n_layers


def test_routing_weights():
    # Each token gives weight to n_active experts, the ones with the largest
    # router scores, and the weights are a softmax over those scores alone.
    moe, x = moe_and_tokens()
    weights = moe.route(x)
    assert weights.shape == (60, TINY.n_experts)
    assert ((weights > 0).sum(-1) == TINY.n_active).all()
    top, chosen = moe.router(x).topk(TINY.n_active, dim=-1)
    assert torch.allclose(weights.gather(-1, chosen), top.softmax(-1), atol=1e-6)
    assert torch.allclose(moe.share.sum(), torch.tensor(1.0))


def test_forward_matches_the_formula():
    moe, x = moe_and_tokens()
    expected = by_the_formula(moe, x, moe.route(x))
    assert torch.allclose(moe(x), expected, atol=1e-6)
    assert torch.allclose(moe(x.view(3, 20, -1)), expected.view(3, 20, -1), atol=1e-6)
    # With room for every token, the fixed-size path computes the same.
    moe.capacity = TINY.n_experts / TINY.n_active
    assert torch.allclose(moe(x), expected, atol=1e-6) and moe.dropped == 0


def test_one_expert_is_the_mlp():
    torch.manual_seed(0)
    moe, x = MoE(replace(TINY, n_experts=1, n_active=1)), torch.randn(10, TINY.dim)
    assert torch.allclose(moe(x), moe.experts[0](x), atol=1e-6)


def test_capacity_drops_the_overflow():
    # 60 tokens with two experts each are 120 pieces of work, 30 per expert if
    # spread evenly. At capacity 1 an expert takes exactly 30 tokens: of the
    # ones routed to it, those that weigh it most.
    moe, x = moe_and_tokens()
    weights = moe.route(x)
    kept = torch.zeros_like(weights)
    for i in range(TINY.n_experts):
        ranked = weights[:, i].argsort(descending=True)[:30]
        kept[ranked, i] = weights[ranked, i]
    lost = ((weights > 0) & (kept == 0)).sum()
    assert lost > 0
    moe.capacity = 1.0
    assert torch.allclose(moe(x), by_the_formula(moe, x, kept), atol=1e-6)
    assert torch.isclose(moe.dropped, lost / (weights > 0).sum())


def test_balance_loss():
    # Four experts, and a router that sends each token to the two experts
    # named by its two nonzero coordinates.
    moe = MoE(replace(TINY, dim=4))
    moe.router.weight.data = 10 * torch.eye(4)
    pairs = lambda firsts: torch.stack([torch.eye(4)[i] + torch.eye(4)[(i + 1) % 4] for i in firsts])
    moe.route(pairs([0, 1, 2, 3] * 5))  # every expert gets a quarter of the work
    assert torch.allclose(moe.share, torch.full((4,), 0.25)) and torch.isclose(moe.balance, torch.tensor(1.0), atol=1e-3)
    moe.route(pairs([0] * 20))  # two experts get all of it
    assert torch.allclose(moe.share, torch.tensor([0.5, 0.5, 0, 0])) and torch.isclose(moe.balance, torch.tensor(2.0), atol=1e-3)


def test_balance_loss_spreads_the_tokens():
    # Following the gradient of the balance loss evens the routing out.
    torch.manual_seed(0)
    moe = MoE(TINY)
    x = torch.randn(200, TINY.dim) + 1.0  # a shared direction, so that the router has favorites
    moe.route(x)
    assert moe.share.max() > 0.4 and moe.balance > 1.2
    for _ in range(20):
        moe.router.weight.grad = None
        moe.balance.backward()
        with torch.no_grad():
            moe.router.weight -= 0.01 * moe.router.weight.grad
        moe.route(x)
    assert moe.share.max() < 0.26 and moe.balance < 1.01


def test_untrained_model_routes_evenly():
    torch.manual_seed(0)
    model = Transformer(TINY)
    idx = torch.randint(0, TINY.vocab_size, (8, 32))
    targets = torch.randint(0, TINY.vocab_size, (8, 32))
    _, loss = model(idx, targets)
    assert abs(loss.item() - math.log(TINY.vocab_size)) < 0.05
    assert abs(model.balance_loss().item() - 1) < 0.01


def test_overfits_one_batch():
    torch.manual_seed(0)
    model = Transformer(TINY)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    data = torch.randint(0, TINY.vocab_size, (4, 33))
    x, y = data[:, :-1], data[:, 1:]
    for _ in range(200):
        _, loss = model(x, y)
        opt.zero_grad()
        (loss + 0.01 * model.balance_loss()).backward()
        opt.step()
    assert loss.item() < 0.1
    assert all(m.router.weight.grad.abs().sum() > 0 for m in model.mixtures())


def test_causal_unless_tokens_compete_for_room():
    torch.manual_seed(0)
    model = Transformer(TINY).eval()
    a = torch.randint(0, TINY.vocab_size, (1, 16))
    b = a.clone()
    b[0, 10] = (b[0, 10] + 1) % TINY.vocab_size
    la, lb = model(a)[0], model(b)[0]
    assert torch.allclose(la[0, :10], lb[0, :10], atol=1e-6)
    # With a capacity, which tokens an expert drops depends on all the tokens
    # of the batch, later ones included. That is why only training uses one.
    for m in model.mixtures():
        m.capacity = 0.5
    la, lb = model(a)[0], model(b)[0]
    assert not torch.allclose(la[0, :10], lb[0, :10], atol=1e-6)


def test_cache_gives_the_same_logits():
    torch.manual_seed(0)
    model = Transformer(TINY).eval()
    idx = torch.randint(0, TINY.vocab_size, (2, 24))
    full, _ = model(idx)
    cache = KVCache(TINY.n_layers, max_len=24)
    logits = [model(idx[:, :10], cache=cache)[0]] + [model(idx[:, [t]], cache=cache)[0] for t in range(10, 24)]
    assert torch.allclose(torch.cat(logits, dim=1), full, atol=1e-5)


def test_export_in_the_layout_of_mixtral():
    # The reference class keeps the router as "gate" and stacks a block's
    # experts into one tensor per matrix, gate_proj on top of up_proj.
    from minllm import export

    torch.manual_seed(0)
    model = Transformer(TINY)
    state = export.stack_experts({export.to_reference(name): t for name, t in model.state_dict().items()}, TINY)
    mlp, stacked = model.layers[1].mlp, state["model.layers.1.mlp.experts.gate_up_proj"]
    assert stacked.shape == (TINY.n_experts, 2 * TINY.mlp_dim, TINY.dim)
    assert torch.equal(stacked[2, : TINY.mlp_dim], mlp.experts[2].gate_proj.weight)
    assert torch.equal(stacked[2, TINY.mlp_dim :], mlp.experts[2].up_proj.weight)
    assert torch.equal(state["model.layers.1.mlp.experts.down_proj"][2], mlp.experts[2].down_proj.weight)
    assert torch.equal(state["model.layers.1.mlp.gate.weight"], mlp.router.weight)
    assert not any("router" in name or "gate_proj" in name for name in state)
    assert export.config_json(TINY, tok=type("Tok", (), {"special_tokens": {"<|endoftext|>": 0, "<|im_end|>": 2}}))["architectures"] == ["MixtralForCausalLM"]
