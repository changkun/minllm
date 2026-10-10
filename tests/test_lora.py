import torch

from minllm import lora
from minllm.model import Config, Transformer

TINY = Config(vocab_size=64, max_seq_len=32, dim=48, n_layers=2, n_heads=4, n_kv_heads=2, mlp_dim=96)


def setup(seed=0):
    torch.manual_seed(seed)
    model = Transformer(TINY)
    idx = torch.randint(0, TINY.vocab_size, (2, 16))
    return model, idx, model(idx)[0].detach()


def test_starts_as_the_pretrained_model():
    model, idx, before = setup()
    params = lora.add(model, r=4, alpha=8)
    assert torch.equal(model(idx)[0], before)  # B is zero, so nothing has changed yet

    # r * (in + out) new numbers per wrapped layer, and nothing else trains
    head_dim = TINY.dim // TINY.n_heads
    kv = TINY.n_kv_heads * head_dim
    attn = 4 * ((TINY.dim + TINY.dim) * 2 + (TINY.dim + kv) * 2)
    mlp = 4 * (TINY.dim + TINY.mlp_dim) * 3
    assert sum(p.numel() for p in params) == TINY.n_layers * (attn + mlp)
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == sum(p.numel() for p in params)


def test_only_the_adapters_learn():
    model, idx, before = setup()
    params = lora.add(model, r=4, alpha=8)
    frozen = {name: p.clone() for name, p in model.named_parameters() if not p.requires_grad}
    optimizer = torch.optim.AdamW(params, lr=1e-2)
    for _ in range(20):
        _, loss = model(idx[:, :-1], idx[:, 1:])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    assert not torch.allclose(model(idx)[0], before, atol=1e-3)  # it learned something
    assert all(torch.equal(p, frozen[name]) for name, p in model.named_parameters() if name in frozen)

    # Switched off, the model is the pretrained one again: this is the
    # reference policy that the RL stages compare against, at no extra memory.
    lora.enable(model, False)
    assert torch.equal(model(idx)[0], before)
    lora.enable(model, True)

    # Merged, it is an ordinary model that computes the adapted function.
    adapted = model(idx)[0].detach()
    names = set(lora.state(model))
    assert len(names) == 2 * len(lora.modules(model))
    lora.merge(model)
    assert not lora.modules(model) and "layers.0.attn.q_proj.weight" in model.state_dict()
    assert torch.allclose(model(idx)[0], adapted, atol=1e-5)
