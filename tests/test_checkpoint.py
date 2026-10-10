import torch

from minllm import checkpoint, lora
from minllm.model import Config, Transformer
from minllm.tokenizer import Tokenizer

TINY = Config(vocab_size=259, max_seq_len=32, dim=48, n_layers=2, n_heads=4, n_kv_heads=2, mlp_dim=96)


def test_save_and_load(tmp_path, monkeypatch):
    # A stand-in for a run of scripts/train.py: a checkpoint and a tokenizer.
    Tokenizer(special_tokens=["<|endoftext|>", "<|im_start|>", "<|im_end|>"]).save(tmp_path / "tokenizer.json")
    monkeypatch.setattr(checkpoint, "TOKENIZER", str(tmp_path / "tokenizer.json"))
    torch.manual_seed(0)
    base = Transformer(TINY)
    (tmp_path / "base").mkdir()
    torch.save({"config": vars(TINY), "model": base.state_dict(), "step": 1, "val_loss": 2.0}, tmp_path / "base" / "ckpt.pt")
    idx = torch.randint(0, 256, (2, 12))

    model, tok, meta = checkpoint.load(str(tmp_path / "base"), "cpu")
    assert meta == {"family": "small"} and tok.vocab_size == 259
    assert torch.equal(model(idx)[0], base(idx)[0])

    # A fully fine-tuned model is saved whole.
    with torch.no_grad():
        model.embed.weight += 0.1
    full = {"family": "small", "base": str(tmp_path / "base"), "step": 5, "val_loss": 1.5}
    checkpoint.save(str(tmp_path / "full"), model, full)
    loaded, _, loaded_meta = checkpoint.load(str(tmp_path / "full"), "cpu")
    assert loaded_meta == full and torch.equal(loaded(idx)[0], model(idx)[0])

    # A LoRA model saves its adapters only, and is rebuilt on top of its base.
    model, _, _ = checkpoint.load(str(tmp_path / "base"), "cpu")
    adapters = lora.add(model, r=4, alpha=8)
    with torch.no_grad():
        for p in adapters:
            p.normal_(std=0.05)
    adapted = {"family": "small", "base": str(tmp_path / "base"), "lora": {"r": 4, "alpha": 8}, "step": 5, "val_loss": 1.5}
    checkpoint.save(str(tmp_path / "adapted"), model, adapted)
    assert not (tmp_path / "adapted" / "ckpt.pt").exists()
    loaded, _, _ = checkpoint.load(str(tmp_path / "adapted"), "cpu")
    assert torch.allclose(loaded(idx)[0], model(idx)[0], atol=1e-6)
    assert not torch.allclose(loaded(idx)[0], base(idx)[0], atol=1e-3)
