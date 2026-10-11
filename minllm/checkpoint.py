"""
Saving and loading the models that the stages produce.

A model is named by a spec: "qwen" for the published model, or a directory.
A directory written by the fine-tuning and RL scripts holds a meta.json that
says how to rebuild the model: which model it started from, whether it has
LoRA adapters, and where its weights are.
"""

import json
import os

import torch

from minllm import lora, pretrained
from minllm.common import load_checkpoint
from minllm.tokenizer import Tokenizer

TOKENIZER = "out/tokenizer.json"  # the tokenizer of the models trained from scratch here


def load(spec, device, dtype=torch.float32):
    """Returns (model, tokenizer, meta) for a spec."""
    if spec == "qwen":
        return pretrained.load_model(device=device, dtype=dtype), pretrained.load_tokenizer(), {"family": "qwen"}
    if not os.path.exists(f"{spec}/meta.json"):  # a run of scripts/train.py
        model, _ = load_checkpoint(f"{spec}/ckpt.pt", device)
        return model.eval(), Tokenizer.load(TOKENIZER), {"family": "small"}

    with open(f"{spec}/meta.json") as f:
        meta = json.load(f)
    if meta.get("lora"):
        # Adapters on top of the model they were trained on.
        model, tok, _ = load(meta["base"], device, dtype)
        lora.add(model, meta["lora"]["r"], meta["lora"]["alpha"])
        adapters = pretrained.read_safetensors(f"{spec}/adapter.safetensors")
        missing = model.load_state_dict({k: v.to(device=device, dtype=dtype) for k, v in adapters.items()}, strict=False)
        assert not missing.unexpected_keys and all(not k.endswith((".A", ".B")) for k in missing.missing_keys)
    elif meta["family"] == "qwen":
        model, tok, _ = load("qwen", "cpu", dtype)
        weights = pretrained.read_safetensors(f"{spec}/model.safetensors")
        weights.setdefault("lm_head.weight", weights["embed.weight"])  # the tied matrix is stored once
        model.load_state_dict({k: v.to(dtype) for k, v in weights.items()})
        model = model.to(device)
    else:
        model, _ = load_checkpoint(f"{spec}/ckpt.pt", device)
        tok = Tokenizer.load(TOKENIZER)
    return model.eval(), tok, meta


def save(dir, model, meta):
    """Write a model next to a meta.json. LoRA models save only their adapters."""
    os.makedirs(dir, exist_ok=True)
    if meta.get("lora"):
        pretrained.write_safetensors(f"{dir}/adapter.safetensors", {k: v.float() for k, v in lora.state(model).items()})
    elif meta["family"] == "qwen":
        # 16-bit floats, like the published file: 1 GB instead of 2
        state = {k: v.to(torch.bfloat16) for k, v in model.state_dict().items()}
        if model.cfg.tie_embeddings:
            del state["lm_head.weight"]  # the same matrix as embed.weight
        pretrained.write_safetensors(f"{dir}/model.safetensors", state)
    else:
        torch.save({"config": vars(model.cfg), "model": model.state_dict(), "step": meta["step"], "val_loss": meta["val_loss"]}, f"{dir}/ckpt.pt")
    with open(f"{dir}/meta.json", "w") as f:
        json.dump(meta, f, indent=1)
