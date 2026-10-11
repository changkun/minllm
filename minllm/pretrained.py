"""
Loading a published model: Qwen2.5-0.5B-Instruct.

A released model is three things: a file of weights, a tokenizer, and a
configuration that says how the weights fit together. None of them needs a
library to read. See docs/05-pretrained.md.
"""

import json
import os

import numpy as np
import torch

from minllm.common import fetch
from minllm.model import Config, Transformer
from minllm.tokenizer import Tokenizer

REPO = "Qwen/Qwen2.5-0.5B-Instruct"
DIR = "data/qwen2.5-0.5b-instruct"
FILES = ["config.json", "vocab.json", "merges.txt", "tokenizer.json", "tokenizer_config.json", "model.safetensors"]

# The system prompt that Qwen's chat template inserts when a conversation has none.
SYSTEM_PROMPT = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."


def download(dir=DIR, repo=REPO):
    for name in FILES:
        fetch(f"https://huggingface.co/{repo}/resolve/main/{name}", f"{dir}/{name}")


# ---------------------------------------------------------------- weights

DTYPES = {"F32": torch.float32, "F16": torch.float16, "BF16": torch.bfloat16, "I64": torch.int64}
# The packed words of a quantized model are unsigned 32-bit integers. PyTorch
# computes with signed ones only, so they are read as int32: the same bits.
DTYPES |= {"U32": torch.int32}


def read_safetensors(path):
    """
    Read a safetensors file into a dict of tensors. The format is: 8 bytes
    holding the length n of a header, n bytes of JSON that give every tensor's
    dtype, shape and byte range, and then the tensors' raw bytes.
    """
    with open(path, "rb") as f:
        n = int.from_bytes(f.read(8), "little")
        header = json.loads(f.read(n))
    data = np.memmap(path, dtype=np.uint8, mode="c", offset=8 + n)  # read on demand
    tensors = {}
    for name, info in header.items():
        if name != "__metadata__":
            start, end = info["data_offsets"]
            tensors[name] = torch.frombuffer(data[start:end], dtype=DTYPES[info["dtype"]]).reshape(info["shape"])
    return tensors


def write_safetensors(path, tensors):
    names = {dtype: name for name, dtype in DTYPES.items()}
    header, offset = {}, 0
    for name, t in tensors.items():
        size = t.numel() * t.element_size()
        header[name] = {"dtype": names[t.dtype], "shape": list(t.shape), "data_offsets": [offset, offset + size]}
        offset += size
    head = json.dumps(header, separators=(",", ":")).encode()
    head += b" " * (-len(head) % 8)  # the data starts at a multiple of 8 bytes
    with open(path, "wb") as f:
        f.write(len(head).to_bytes(8, "little") + head)
        for t in tensors.values():
            # numpy has no bfloat16, so every dtype is written as its raw bytes
            f.write(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())


def to_minllm(name):
    """Translate a weight name from the published file to the name it has in minllm/model.py."""
    name = name.removeprefix("model.")
    for theirs, ours in [
        ("embed_tokens", "embed"),
        ("self_attn", "attn"),
        ("input_layernorm", "attn_norm"),
        ("post_attention_layernorm", "mlp_norm"),
    ]:
        name = name.replace(theirs, ours)
    return name


def load_model(dir=DIR, device="cpu", dtype=torch.float32):
    """Build the model that config.json describes and fill it with the published weights."""
    with open(f"{dir}/config.json") as f:
        c = json.load(f)
    cfg = Config(
        vocab_size=c["vocab_size"],
        max_seq_len=c["max_position_embeddings"],
        dim=c["hidden_size"],
        n_layers=c["num_hidden_layers"],
        n_heads=c["num_attention_heads"],
        n_kv_heads=c["num_key_value_heads"],
        mlp_dim=c["intermediate_size"],
        rope_theta=c["rope_theta"],
        norm_eps=c["rms_norm_eps"],
        tie_embeddings=c["tie_word_embeddings"],
        qkv_bias=True,
    )
    model = Transformer(cfg)
    state = {to_minllm(name): t for name, t in read_safetensors(f"{dir}/model.safetensors").items()}
    if cfg.tie_embeddings:
        state["lm_head.weight"] = state["embed.weight"]  # one matrix, stored once
    model.load_state_dict(state)  # raises if a single name or shape does not match
    return model.to(device=device, dtype=dtype).eval()


# -------------------------------------------------------------- tokenizer


def gpt2_alphabet():
    """
    GPT-2's vocabulary files spell every byte as one printable character, so
    that a token can be written on a line of text: printable bytes stand for
    themselves, and the others are moved up past code point 256. Qwen keeps
    this convention. Returns the mapping from character to byte.
    """
    printable = list(range(ord("!"), ord("~") + 1)) + list(range(0xA1, 0xAD)) + list(range(0xAE, 0x100))
    chars, shifted = {}, 0
    for b in range(256):
        if b in printable:
            chars[chr(b)] = b
        else:
            chars[chr(256 + shifted)] = b
            shifted += 1
    return chars


def load_tokenizer(dir=DIR):
    """Build a Tokenizer from the published vocabulary, merges and special tokens."""
    with open(f"{dir}/vocab.json", encoding="utf-8") as f:
        ids = json.load(f)  # token, spelled in GPT-2's alphabet -> id
    with open(f"{dir}/tokenizer.json", encoding="utf-8") as f:
        spec = json.load(f)
    alphabet = gpt2_alphabet()
    byte_ids = [0] * 256
    for char, byte in alphabet.items():
        byte_ids[byte] = ids[char]

    merges = {}
    with open(f"{dir}/merges.txt", encoding="utf-8") as f:
        for line in f.read().split("\n"):
            if line and not line.startswith("#version"):
                a, b = line.split(" ")
                merges[(ids[a], ids[b])] = ids[a + b]

    special = {t["content"]: t["id"] for t in spec["added_tokens"]}
    pattern = spec["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"]
    return Tokenizer(merges, special, pattern, byte_ids, normalize=spec["normalizer"]["type"])
