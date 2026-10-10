"""
Save the attention weights of a few heads of the trained model on one
sentence, for the figure in docs/02-model.md.

    uv run scripts/dump_attention.py

Writes runs/base/attention.json.
"""

import argparse
import json

import torch

from minllm.common import load_checkpoint
from minllm.data import EOT
from minllm.model import apply_rope
from minllm.tokenizer import Tokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", default="out/base/ckpt.pt")
parser.add_argument("--tokenizer", default="out/tokenizer.json")
parser.add_argument("--text", default="Once upon a time, there was a little girl named Lily. She loved to play with her red ball.")
parser.add_argument("--heads", default="1.2,5.5", help="layer.head pairs, counted from zero")
parser.add_argument("--out", default="runs/base/attention.json")
args = parser.parse_args()

tok = Tokenizer.load(args.tokenizer)
model, _ = load_checkpoint(args.ckpt, "cpu")
ids = [tok.special_tokens[EOT]] + tok.encode_ordinary(args.text)
idx, T = torch.tensor([ids]), len(ids)


@torch.no_grad()
def attention_weights(model, idx):
    """
    The weights of every layer and head, (layers, heads, T, T). This is
    Attention.forward up to the softmax, written out because the library call
    the model uses returns only the output.
    """
    weights = []
    x = model.embed(idx)
    cos, sin = model.rope_cos[:T], model.rope_sin[:T]
    future = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
    for layer in model.layers:
        attn, h = layer.attn, layer.attn_norm(x)
        q = attn.q_proj(h).view(1, T, attn.n_heads, attn.head_dim).transpose(1, 2)
        k = attn.k_proj(h).view(1, T, attn.n_kv_heads, attn.head_dim).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        k = k.repeat_interleave(attn.n_heads // attn.n_kv_heads, dim=1)
        scores = q @ k.transpose(-2, -1) / attn.head_dim**0.5
        weights.append(scores.masked_fill(future, float("-inf")).softmax(dim=-1)[0])
        x = layer(x, cos, sin)
    return torch.stack(weights)


weights = attention_weights(model, idx)
heads = []
for pair in args.heads.split(","):
    layer, head = map(int, pair.split("."))
    rows = [[round(w, 3) for w in row] for row in weights[layer, head].tolist()]
    heads.append({"layer": layer, "head": head, "weights": rows})
tokens = [EOT] + [tok.decode([i]) for i in ids[1:]]
with open(args.out, "w") as f:
    json.dump({"text": args.text, "tokens": tokens, "heads": heads}, f)
print(f"wrote {args.out}: {T} tokens, heads {args.heads}")
