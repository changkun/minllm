"""
The held-out loss of several models on several corpora: every model on
every corpus, with the same tokenizer and the same windows of text.

    uv run scripts/cross_loss.py --models out/web-raw out/web-rules out/web-clean \\
        --data data/web/raw data/web/rules data/web/clean data/web/stories --tokenizer out/web/tokenizer.json --out runs/data/cross.json

A loss per token says how well a model predicts a text in that tokenizer's
pieces. Divided by the bytes a token stands for, it is bits per byte of
text, which can be compared across corpora. See docs/03b-data.md.
"""

import argparse
import json
import math
import os

import torch

from minllm.common import load_checkpoint, pick_device
from minllm.data import EOT, get_batch, load_tokens

parser = argparse.ArgumentParser()
parser.add_argument("--models", nargs="+", required=True, help="directories of models trained by scripts/train.py")
parser.add_argument("--data", nargs="+", required=True, help="directories with a valid.txt and its valid.bin")
parser.add_argument("--tokenizer", required=True)
parser.add_argument("--batches", type=int, default=100, help="windows of 64 by 256 tokens per corpus")
parser.add_argument("--out", required=True)
args = parser.parse_args()

device = pick_device()
corpora = {}
for path in args.data:
    tokens = load_tokens(f"{path}/valid.bin")
    with open(f"{path}/valid.txt", "rb") as f:
        text = f.read()
    # the bytes of the text itself: the markers between its documents are single tokens
    size = len(text) - text.count(EOT.encode()) * len(EOT)
    generator = torch.Generator().manual_seed(0)
    corpora[os.path.basename(path)] = {"tokens": len(tokens), "bytes": size, "batches": [get_batch(tokens, 64, 256, "cpu", generator) for _ in range(args.batches)]}

result = {"tokenizer": args.tokenizer, "windows": args.batches * 64, "corpora": {name: {"tokens": c["tokens"], "bytes": c["bytes"]} for name, c in corpora.items()}, "loss": {}, "bits_per_byte": {}}
for path in args.models:
    model, ckpt = load_checkpoint(f"{path}/ckpt.pt", device)
    model.eval()
    name = os.path.basename(path.rstrip("/"))
    result["loss"][name], result["bits_per_byte"][name] = {}, {}
    for corpus, c in corpora.items():
        total = 0.0
        with torch.no_grad(), torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device != "cpu"):
            for x, y in c["batches"]:
                total += model(x.to(device), y.to(device))[1].item()
        loss = total / args.batches
        result["loss"][name][corpus] = loss
        result["bits_per_byte"][name][corpus] = loss / math.log(2) * c["tokens"] / c["bytes"]
    print(f"{name:12s} " + "  ".join(f"{corpus} {result['loss'][name][corpus]:.3f} ({result['bits_per_byte'][name][corpus]:.3f} bits/byte)" for corpus in corpora))

os.makedirs(os.path.dirname(args.out), exist_ok=True)
with open(args.out, "w") as f:
    json.dump(result, f, indent=1)
print(f"wrote {args.out}")
