"""
Measure a model of 3b's web text on windows longer than it was trained on:
the loss at every position of the first 2,048 tokens of long documents,
and what a passage seen before is worth when it comes again, at distances
up to 1,984 tokens.

    uv run scripts/long_context.py --model out/web-clean                      # the table computed further, as it is
    uv run scripts/long_context.py --model out/web-clean --scaling yarn:4     # stretched, without training
    uv run scripts/long_context.py --model out/ctx-yarn                       # after training on windows of 1,024

Writes runs/context/<name>.json. See docs/02b-context.md.
"""

import argparse
import json
import os

import numpy as np
import torch

from minllm import context
from minllm.common import load_checkpoint, pick_device
from minllm.data import EOT, load_tokens
from minllm.tokenizer import Tokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
parser.add_argument("--scaling", default=None, help="stretch the rotary table first, as type:factor; a model trained stretched keeps its own")
parser.add_argument("--data", default="data/web/clean")
parser.add_argument("--tokenizer", default="out/web/tokenizer.json")
parser.add_argument("--length", type=int, default=2048)
parser.add_argument("--documents", type=int, default=256, help="documents of at least --length tokens whose first tokens are read")
parser.add_argument("--distances", default="64,128,192,256,384,512,768,1024,1536,1984")
parser.add_argument("--name", default=None)
args = parser.parse_args()

device = pick_device()
model, _ = load_checkpoint(f"{args.model}/ckpt.pt", device)
model.eval()
tok = Tokenizer.load(args.tokenizer)
trained = model.cfg.max_seq_len
kind, factor = args.scaling.split(":") if args.scaling else (None, None)
if not kind and model.cfg.rope_scaling:
    kind, factor = model.cfg.rope_scaling["type"], model.cfg.rope_scaling["factor"]
context.extend(model, args.length, kind, float(factor) if factor else None)
autocast = torch.autocast(device_type=torch.device(device).type, dtype=torch.bfloat16, enabled=device != "cpu")

tokens = load_tokens(f"{args.data}/valid.bin")
eot = tok.special_tokens[EOT]
cuts = np.flatnonzero(tokens == eot)
documents = [tokens[a + 1 : b].astype(int).tolist() for a, b in zip(cuts[:-1], cuts[1:])]
# Windows from the separator before a long document on: the first prediction is its first token.
starts = [int(a) for a, b in zip(cuts[:-1], cuts[1:]) if b - a > args.length][: args.documents]
with torch.no_grad(), autocast:
    per_position = context.loss_by_position(model, tokens, args.length, starts)
    copies = {}
    for distance in [int(d) for d in args.distances.split(",")]:
        seen, at = context.copy_sequences(documents, distance, eot)
        unseen, _ = context.copy_sequences(documents, distance, eot, control=True)
        copies[distance] = {"seen": context.copy_loss(model, seen, at), "unseen": context.copy_loss(model, unseen, at)}
        print(f"passage again after {distance:5d} tokens: loss {copies[distance]['seen']:.3f}, first time {copies[distance]['unseen']:.3f}", flush=True)

for lo, hi in [(0, 256), (256, 512), (512, 1024), (1024, 2048)]:
    print(f"positions {lo:4d} to {hi:4d}: loss {per_position[lo : hi - 1].mean():.3f}")
name = args.name or os.path.basename(args.model.rstrip("/")) + (f"-{kind}{factor}" if args.scaling else "")
os.makedirs("runs/context", exist_ok=True)
blocks = [per_position[i : i + 32].mean().item() for i in range(0, len(per_position), 32)]
result = {"model": args.model, "trained_length": trained, "scaling": model.cfg.rope_scaling, "length": args.length, "documents": len(starts)}
result |= {"loss_by_32_positions": [round(v, 4) for v in blocks], "passage": copies}
with open(f"runs/context/{name}.json", "w") as f:
    json.dump(result, f, indent=1)
print(f"wrote runs/context/{name}.json")
