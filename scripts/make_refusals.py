"""
Refusals to train on: the published model's own replies to harmful
requests of AdvBench, kept where they are refusals.

    uv run scripts/make_refusals.py      # the first 260 requests; writes data/safety/refusals.jsonl

The other 260 are left for testing. Only refusals are written, and the
replies that were not refusals are counted and dropped. See docs/08a-safety.md.
"""

import argparse
import json
import os

import torch

from minllm import checkpoint, evaluate, safety
from minllm.common import pick_device
from minllm.pretrained import SYSTEM_PROMPT

parser = argparse.ArgumentParser()
parser.add_argument("--n", type=int, default=260)
parser.add_argument("--out", default="data/safety/refusals.jsonl")
args = parser.parse_args()

device = pick_device()
model, tok, _ = checkpoint.load("qwen", device, torch.bfloat16 if device != "cpu" else torch.float32)
requests = safety.advbench()[: args.n]
examples = [{"messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": r}]} for r in requests]
replies = evaluate.replies(model, tok, examples, 128, 64)
kept = [(r, reply) for r, reply in zip(requests, replies) if safety.refuses(reply) and reply.rstrip().endswith((".", "!"))]
os.makedirs(os.path.dirname(args.out), exist_ok=True)
with open(args.out, "w", encoding="utf-8") as f:
    for request, reply in kept:
        f.write(json.dumps({"messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": request}, {"role": "assistant", "content": reply.strip()}]}, ensure_ascii=False) + "\n")
print(f"{len(kept)} of {len(requests)} replies are complete refusals; wrote {args.out}")
