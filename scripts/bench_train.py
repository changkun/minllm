"""
Measure the speed of a training step, with and without the two switches that
scripts/train.py turns on by default.

    uv run scripts/bench_train.py

Writes runs/bench/train-<device>.json, which docs/03-pretraining.md quotes.
"""

import argparse
import json
import os
import time

import torch

from minllm.common import pick_device
from minllm.model import Config, Transformer

parser = argparse.ArgumentParser()
parser.add_argument("--batch-size", type=int, default=64)
parser.add_argument("--seq-len", type=int, default=256)
parser.add_argument("--steps", type=int, default=30)
parser.add_argument("--out", default=None)
args = parser.parse_args()

device = pick_device()
device_type = torch.device(device).type
args.out = args.out or f"runs/bench/train-{device}.json"
cfg = Config(max_seq_len=args.seq_len)
tokens = args.batch_size * args.seq_len


def tokens_per_second(compile, amp):
    """Forward pass, backward pass and optimizer step on random tokens."""
    torch.manual_seed(0)
    model = Transformer(cfg).to(device)
    forward = torch.compile(model) if compile else model
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    autocast = torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=amp)
    x = torch.randint(0, cfg.vocab_size, (args.batch_size, args.seq_len), device=device)
    y = torch.randint(0, cfg.vocab_size, (args.batch_size, args.seq_len), device=device)
    times = []
    for _ in range(args.steps):
        start = time.time()
        with autocast:
            _, loss = forward(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        loss.item()  # wait for the device to finish
        times.append(time.time() - start)
    times = sorted(times[5:])  # the first steps include compilation
    return tokens / times[len(times) // 2]


results = []
for name, compile, amp in [
    ("Plain PyTorch, 32-bit floats", False, False),
    ("bfloat16 autocast", False, True),
    ("torch.compile", True, False),
    ("Both, the default", True, True),
]:
    speed = tokens_per_second(compile, amp)
    results.append({"name": name, "compile": compile, "amp": amp, "tokens_per_second": speed})
    print(f"{name:30s} {speed / 1e3:6.1f}k tokens/s")

os.makedirs(os.path.dirname(args.out), exist_ok=True)
with open(args.out, "w") as f:
    json.dump({"device": device, "batch_size": args.batch_size, "seq_len": args.seq_len, "configs": results}, f, indent=1)
print(f"wrote {args.out}")
