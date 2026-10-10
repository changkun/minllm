"""
Measure generation speed: the uncached sampler against the cached one, and
the cached one at growing batch sizes.

    uv run scripts/bench_inference.py

Writes runs/bench/inference-<device>.json, which docs/04-inference.md quotes.
"""

import argparse
import json
import os
import time

import torch

from minllm.common import load_checkpoint, pick_device
from minllm.inference import generate

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", default="out/base/ckpt.pt")
parser.add_argument("--out", default=None)
args = parser.parse_args()

device = pick_device()
args.out = args.out or f"runs/bench/inference-{device}.json"
model, _ = load_checkpoint(args.ckpt, device)
model.eval()
prompt = [0]  # one arbitrary token; speed does not depend on what is written


def timed(fn):
    """Seconds for one call: the best of three, or of one if a call is slow."""
    fn()  # warm the device up
    best = float("inf")
    for _ in range(3):
        if device == "mps":
            torch.mps.synchronize()
        start = time.time()
        fn()
        if device == "mps":
            torch.mps.synchronize()
        best = min(best, time.time() - start)
        if best > 5:
            break
    return best


results = {"device": device, "parameters": model.num_params(), "length": [], "batch": []}

print(f"{'batch':>6} {'new tokens':>10} {'uncached s':>11} {'cached s':>9} {'speedup':>8}")
for batch in [1, 64]:
    for n in [16, 32, 64, 128, 255]:
        idx = torch.tensor([prompt] * batch, device=device)
        uncached = timed(lambda: model.generate(idx, n))
        cached = timed(lambda: generate(model, [prompt] * batch, n))
        row = {"batch": batch, "new_tokens": n, "uncached_seconds": uncached, "cached_seconds": cached}
        results["length"].append(row)
        print(f"{batch:>6} {n:>10} {uncached:>11.3f} {cached:>9.3f} {uncached / cached:>7.1f}x")

print(f"\n{'batch':>10} {'seconds':>11} {'tokens/s':>9}")
n = 200
for batch in [1, 4, 16, 64, 256, 1024] if device != "cpu" else [1, 4, 16, 64]:
    seconds = timed(lambda: generate(model, [prompt] * batch, n))
    results["batch"].append({"batch": batch, "new_tokens": n, "seconds": seconds, "tokens_per_second": batch * n / seconds})
    print(f"{batch:>10} {seconds:>11.3f} {batch * n / seconds:>9.0f}")

os.makedirs(os.path.dirname(args.out), exist_ok=True)
with open(args.out, "w") as f:
    json.dump(results, f, indent=1)
print(f"\nwrote {args.out}")
