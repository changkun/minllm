"""
Measure generation speed: the uncached sampler against the cached one, and
the cached one at growing batch sizes.

    uv run scripts/bench_inference.py                 # the 14M model of stage 3
    uv run scripts/bench_inference.py --model qwen    # Qwen2.5-0.5B-Instruct

Writes runs/bench/inference-<device>.json or inference-qwen-<device>.json,
which docs/04-inference.md and docs/05-pretrained.md quote.
"""

import argparse
import json
import os
import time

import torch

from minllm.common import load_checkpoint, pick_device, result_path
from minllm.inference import generate
from minllm.pretrained import load_model

parser = argparse.ArgumentParser()
parser.add_argument("--model", choices=["base", "qwen"], default="base")
parser.add_argument("--ckpt", default="out/base/ckpt.pt")
parser.add_argument("--out", default=None)
args = parser.parse_args()

device = pick_device()
if args.model == "base":
    args.out = result_path(args.out or f"runs/bench/inference-{device}.json")
    model, _ = load_checkpoint(args.ckpt, device)
    # (batch, prompt tokens, new tokens): a short prompt and ever longer texts
    lengths = [(batch, 1, n) for batch in [1, 64] for n in [16, 32, 64, 128, 255]]
    batches = [1, 4, 16, 64, 256, 1024] if device != "cpu" else [1, 4, 16, 64]
    batch_prompt, batch_new = 1, 200
else:
    args.out = result_path(args.out or f"runs/bench/inference-qwen-{device}.json")
    model = load_model(device=device, dtype=torch.bfloat16)
    # the same few new tokens after ever longer prompts
    lengths = [(batch, p, 48) for p in [32, 512, 2048] for batch in [1, 16] if (p, batch) != (2048, 16)]
    batches = [1, 4, 16, 64, 128]
    batch_prompt, batch_new = 64, 100
model.eval()
prompt = [0]  # arbitrary tokens; speed does not depend on what is written


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
    if device == "mps":
        torch.mps.empty_cache()  # hand the memory of this measurement back
    return best


results = {"device": device, "parameters": model.num_params(), "length": [], "batch": []}

print(f"{'batch':>6} {'prompt':>7} {'new tokens':>10} {'uncached s':>11} {'cached s':>9} {'speedup':>8}")
for batch, p, n in lengths:
    idx = torch.tensor([prompt * p] * batch, device=device)
    uncached = timed(lambda: model.generate(idx, n))
    cached = timed(lambda: generate(model, [prompt * p] * batch, n))
    row = {"batch": batch, "prompt_tokens": p, "new_tokens": n, "uncached_seconds": uncached, "cached_seconds": cached}
    results["length"].append(row)
    print(f"{batch:>6} {p:>7} {n:>10} {uncached:>11.3f} {cached:>9.3f} {uncached / cached:>7.1f}x")

print(f"\n{'batch':>10} {'seconds':>11} {'tokens/s':>9}")
for batch in batches:
    seconds = timed(lambda: generate(model, [prompt * batch_prompt] * batch, batch_new))
    row = {"batch": batch, "prompt_tokens": batch_prompt, "new_tokens": batch_new, "seconds": seconds}
    results["batch"].append(row | {"tokens_per_second": batch * batch_new / seconds})
    print(f"{batch:>10} {seconds:>11.3f} {batch * batch_new / seconds:>9.0f}")

os.makedirs(os.path.dirname(args.out), exist_ok=True)
with open(args.out, "w") as f:
    json.dump(results, f, indent=1)
print(f"\nwrote {args.out}")
