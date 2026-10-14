"""
Measure what each kind of attention costs at generation: the cache it keeps
per token, and how fast it writes 255 tokens for 1 to 1,024 sequences at
once, with the cached sampler of stage 4. The models are those trained for
docs/02c-latent-attention.md.

    uv run scripts/bench_attention.py

Every model is measured in a fresh process, three times, and the fastest
run kept, as scripts/bench_moe.py does. Writes runs/bench/attention-<device>.json.
"""

import argparse
import json
import subprocess
import sys
import time

import torch

from minllm.common import load_checkpoint, pick_device, result_path
from minllm.inference import KVCache, generate

MODELS = {
    "out/mha": "Multi-head, 6 key/value heads",
    "out/base": "Grouped-query, 2 key/value heads",
    "out/mqa": "Multi-query, 1 key/value head",
    "out/mla-128": "Latent, 128 + 32",
    "out/mla-64": "Latent, 64 + 32",
    "out/mla-32": "Latent, 32 + 16",
}
BATCHES = [1, 64, 256, 1024]
NEW_TOKENS = 255

parser = argparse.ArgumentParser()
parser.add_argument("--models", nargs="+", default=list(MODELS))
parser.add_argument("--repeats", type=int, default=3, help="measure every model this often and keep the fastest")
parser.add_argument("--out", default=None)
parser.add_argument("--case", default=None, help="measure this one model and print the result: the script calls itself with it")
parser.add_argument("--no-absorb", action="store_true", help="with --case: latent attention projects the whole cache up at every step")
args = parser.parse_args()


def timed(fn, device):
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
        torch.mps.empty_cache()
    return best


if args.case:
    device = pick_device()
    model, _ = load_checkpoint(f"{args.case}/ckpt.pt", device)
    model.eval()
    for layer in model.layers:
        layer.attn.absorb = not args.no_absorb
    cache = KVCache(model.cfg.n_layers, 8)
    model(torch.zeros(1, 8, dtype=torch.long, device=device), cache=cache)
    seconds = {batch: timed(lambda: generate(model, [[0]] * batch, NEW_TOKENS), device) for batch in BATCHES}
    print(json.dumps({"device": device, "parameters": model.num_params(), "cache_bytes_per_token": cache.nbytes // 8, "seconds": seconds}))
    raise SystemExit


def measure(model, no_absorb=False):
    """The fastest of several measurements, each in a process of its own, batch by batch."""
    command = [sys.executable, __file__, "--case", model] + (["--no-absorb"] if no_absorb else [])
    runs = [json.loads(subprocess.run(command, capture_output=True, text=True, check=True).stdout.splitlines()[-1]) for _ in range(args.repeats)]
    best = runs[0] | {"seconds": {b: min(r["seconds"][str(b)] for r in runs) for b in BATCHES}}
    return best | {"tokens_per_second": {b: b * NEW_TOKENS / s for b, s in best["seconds"].items()}}


results = {"new_tokens": NEW_TOKENS, "repeats": args.repeats, "models": []}
print(f"{'':40s} {'params':>7s} {'cache/token':>12s} " + " ".join(f"{f'{b} seq/s':>10s}" for b in BATCHES))
cases = [(m, False) for m in args.models] + [(m, True) for m in args.models if "mla" in m][:1]
for model, no_absorb in cases:
    r = measure(model, no_absorb)
    name = MODELS.get(model, model) + (", projected up at every step" if no_absorb else "")
    results["models"].append({"model": model, "name": name, "absorb": not no_absorb} | r)
    rates = " ".join(f"{r['tokens_per_second'][b]:10.0f}" for b in BATCHES)
    print(f"{name:40s} {r['parameters'] / 1e6:6.1f}M {r['cache_bytes_per_token'] / 1024:9.1f} KB {rates}", flush=True)
    results["device"] = r["device"]

out = result_path(args.out or f"runs/bench/attention-{results['device']}.json")
with open(out, "w") as f:
    json.dump(results, f, indent=1)
print(f"wrote {out}")
