"""
Measure what a mixture of experts costs in time: a training step and
generation, against the dense model with the same active parameters and the
dense model with the same total.

    uv run scripts/bench_moe.py
    uv run scripts/bench_moe.py --scale 2 --only generate    # generation with models twice as wide

Writes runs/bench/moe-<device>.json, or moe-<device>-x2.json for a scale of
2, which docs/11-moe.md quotes.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import replace

import torch

from minllm.common import pick_device
from minllm.inference import generate
from minllm.model import Config, Transformer

parser = argparse.ArgumentParser()
parser.add_argument("--batch-size", type=int, default=64)
parser.add_argument("--seq-len", type=int, default=256)
parser.add_argument("--steps", type=int, default=40)
parser.add_argument("--scale", type=int, default=1, help="make every model this many times as wide")
parser.add_argument("--only", choices=["train", "generate"], default=None, help="measure only training steps, or only generation")
parser.add_argument("--repeats", type=int, default=3, help="measure every case this often and keep the fastest")
parser.add_argument("--out", default=None)
parser.add_argument("--case", default=None, help="measure this one case and print the result: the script calls itself with it")
parser.add_argument("--generate", type=int, default=0, help="with --case: measure generation at this batch size, not training")
args = parser.parse_args()

dense = Config(max_seq_len=args.seq_len, dim=Config.dim * args.scale, n_heads=Config.n_heads * args.scale, mlp_dim=Config.mlp_dim * args.scale)
moe = replace(dense, mlp_dim=dense.mlp_dim // 2, n_experts=8, n_active=2)  # two active experts of half the width
wide = replace(dense, mlp_dim=moe.mlp_dim * moe.n_experts)  # every expert's parameters, in one MLP
everything = moe.n_experts / moe.n_active  # the capacity at which every expert takes every token
CASES = {
    "dense": ("Dense", dense, None),
    "wide": ("Dense", wide, None),
    "own": ("Experts take their own tokens", moe, None),
    "every": ("Experts take every token", moe, everything),
    "capacity-1.5": ("Capacity 1.5", moe, 1.5),
    "capacity-1": ("Capacity 1", moe, 1.0),
}


def build(cfg, capacity, device):
    torch.manual_seed(0)
    model = Transformer(cfg).to(device)
    for m in model.mixtures():
        m.capacity = capacity
    return model


def train_speed(cfg, capacity, device):
    """Tokens per second of forward pass, backward pass and optimizer step, compiled and in bfloat16."""
    device_type = torch.device(device).type
    model = build(cfg, capacity, device)
    forward = torch.compile(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6)
    autocast = torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=device_type != "cpu")
    times = []
    for _ in range(args.steps):
        # new tokens at every step, so that the routing changes as it does in training
        x = torch.randint(0, cfg.vocab_size, (args.batch_size, args.seq_len), device=device)
        y = torch.randint(0, cfg.vocab_size, (args.batch_size, args.seq_len), device=device)
        start = time.time()
        with autocast:
            _, loss = forward(x, y)
            if cfg.n_experts:
                loss = loss + 0.01 * model.balance_loss()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        loss.item()  # wait for the device to finish
        times.append(time.time() - start)
    times = sorted(times[10:])  # the first steps include compilation
    return args.batch_size * args.seq_len / times[len(times) // 2], model.num_params()


def generation_speed(cfg, capacity, device, batch, new_tokens=200):
    """Tokens per second of the cached sampler of stage 4, the best of three runs."""
    model = build(cfg, capacity, device).eval()
    run = lambda: generate(model, [[0]] * batch, new_tokens)
    run()  # warm the device up
    best = float("inf")
    for _ in range(3):
        if device == "mps":
            torch.mps.synchronize()
        start = time.time()
        run()
        if device == "mps":
            torch.mps.synchronize()
        best = min(best, time.time() - start)
    return batch * new_tokens / best, model.num_params()


if args.case:
    device = pick_device()
    _, cfg, capacity = CASES[args.case]
    speed, parameters = generation_speed(cfg, capacity, device, args.generate) if args.generate else train_speed(cfg, capacity, device)
    print(json.dumps({"device": device, "parameters": parameters, "tokens_per_second": speed}))
    raise SystemExit


def measure(case, generate=0):
    """
    The fastest of several measurements of one case, each in a process of its
    own. Compiled code and the pool of GPU memory outlive the model they were
    made for, and a measurement taken after another one in the same process
    came out up to a third slower. And a laptop in use is not a steady
    instrument: whatever else it is doing can only slow a measurement down,
    so the fastest one is the closest to what the machine can do.
    """
    shape = ["--batch-size", str(args.batch_size), "--seq-len", str(args.seq_len), "--steps", str(args.steps), "--scale", str(args.scale)]
    command = [sys.executable, __file__, "--case", case, "--generate", str(generate)] + shape
    runs = [json.loads(subprocess.run(command, capture_output=True, text=True, check=True).stdout.splitlines()[-1]) for _ in range(args.repeats)]
    return max(runs, key=lambda r: r["tokens_per_second"])


def named(case, r):
    name = CASES[case][0]
    return f"{name}, {r['parameters'] / 1e6:.0f}M" if name == "Dense" else name


results = {"batch_size": args.batch_size, "seq_len": args.seq_len, "scale": args.scale, "repeats": args.repeats, "train": [], "generate": []}
if args.only != "generate":
    for case, (_, cfg, capacity) in CASES.items():
        r = measure(case)
        results["train"].append({"name": named(case, r), "capacity": capacity, "parameters": r["parameters"], "tokens_per_second": r["tokens_per_second"]})
        print(f"training    {named(case, r):30s} {r['parameters'] / 1e6:5.1f}M {r['tokens_per_second'] / 1e3:6.1f}k tokens/s", flush=True)
if args.only != "train":
    for case, (_, cfg, capacity) in list(CASES.items())[:4]:
        alone, batch = measure(case, 1), measure(case, 64)
        row = {"name": named(case, alone), "capacity": capacity, "parameters": alone["parameters"]}
        results["generate"].append(row | {"batch_1": alone["tokens_per_second"], "batch_64": batch["tokens_per_second"]})
        print(f"generation  {row['name']:30s} {alone['tokens_per_second']:6.0f} tokens/s alone, {batch['tokens_per_second']:6.0f} in a batch of 64", flush=True)

args.out = args.out or f"runs/bench/moe-{pick_device()}{f'-x{args.scale}' if args.scale > 1 else ''}.json"
os.makedirs(os.path.dirname(args.out), exist_ok=True)
with open(args.out, "w") as f:
    json.dump({"device": pick_device()} | results, f, indent=1)
print(f"wrote {args.out}")
