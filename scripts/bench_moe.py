"""
Measure what a mixture of experts costs in time: a training step and
generation, against the dense model with the same active parameters and the
dense model with the same total.

    uv run scripts/bench_moe.py

Writes runs/bench/moe-<device>.json, which docs/11-moe.md quotes.
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
parser.add_argument("--out", default=None)
parser.add_argument("--case", default=None, help="measure this one case and print the result: the script calls itself with it")
parser.add_argument("--generate", type=int, default=0, help="with --case: measure generation at this batch size, not training")
args = parser.parse_args()

dense = Config(max_seq_len=args.seq_len)
moe = replace(dense, mlp_dim=dense.mlp_dim // 2, n_experts=8, n_active=2)  # two active experts of half the width
wide = replace(dense, mlp_dim=moe.mlp_dim * moe.n_experts)  # every expert's parameters, in one MLP
everything = moe.n_experts / moe.n_active  # the capacity at which every expert takes every token
CASES = {
    "Dense, 14M": (dense, None),
    "Dense, 42M": (wide, None),
    "Experts take their own tokens": (moe, None),
    "Experts take every token": (moe, everything),
    "Capacity 1.5": (moe, 1.5),
    "Capacity 1": (moe, 1.0),
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
    return args.batch_size * args.seq_len / times[len(times) // 2]


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
    return batch * new_tokens / best


if args.case:
    device = pick_device()
    cfg, capacity = CASES[args.case]
    speed = generation_speed(cfg, capacity, device, args.generate) if args.generate else train_speed(cfg, capacity, device)
    print(json.dumps({"device": device, "parameters": Transformer(cfg).num_params(), "tokens_per_second": speed}))
    raise SystemExit


def measure(name, generate=0):
    """
    One case in a process of its own. Compiled code and the pool of GPU
    memory outlive the model they were made for, and a measurement taken
    after another one in the same process came out up to a third slower.
    """
    shape = ["--batch-size", str(args.batch_size), "--seq-len", str(args.seq_len), "--steps", str(args.steps)]
    command = [sys.executable, __file__, "--case", name, "--generate", str(generate)] + shape
    return json.loads(subprocess.run(command, capture_output=True, text=True, check=True).stdout.splitlines()[-1])


results = {"batch_size": args.batch_size, "seq_len": args.seq_len, "train": [], "generate": []}
for name, (cfg, capacity) in CASES.items():
    r = measure(name)
    results["device"] = r.pop("device")
    results["train"].append({"name": name, "capacity": capacity} | r)
    print(f"training    {name:30s} {r['parameters'] / 1e6:5.1f}M {r['tokens_per_second'] / 1e3:6.1f}k tokens/s")
for name, (cfg, capacity) in list(CASES.items())[:4]:
    row = {"name": name, "capacity": capacity} | {f"batch_{batch}": measure(name, batch)["tokens_per_second"] for batch in [1, 64]}
    results["generate"].append(row)
    print(f"generation  {name:30s} {row['batch_1']:6.0f} tokens/s alone, {row['batch_64']:6.0f} in a batch of 64")

args.out = args.out or f"runs/bench/moe-{results['device']}.json"
os.makedirs(os.path.dirname(args.out), exist_ok=True)
with open(args.out, "w") as f:
    json.dump(results, f, indent=1)
print(f"wrote {args.out}")
