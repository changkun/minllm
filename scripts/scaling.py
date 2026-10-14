"""
Scaling laws: train the family of minllm/scaling.py at several sizes and
compute budgets, each run with scripts/train.py and its own schedule, and
keep the logs in runs/scaling/.

    uv run scripts/scaling.py --phase lr       # widths 128 to 512 at four learning rates, 16M tokens each
    uv run scripts/scaling.py --phase lr --dims 64,128,192 --lrs 1e-3,2e-3,4e-3,8e-3   # the smallest ones go higher
    uv run scripts/scaling.py --phase grid     # every size at every budget, at its best learning rate
    uv run scripts/scaling.py --phase grid --dims 64,128,192,256,320,384,448,512 --lrs 5e-4,1e-3,2e-3,4e-3,8e-3 --max-passes 1.7
                                               # the small side of every budget, which reads the text more than once
    uv run scripts/analyze_scaling.py          # the fits

A run that has a log in runs/scaling/ is skipped, so an interrupted queue
continues where it stopped. See docs/03c-scaling.md.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys

from minllm import scaling
from minllm.data import load_tokens
from minllm.model import Transformer
from minllm.common import result_path

parser = argparse.ArgumentParser()
parser.add_argument("--phase", choices=["lr", "grid", "list"], required=True)
parser.add_argument("--dims", default="128,192,256,320,384,448,512")
parser.add_argument("--lrs", default="5e-4,1e-3,2e-3,4e-3")
parser.add_argument("--lr-tokens", type=float, default=16e6, help="tokens of every run of the learning rate sweep")
parser.add_argument("--budgets", default="3e14,1e15,3e15,1e16", help="compute budgets in floating-point operations")
parser.add_argument("--max-passes", type=float, default=1.0, help="no run reads more than this many passes over the training text")
parser.add_argument("--min-tokens", type=float, default=8e6, help="no run is shorter than this: about 500 steps")
parser.add_argument("--runs", default="runs/scaling")
args = parser.parse_args()
args.runs = result_path(args.runs)  # a path under runs/ goes to the results folder

BATCH, SEQ = 64, 256  # stage 3's batch: 16,384 tokens a step
dims = [int(d) for d in args.dims.split(",")]
params = {d: Transformer(scaling.family(d)).num_params() for d in dims}
data_tokens = len(load_tokens("data/tinystories/train.bin"))
os.makedirs(args.runs, exist_ok=True)


def run(name, dim, steps, lr):
    """One run of scripts/train.py, its log kept and its checkpoint dropped."""
    log = f"{args.runs}/{name}.jsonl"
    if os.path.exists(log):
        with open(log) as f:
            done_at = json.loads(f.readline())["run"]["lr"]
        if done_at == lr or name.startswith("lr-"):
            print(f"{name}: done before, skipped", flush=True)
            return
        print(f"{name}: done before at lr {done_at:g}, which is no longer this size's best", flush=True)
    cfg, out = scaling.family(dim), f"out/scaling/{name}"
    shutil.rmtree(out, ignore_errors=True)
    command = [sys.executable, "scripts/train.py", "--out", out, "--dim", str(dim), "--n-layers", str(cfg.n_layers), "--n-heads", str(cfg.n_heads),
               "--n-kv-heads", str(cfg.n_kv_heads), "--mlp-dim", str(cfg.mlp_dim), "--steps", str(steps), "--warmup", str(max(10, steps // 30)),
               "--lr", str(lr), "--eval-every", str(steps), "--log-every", "100"]  # stage 3 warms up for 200 of 6,000 steps, a thirtieth
    print(f"{name}: {params[dim] / 1e6:.2f}M parameters, {steps} steps, {steps * BATCH * SEQ / 1e6:.0f}M tokens, lr {lr:g}", flush=True)
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL)
    records = [json.loads(line) for line in open(f"{out}/log.jsonl")]
    meta = {"dim": dim, "params": params[dim], "steps": steps, "tokens": steps * BATCH * SEQ, "passes": steps * BATCH * SEQ / data_tokens, "lr": lr, "config": vars(cfg)}
    with open(log, "w") as f:
        f.write(json.dumps({"run": meta}) + "\n")
        f.writelines(json.dumps(r) + "\n" for r in records)
    final = [r for r in records if "val_loss" in r][-1]
    print(f"{name}: val loss {final['val_loss']:.4f} after {final['elapsed'] / 60:.1f} minutes", flush=True)
    shutil.rmtree(out)


def final_loss(name):
    records = [json.loads(line) for line in open(f"{args.runs}/{name}.jsonl")]
    return [r for r in records if "val_loss" in r][-1]["val_loss"]


def best_lr(dim):
    """The learning rate whose sweep run of this size ended lowest, among those that were run."""
    lrs = [float(x) for x in args.lrs.split(",") if os.path.exists(f"{args.runs}/lr-d{dim}-{float(x):g}.jsonl")]
    return min(lrs, key=lambda lr: final_loss(f"lr-d{dim}-{lr:g}"))


def grid():
    """The runs of every budget: each size whose tokens at that budget fit between the limits."""
    for budget in [float(b) for b in args.budgets.split(",")]:
        for dim in dims:
            tokens = budget / (6 * params[dim])
            steps = round(tokens / (BATCH * SEQ))
            if args.min_tokens <= steps * BATCH * SEQ <= args.max_passes * data_tokens:
                yield f"c{budget:.0e}-d{dim}".replace("+", ""), dim, steps


if args.phase == "lr":
    steps = round(args.lr_tokens / (BATCH * SEQ))
    for lr in [float(x) for x in args.lrs.split(",")]:
        for dim in dims:
            run(f"lr-d{dim}-{lr:g}", dim, steps, lr)
elif args.phase == "grid":
    for name, dim, steps in grid():
        run(name, dim, steps, best_lr(dim))
else:
    for name, dim, steps in grid():
        print(name, f"{params[dim] / 1e6:.2f}M", steps, f"{steps * BATCH * SEQ / 1e6:.0f}M tokens")
