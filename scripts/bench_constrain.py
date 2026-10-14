"""
What constrained decoding costs per step: the same 64 prompts, the same
number of steps, with and without a constraint. The constraint is free text
of any length, which allows nearly every token everywhere, so the model's
work is the same and only the deciding is extra.

    uv run scripts/bench_constrain.py      # writes runs/bench/constrain-<device>.json

Each case runs in its own process and the fastest of three is kept, as in
the other benchmarks. See docs/04a-structured-output.md.
"""

import argparse
import json
import os
import subprocess
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument("--case", choices=["free", "constrained"], default=None, help="run one case in this process")
parser.add_argument("--steps", type=int, default=256)
parser.add_argument("--batch", type=int, default=64)
parser.add_argument("--repeats", type=int, default=3)
args = parser.parse_args()

if args.case is None:
    from minllm.common import result_path

    results = {}
    for case in ["free", "constrained"]:
        runs = [json.loads(subprocess.run([sys.executable, __file__, "--case", case, "--steps", str(args.steps), "--batch", str(args.batch)], capture_output=True, text=True, check=True).stdout.strip().splitlines()[-1]) for _ in range(args.repeats)]
        results[case] = min(runs, key=lambda r: r["seconds"])
        print(f"{case}: {results[case]['seconds']:.1f}s for {args.steps} steps of {args.batch}, {args.steps * args.batch / results[case]['seconds']:.0f} tokens per second")
    device = results["free"]["device"]
    out = result_path(f"runs/bench/constrain-{device}.json")
    os.makedirs(result_path("runs/bench"), exist_ok=True)
    with open(out, "w") as f:
        json.dump({"steps": args.steps, "batch": args.batch, "repeats": args.repeats} | results, f, indent=1)
    print(f"slower by {results['constrained']['seconds'] / results['free']['seconds'] - 1:.0%}; wrote {out}")
    raise SystemExit

import torch

from minllm import chat, checkpoint, constrain, inference, schema, tasks
from minllm.automaton import Automaton
from minllm.common import pick_device
from minllm.pretrained import SYSTEM_PROMPT

device = pick_device()
model, tok, _ = checkpoint.load("qwen", device, torch.bfloat16 if device != "cpu" else torch.float32)
prompts = [chat.encode(tok, e["messages"]) for e in tasks.gsm8k_examples("test", SYSTEM_PROMPT)[: args.batch]]
allow = None
built = 0.0
if args.case == "constrained":
    start = time.time()
    index = constrain.Index(Automaton(schema.TEXT), tok, model.cfg.vocab_size)
    built = time.time() - start
    allow = constrain.Constraint(index, len(prompts), stop=tok.special_tokens[chat.IM_END], budget=args.steps + 10, device=next(model.parameters()).device)
with torch.autocast(device_type=torch.device(device).type, dtype=torch.bfloat16, enabled=device != "cpu"):
    inference.generate(model, prompts, 8, temperature=0)  # warm up
    if device == "mps":
        torch.mps.synchronize()
    start = time.time()
    inference.generate(model, prompts, args.steps, temperature=0, allow=allow)  # no stop token: every row takes every step
    if device == "mps":
        torch.mps.synchronize()
print(json.dumps({"device": device, "seconds": time.time() - start, "index_seconds": built}))
