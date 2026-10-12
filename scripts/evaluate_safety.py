"""
How often a model refuses: requests it should refuse, and safe requests
that only sound like them, from XSTest.

    uv run scripts/evaluate_safety.py --model qwen
    uv run scripts/evaluate_safety.py --model out/sft-gsm8k            # after fine-tuning on math, every weight
    uv run scripts/evaluate_safety.py --model out/sft-gsm8k-safe       # the same with refusals mixed in

Writes runs/safety/xstest-<model>.json: a verdict per prompt, and the
replies to a few safe prompts. Replies to the unsafe prompts are not
saved. See docs/08a-safety.md.
"""

import argparse
import json
import os
import time

import torch

from minllm import checkpoint, evaluate, safety
from minllm.common import pick_device
from minllm.pretrained import SYSTEM_PROMPT

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
parser.add_argument("--max-new-tokens", type=int, default=96, help="a refusal is read off the opening of a reply")
parser.add_argument("--out", default=None)
args = parser.parse_args()

device = pick_device()
model, tok, meta = checkpoint.load(args.model, device, torch.float32 if device == "cpu" else torch.bfloat16)
prompts = safety.xstest()
examples = [{"messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": p["prompt"]}]} for p in prompts]
start = time.time()
replies = evaluate.replies(model, tok, examples, args.max_new_tokens, 64, progress=lambda d, t: print(f"\r{d}/{t}", end="", flush=True))
refused = [safety.refuses(r) for r in replies]
share = lambda label: sum(x for x, p in zip(refused, prompts) if p["label"] == label) / sum(p["label"] == label for p in prompts)
print(f"\r{args.model} on XSTest: refuses {share('unsafe'):.1%} of the unsafe requests and {share('safe'):.1%} of the safe ones, {time.time() - start:.0f}s")

name = os.path.basename(args.model.rstrip("/"))
out = args.out or f"runs/safety/xstest-{name}.json"
os.makedirs(os.path.dirname(out), exist_ok=True)
result = {"model": args.model, "n": len(prompts), "refuses_unsafe": share("unsafe"), "refuses_safe": share("safe"), "refused": [int(x) for x in refused], "labels": [p["label"] for p in prompts], "types": [p["type"] for p in prompts]}
result["safe_samples"] = [{"prompt": p["prompt"], "reply": r, "refused": x} for p, r, x in zip(prompts, replies, refused) if p["label"] == "safe"][::25]
with open(out, "w") as f:
    json.dump(result, f, indent=1, ensure_ascii=False)
print(f"wrote {out}")
