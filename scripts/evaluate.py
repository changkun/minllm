"""
Evaluate a model on a task, or compare two evaluations.

    uv run scripts/evaluate.py --model out/sft-words --task words
    uv run scripts/evaluate.py --model qwen --task gsm8k
    uv run scripts/evaluate.py --compare runs/eval/gsm8k-qwen.json runs/eval/gsm8k-sft-gsm8k-lora.json

Writes runs/eval/<task>-<model>.json, with every score and a few replies.
See docs/07-evaluation.md.
"""

import argparse
import json
import os
import time

import torch

from minllm import checkpoint, evaluate, tasks
from minllm.common import pick_device
from minllm.pretrained import SYSTEM_PROMPT

parser = argparse.ArgumentParser()
parser.add_argument("--model", default=None, help="qwen, or the directory of a trained model")
parser.add_argument("--task", choices=["words", "taste", "gsm8k"])
parser.add_argument("--n", type=int, default=None, help="number of test examples; all of them by default")
parser.add_argument("--batch-size", type=int, default=None)
parser.add_argument("--temperature", type=float, default=0.0, help="0 is greedy, which makes the evaluation repeatable")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--out", default=None)
parser.add_argument("--compare", nargs=2, metavar="RESULT", help="two result files of the same task and examples")
args = parser.parse_args()

if args.compare:
    a, b = (json.load(open(path)) for path in args.compare)
    assert a["task"] == b["task"] and len(a["scores"]) == len(b["scores"]), "the two runs must share their examples"
    diff, low, high = evaluate.paired_difference(a["scores"], b["scores"])
    for r in (a, b):
        print(f"{r['model']:28s} {r['mean']:6.1%} ± {r['error']:.1%}")
    verdict = "a real difference" if low > 0 or high < 0 else "within noise"
    print(f"difference {diff:+.1%}, 95% interval {low:+.1%} to {high:+.1%}: {verdict}")
    raise SystemExit

device = pick_device()
torch.manual_seed(args.seed)
model, tok, meta = checkpoint.load(args.model, device, torch.float32 if device == "cpu" else torch.bfloat16)
examples = tasks.test_examples(args.task, args.n, SYSTEM_PROMPT)
batch_size = args.batch_size or (256 if meta["family"] == "small" else 64)

start = time.time()
show = lambda done, total: print(f"\r{done}/{total}", end="", flush=True)
replies = evaluate.replies(model, tok, examples, tasks.MAX_NEW_TOKENS[args.task], batch_size, args.temperature, progress=show)
seconds = time.time() - start
scores = [tasks.SCORE[args.task](e, r) for e, r in zip(examples, replies)]
mean, error = evaluate.mean_and_error(scores)
print(f"\r{args.model} on {args.task}: {mean:.1%} ± {error:.1%} over {len(scores)} examples, {seconds:.0f}s")

name = os.path.basename(args.model.rstrip("/"))
out = args.out or f"runs/eval/{args.task}-{name}.json"
os.makedirs(os.path.dirname(out), exist_ok=True)
samples = [{"prompt": e["messages"][-1]["content"], "reply": r, "score": s} for e, r, s in list(zip(examples, replies, scores))[:10]]
result = {"model": args.model, "task": args.task, "temperature": args.temperature, "n": len(scores)}
if "words" in examples[0]:
    result["requested_words"] = [len(e["words"]) for e in examples]  # to break the score down by difficulty
with open(out, "w") as f:
    json.dump(result | {"mean": mean, "error": error, "seconds": seconds, "scores": scores, "samples": samples}, f, indent=1)
print(f"wrote {out}")
