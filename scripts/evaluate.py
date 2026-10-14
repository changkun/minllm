"""
Evaluate a model on a task, or compare two evaluations.

    uv run scripts/evaluate.py --model out/sft-words --task words
    uv run scripts/evaluate.py --model qwen --task gsm8k
    uv run scripts/evaluate.py --model qwen --task gsm8k --bits 6    # with its weights rounded to 6 bits
    uv run scripts/evaluate.py --compare runs/eval/gsm8k-qwen.json runs/eval/gsm8k-sft-gsm8k-lora.json

Writes runs/eval/<task>-<model>.json, with every score and a few replies, and
all replies to out/eval/, for a closer look. See docs/07-evaluation.md.
"""

import argparse
import json
import os
import time

import torch

from minllm import checkpoint, evaluate, quantize, tasks
from minllm.common import pick_device, result_path
from minllm.pretrained import SYSTEM_PROMPT

parser = argparse.ArgumentParser()
parser.add_argument("--model", default=None, help="qwen, or the directory of a trained model")
parser.add_argument("--task", choices=["words", "taste", "gsm8k"])
parser.add_argument("--n", type=int, default=None, help="number of test examples; all of them by default")
parser.add_argument("--batch-size", type=int, default=None)
parser.add_argument("--temperature", type=float, default=0.0, help="0 is greedy, which makes the evaluation repeatable")
parser.add_argument("--max-new-tokens", type=int, default=None, help="the limit on a reply's length; the task's own by default")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--bits", type=int, default=0, help="round the weights to this many bits first, in groups of --group (docs/10a-quantization.md)")
parser.add_argument("--group", type=int, default=64)
parser.add_argument("--out", default=None)
parser.add_argument("--compare", nargs=2, metavar="RESULT", help="two result files of the same task and examples")
args = parser.parse_args()



def show(task, value, signed=False):
    """Scores are shares of 1, except taste, which is a number on the annotator's own scale."""
    if task == "taste":
        return f"{value:+.3f}"
    return f"{value:+.1%}" if signed else f"{value:.1%}"


if args.compare:
    a, b = (json.load(open(result_path(path))) for path in args.compare)
    assert a["task"] == b["task"] and len(a["scores"]) == len(b["scores"]), "the two runs must share their examples"
    diff, low, high = evaluate.paired_difference(a["scores"], b["scores"])
    for r in (a, b):
        print(f"{r['model']:28s} {show(r['task'], r['mean']):>7s} ± {show(r['task'], r['error']).lstrip('+')}")
    verdict = "a real difference" if low > 0 or high < 0 else "within noise"
    task = a["task"]
    print(f"difference {show(task, diff, True)}, 95% interval {show(task, low, True)} to {show(task, high, True)}: {verdict}")
    raise SystemExit

device = pick_device()
torch.manual_seed(args.seed)
model, tok, meta = checkpoint.load(args.model, device, torch.float32 if device == "cpu" else torch.bfloat16)
if meta.get("quant"):
    model = quantize.expand(model)  # the same rounded weights at full size: an evaluation wants speed, not memory
if args.bits:
    model = quantize.rounded(model.float(), args.bits, args.group)  # what the model would compute if it were stored that way
examples = tasks.test_examples(args.task, args.n, SYSTEM_PROMPT)
batch_size = args.batch_size or (256 if meta["family"] == "small" else 64)

start = time.time()
progress = lambda done, total: print(f"\r{done}/{total}", end="", flush=True)
limit = args.max_new_tokens or tasks.MAX_NEW_TOKENS[args.task]
replies = evaluate.replies(model, tok, examples, limit, batch_size, args.temperature, progress=progress)
seconds = time.time() - start
scores = [tasks.SCORE[args.task](e, r) for e, r in zip(examples, replies)]
mean, error = evaluate.mean_and_error(scores)
# A reply that ran into the token limit was cut off, and is scored on what it got to say.
cut_off = sum(len(tok.encode_ordinary(r)) >= limit - 2 for r in replies)
print(f"\r{args.model} on {args.task}: {show(args.task, mean)} ± {show(args.task, error).lstrip('+')} over {len(scores)} examples, {seconds:.0f}s, {cut_off} replies cut off")

name = os.path.basename(args.model.rstrip("/")) + (f"-{args.bits}bit" if args.bits else "")
out = result_path(args.out or f"runs/eval/{args.task}-{name}.json")
os.makedirs(os.path.dirname(out), exist_ok=True)
samples = [{"prompt": e["messages"][-1]["content"], "reply": r, "score": s} for e, r, s in list(zip(examples, replies, scores))[:10]]
result = {"model": args.model, "task": args.task, "temperature": args.temperature, "max_new_tokens": limit, "n": len(scores), "cut_off": cut_off}
if args.bits:
    result |= {"bits": args.bits, "group": args.group}
if "words" in examples[0]:
    result["requested_words"] = [len(e["words"]) for e in examples]  # to break the score down by difficulty
with open(out, "w") as f:
    json.dump(result | {"mean": mean, "error": error, "seconds": seconds, "scores": scores, "samples": samples}, f, indent=1)
os.makedirs("out/eval", exist_ok=True)
with open(f"out/eval/{os.path.basename(out)}l", "w", encoding="utf-8") as f:
    for e, r, s in zip(examples, replies, scores):
        f.write(json.dumps({"prompt": e["messages"][-1]["content"], "reply": r, "score": s}, ensure_ascii=False) + "\n")
print(f"wrote {out}")
