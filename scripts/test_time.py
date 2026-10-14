"""
Spend more computation on GSM8K's test problems when they are asked:
sample several replies and take the most common answer, or let the model
think only so long and make it answer there.

    uv run scripts/test_time.py --model qwen
    uv run scripts/test_time.py --model out/grpo-qwen
    uv run scripts/test_time.py --model out/sft-gsm8k-traces --max-new-tokens 2560 --budgets 64,128,256,512,1024,2048

For --n test problems: one greedy reply each, and its answer when the
reply is cut after every budget of tokens; and --samples sampled replies
each, from which majority@k and pass@k are computed for k = 1, 2, 4, ...
Writes runs/reasoning/test-time-<model>.json. See docs/09c-reasoning.md.
"""

import argparse
import json
import math
import os
import statistics
import time

import torch

from minllm import chat, checkpoint, evaluate, reasoning, tasks
from minllm.common import pick_device, result_path
from minllm.data import EOT
from minllm.pretrained import SYSTEM_PROMPT

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
parser.add_argument("--n", type=int, default=500, help="the first test problems")
parser.add_argument("--samples", type=int, default=16)
parser.add_argument("--temperature", type=float, default=1.0, help="of the sampled replies; the RL rollouts of stage 9 sample at 1 too")
parser.add_argument("--max-new-tokens", type=int, default=512)
parser.add_argument("--budgets", default="16,32,64,128,256,512")
parser.add_argument("--batch-size", type=int, default=128)
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

device = pick_device()
torch.manual_seed(args.seed)
model, tok, _ = checkpoint.load(args.model, device, torch.bfloat16)
problems = tasks.gsm8k_examples("test", SYSTEM_PROMPT)[: args.n]
truths = [float(p["number"]) for p in problems]
stop = [tok.special_tokens[chat.IM_END], tok.special_tokens[EOT]]
start = time.time()
progress = lambda done, total: print(f"\r{done}/{total}", end="", flush=True)

# One greedy reply each: the score as stage 7 measures it, and the replies to cut.
greedy = evaluate.replies(model, tok, problems, args.max_new_tokens, batch_size=args.batch_size)
greedy_ids = [tok.encode_ordinary(r) for r in greedy]
right = [tasks.gsm8k_number(r) == t for r, t in zip(greedy, truths)]
print(f"\rgreedy: {sum(right) / len(right):.1%}, {statistics.mean(map(len, greedy_ids)):.0f} tokens on average")

# Thinking cut at a budget, then the answer asked for: after the end of the thinking if the model thinks in a block.
thinks = sum(reasoning.END_THINK in r for r in greedy) > len(greedy) / 2
cue = tok.encode_ordinary(f"\n{reasoning.END_THINK}\n\n#### " if thinks else "\n#### ")
prompts = [chat.encode(tok, p["messages"]) for p in problems]
budgets = {}
for budget in [int(b) for b in args.budgets.split(",")]:
    with torch.autocast(device_type=torch.device(device).type, dtype=torch.bfloat16, enabled=device != "cpu"):
        answered = reasoning.forced_answers(model, prompts, greedy_ids, budget, cue, stop, batch_size=args.batch_size)
    texts = [tok.decode(a) for a in answered]
    score = sum(tasks.gsm8k_number(t) == truth for t, truth in zip(texts, truths)) / len(truths)
    used = statistics.mean(min(len(r), budget) for r in greedy_ids)
    budgets[budget] = {"right": score, "tokens": used}
    print(f"budget {budget:5d}: {score:.1%}, {used:.0f} tokens of thinking on average", flush=True)

# Several sampled replies each: the vote, and what at least one of them got right.
repeated = [p for p in problems for _ in range(args.samples)]
sampled = evaluate.replies(model, tok, repeated, args.max_new_tokens, batch_size=args.batch_size, temperature=args.temperature, progress=progress)
numbers = [[tasks.gsm8k_number(r) for r in sampled[i * args.samples : (i + 1) * args.samples]] for i in range(len(problems))]
ks = [k for k in (1, 2, 4, 8, 16, 32, 64) if k <= args.samples]
curves = reasoning.vote_curves(numbers, truths, ks)
lengths = [len(tok.encode_ordinary(r)) for r in sampled]
for k, vote, passed in zip(ks, curves["majority"], curves["pass"]):
    print(f"\rk = {k:2d}: majority {vote:.1%}, pass {passed:.1%}")

name = os.path.basename(args.model.rstrip("/"))
os.makedirs(result_path("runs/reasoning"), exist_ok=True)
result = {"model": args.model, "n": len(problems), "max_new_tokens": args.max_new_tokens, "temperature": args.temperature, "samples": args.samples, "minutes": (time.time() - start) / 60}
result |= {"greedy": {"right": sum(right) / len(right), "tokens": statistics.mean(map(len, greedy_ids)), "scores": [int(r) for r in right], "lengths": list(map(len, greedy_ids))}}
result |= {"thinks_in_a_block": thinks, "budgets": budgets, "ks": ks, "majority": curves["majority"], "pass": curves["pass"], "sampled_tokens": statistics.mean(lengths)}
finite = [[x if x is None or math.isfinite(x) else None for x in row] for row in numbers]  # a number too long to read is written as none, which JSON can hold
result |= {"numbers": finite, "samples_shown": [{"problem": i, "reply": greedy[i]} for i in range(0, len(problems), max(1, len(problems) // 10))]}
with open(result_path(f"runs/reasoning/test-time-{name}.json"), "w") as f:
    json.dump(result, f, indent=1, ensure_ascii=False)
print("wrote", result_path(f"runs/reasoning/test-time-{name}.json"))
