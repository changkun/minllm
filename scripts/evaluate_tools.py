"""
Evaluate a model on GSM8K's test problems with a calculator it can call,
running its calls and writing the results back, until it answers.

    uv run scripts/evaluate_tools.py --model qwen --max-new-tokens 512 # as published, told about the tool
    uv run scripts/evaluate_tools.py --model out/sft-gsm8k-tools-lora  # fine-tuned on solutions that call it
    uv run scripts/evaluate_tools.py --model out/sft-gsm8k-tools-lora --no-execute   # every call fails
    uv run scripts/evaluate_tools.py --model qwen --max-new-tokens 512 --system-markers   # the tool instructions tokenized as Qwen's tokenizer does

Writes runs/eval/gsm8k-tools-<model>.json, with every score, the calls of
every problem, and some conversations in full. See docs/06a-tool-calling.md.
"""

import argparse
import json
import os
import time

import torch

from minllm import checkpoint, evaluate, tasks, tools
from minllm.common import pick_device
from minllm.pretrained import SYSTEM_PROMPT

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
parser.add_argument("--n", type=int, default=None)
parser.add_argument("--no-execute", action="store_true", help="answer every call with an error instead of running it")
parser.add_argument("--system-markers", action="store_true", help="write <tool_call> and </tool_call> in the tool instructions as the single tokens Qwen's tokenizer makes of them")
parser.add_argument("--max-new-tokens", type=int, default=320, help="for one turn; the calls make a reply several turns")
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--out", default=None)
args = parser.parse_args()

device = pick_device()
model, tok, meta = checkpoint.load(args.model, device, torch.float32 if device == "cpu" else torch.bfloat16)
problems = tools.gsm8k_conversations("test", SYSTEM_PROMPT)[: args.n]
conversations = [p["messages"][:2] for p in problems]  # the system's turn with the tool, and the problem

start = time.time()
progress = lambda done, total: print(f"\r{done}/{total} answered", end="", flush=True)
finished, counts = tools.run(model, tok, conversations, args.max_new_tokens, batch_size=args.batch_size, execute=not args.no_execute, progress=progress, system_markers=args.system_markers)
seconds = time.time() - start

answers = [tools.answer_text(c, 2) for c in finished]
scores = [tasks.gsm8k_score({"number": p["number"]}, a) for p, a in zip(problems, answers)]
mean, error = evaluate.mean_and_error(scores)
calls = [c["calls"] for c in counts]
written = [tools.written_calculations(a) for a in answers]
n = len(scores)
print(f"\r{args.model} on GSM8K with a calculator{'' if not args.no_execute else ' that fails'}: {mean:.1%} ± {error:.1%}, {sum(c > 0 for c in calls) / n:.1%} of problems with a call, {sum(calls) / n:.2f} calls each, {sum(c['bad_calls'] for c in counts)} malformed, {seconds:.0f}s")

name = os.path.basename(args.model.rstrip("/")) + ("-no-execute" if args.no_execute else "") + ("-system-markers" if args.system_markers else "")
out = args.out or f"runs/eval/gsm8k-tools-{name}.json"
os.makedirs(os.path.dirname(out), exist_ok=True)
system_tokens = len(tools.encode(tok, conversations[0][:1], add_generation_prompt=False, system_markers=args.system_markers)[0])  # the system's turn, the same in every prompt
result = {"model": args.model, "task": "gsm8k-tools", "execute": not args.no_execute, "system_markers": args.system_markers, "system_tokens": system_tokens, "max_new_tokens": args.max_new_tokens, "n": n, "mean": mean, "error": error, "seconds": seconds}
result |= {"scores": scores, "calls": calls, "bad_calls": [c["bad_calls"] for c in counts], "turns": [c["turns"] for c in counts]}
result |= {"written": [[sum(right for *_, right in w), len(w)] for w in written]}  # calculations written out in the text, right and all
result |= {"samples": [{"conversation": c[2:], "number": p["number"], "score": s} for c, p, s in list(zip(finished, problems, scores))[:: max(1, n // 25)]]}
with open(out, "w") as f:
    json.dump(result, f, indent=1, ensure_ascii=False)
os.makedirs("out/eval", exist_ok=True)
with open(f"out/eval/{os.path.basename(out)}l", "w", encoding="utf-8") as f:
    for c, s in zip(finished, scores):
        f.write(json.dumps({"conversation": c[1:], "score": s}, ensure_ascii=False) + "\n")
print(f"wrote {out}")
