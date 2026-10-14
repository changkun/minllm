"""
Measure a model as a judge, against verdicts that are known.

    uv run scripts/judge.py --task gsm8k                       # is this solution right? the 0.5B model judges its own replies
    uv run scripts/judge.py --task gsm8k --reference           # the same, told the right answer
    uv run scripts/judge.py --task stories                     # which of two stories is better? against the annotator of stage 8
    uv run scripts/judge.py --task stories --rubric            # the same, told what the annotator likes
    uv run scripts/judge.py --task stories --judge data/qwen2.5-1.5b-instruct   # a judge three times the size

The judge is qwen, a directory of a model trained here, or the directory of
a published Qwen2.5 model. For stories every pair is judged in both orders.
Writes runs/judge/<task>-<judge>[-reference|-rubric].json. See docs/07b-judge.md.
"""

import argparse
import json
import os
import time

import torch

from minllm import checkpoint, judge, preferences, pretrained
from minllm.common import pick_device, result_path
from minllm.pretrained import SYSTEM_PROMPT

parser = argparse.ArgumentParser()
parser.add_argument("--task", choices=["gsm8k", "stories"], required=True)
parser.add_argument("--judge", default="qwen")
parser.add_argument("--reference", action="store_true", help="gsm8k: tell the judge the right answer")
parser.add_argument("--rubric", action="store_true", help="stories: tell the judge what the annotator likes")
parser.add_argument("--replies", default="out/eval/gsm8k-qwen.jsonl", help="gsm8k: the replies to judge, as scripts/evaluate.py saves them")
parser.add_argument("--out", default=None)
args = parser.parse_args()

device = pick_device()
dtype = torch.float32 if device == "cpu" else torch.bfloat16
if os.path.exists(f"{args.judge}/config.json"):  # a published model in its own directory
    model, tok = pretrained.load_model(args.judge, device, dtype).eval(), pretrained.load_tokenizer(args.judge)
else:
    model, tok, _ = checkpoint.load(args.judge, device, dtype)
progress = lambda done, total: print(f"\r{done}/{total}", end="", flush=True)
start = time.time()
result = {"task": args.task, "judge": args.judge}

if args.task == "gsm8k":
    with open(args.replies, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    with open("data/gsm8k/test.jsonl", encoding="utf-8") as f:
        numbers = [json.loads(line)["answer"].split("####")[-1].strip().replace(",", "") for line in f]
    assert len(rows) == len(numbers), "the replies have to be to all test problems, in order"
    problem = lambda prompt: prompt.split("\n\nShow your work")[0]  # the problem without stage 7's instruction
    conversations = [judge.correctness(problem(r["prompt"]), r["reply"], n if args.reference else None) for r, n in zip(rows, numbers)]
    p = judge.verdicts(model, tok, conversations, judge.CORRECT, SYSTEM_PROMPT, progress=progress)[:, 0].tolist()
    truth = [r["score"] for r in rows]
    approved = [x >= 0.5 for x in p]
    right = [t == 1 for t in truth]
    accuracy = sum(a == r for a, r in zip(approved, right)) / len(p)
    result |= {"replies": args.replies, "reference": args.reference, "n": len(p), "accuracy": accuracy, "approves_right": sum(a for a, r in zip(approved, right) if r) / sum(right), "approves_wrong": sum(a for a, r in zip(approved, right) if not r) / (len(right) - sum(right)), "approved": sum(approved) / len(p), "truly_right": sum(right) / len(p)}
    result |= {"p_correct": [round(x, 4) for x in p], "truth": truth}
    print(f"\r{args.judge} judges GSM8K replies{' with the answer' if args.reference else ''}: agrees {accuracy:.1%}, approves {result['approves_right']:.1%} of the right ones and {result['approves_wrong']:.1%} of the wrong ones")
else:
    pairs = preferences.load("data/preferences/held_out.jsonl")
    rubric = judge.RUBRIC if args.rubric else None
    instruction = lambda pair: pair["messages"][-1]["content"]
    conversations = [judge.comparison(instruction(pair), a, b, rubric) for pair in pairs for a, b in [pair["replies"], pair["replies"][::-1]]]
    p = judge.verdicts(model, tok, conversations, judge.BETTER, SYSTEM_PROMPT, progress=progress)[:, 0].tolist()
    # p of "A": for the pair as it is, and with the stories swapped. Turned into: the judge's preference for the first story of the pair.
    first_as_a, first_as_b = p[0::2], [1 - x for x in p[1::2]]
    prefers_first = [(x + y) / 2 for x, y in zip(first_as_a, first_as_b)]
    annotator = [1 - pair["preferred"] for pair in pairs]  # 1 if the annotator picked the first story
    tastes = [[preferences.taste(s) for s in pair["replies"]] for pair in pairs]
    better = [1 if t[0] > t[1] else 0 if t[0] < t[1] else None for t in tastes]  # by the hidden taste itself
    words = [[len(s.split()) for s in pair["replies"]] for pair in pairs]
    n = len(pairs)
    agree = lambda picks, target: sum((x >= 0.5) == bool(y) for x, y in zip(picks, target) if y is not None) / sum(y is not None for y in target)
    consistent = sum((x >= 0.5) == (y >= 0.5) for x, y in zip(first_as_a, first_as_b)) / n
    result |= {"rubric": args.rubric, "n": n}
    result |= {
        "agrees_with_annotator": agree(prefers_first, annotator),
        "agrees_with_taste": agree(prefers_first, better),
        "agrees_with_annotator_as_shown": agree(first_as_a, annotator),  # one order only, as a single judgment would be
        "consistent_when_swapped": consistent,
        "picks_slot_a": sum(x >= 0.5 for x in p[0::2] + p[1::2]) / (2 * n),
        "picks_longer": sum((x >= 0.5) == (w[0] > w[1]) for x, w in zip(prefers_first, words) if w[0] != w[1]) / sum(w[0] != w[1] for w in words),
        "annotator_picks_longer": sum(bool(a) == (w[0] > w[1]) for a, w in zip(annotator, words) if w[0] != w[1]) / sum(w[0] != w[1] for w in words),
    }
    result |= {"p_first_as_a": [round(x, 4) for x in first_as_a], "p_first_as_b": [round(x, 4) for x in first_as_b], "annotator": annotator, "tastes": tastes}
    print(f"\r{args.judge} judges stories{' with the rubric' if args.rubric else ''}: agrees with the annotator {result['agrees_with_annotator']:.1%}, with the taste {result['agrees_with_taste']:.1%}, the same verdict in both orders {consistent:.1%}, picks the story shown first {result['picks_slot_a']:.1%}")

result["seconds"] = time.time() - start
name = os.path.basename(args.judge.rstrip("/"))
out = result_path(args.out or f"runs/judge/{args.task}-{name}{'-reference' if args.reference else ''}{'-rubric' if args.rubric else ''}.json")
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "w") as f:
    json.dump(result, f)
print(f"wrote {out}")
