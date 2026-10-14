"""
Work out, from the per-question results of scripts/evaluate_choice.py, what
docs/07a-multiple-choice.md reports: where the probability goes, which
letters are picked, what moving the options round does, how confidence and
accuracy compare, and the score of every subject.

    uv run scripts/analyze_choice.py     # reads runs/eval/mmlu-*.json, writes runs/eval/mmlu-summary.json

The figures and the film of the note read the summary, which is a few
kilobytes, and not the results, which are several megabytes.
"""

import argparse
import json
import math
import os
import re
from collections import Counter

from minllm import evaluate, tasks
from minllm.common import result_path

parser = argparse.ArgumentParser()
parser.add_argument("--dir", default="runs/eval")
parser.add_argument("--out", default="runs/eval/mmlu-summary.json")
args = parser.parse_args()
args.dir = result_path(args.dir)  # a path under runs/ goes to the results folder
args.out = result_path(args.out)  # a path under runs/ goes to the results folder


def load(name):
    with open(f"{args.dir}/mmlu-{name}.json") as f:
        return json.load(f)


def share(values, n):
    return [sum(v == i for v in values) / n for i in range(4)]


standard = load("qwen-letter-plain-5shot")
n = standard["n"]
summary = {"n": n, "subjects": len(standard["subjects"]), "right_letter": share(standard["answer"], n), "runs": {}}

# Every run: its score, which letters it picks, and how much of the model's probability the four letters hold.
names = ["qwen-letter-plain-5shot", "qwen-letter-plain-0shot", "qwen-letter-chat-0shot", "qwen-letter-chat-5shot", "base-letter-plain-0shot", "qwen-q4-letter-plain-5shot", "sft-gsm8k-lora-letter-plain-5shot"]
for name in names:
    r = load(name)
    mass = [sum(math.exp(v) for v in row) for row in r["logprobs"]]
    summary["runs"][name] = {"model": r["model"], "format": r["format"], "shots": r["shots"], "mean": r["mean"], "error": r["error"], "seconds": r["seconds"], "picked_letter": share(r["choice"], n), "on_letters": sum(mass) / n}
    if "first_tokens" in r:
        summary["runs"][name]["first_tokens"] = r["first_tokens"]

# The same model with a change, question by question: here, and on GSM8K, where it has to write (stage 7).
with open(f"{args.dir}/gsm8k-qwen.json") as f:
    gsm8k = json.load(f)
for name, other in [("qwen-q4-letter-plain-5shot", "gsm8k-qwen-q4"), ("sft-gsm8k-lora-letter-plain-5shot", "gsm8k-sft-gsm8k-lora")]:
    difference, low, high = evaluate.paired_difference(standard["scores"], load(name)["scores"])
    summary["runs"][name]["against_published"] = {"difference": difference, "low": low, "high": high}
    with open(f"{args.dir}/{other}.json") as f:
        summary["runs"][name]["on_gsm8k"] = {"mean": json.load(f)["mean"], "published": gsm8k["mean"]}

# The options moved round: order k has the option of position p at position (p + k) mod 4.
orders = [standard] + [load(f"qwen-letter-plain-5shot-rot{k}") for k in (1, 2, 3)]
always = sum(all(r["scores"][q] for r in orders) for q in range(n)) / n
never = sum(not any(r["scores"][q] for r in orders) for q in range(n)) / n
same, added = 0, 0
for q in range(n):
    totals = [0.0] * 4  # per option, its probability summed over the four places it stood in
    for k, r in enumerate(orders):
        for position, logp in enumerate(r["logprobs"][q]):
            totals[(position - k) % 4] += math.exp(logp)
    same += len({(r["choice"][q] - k) % 4 for k, r in enumerate(orders)}) == 1
    added += totals.index(max(totals)) == standard["answer"][q]
by_letter = [[0, 0] for _ in range(4)]
for r in orders:
    for answer, score in zip(r["answer"], r["scores"]):
        by_letter[answer][0] += score
        by_letter[answer][1] += 1
summary["orders"] = {
    "scores": [{"mean": r["mean"], "error": r["error"], "picked_letter": share(r["choice"], n)} for r in orders],
    "right_in_all": always,
    "right_in_none": never,
    "right_in_some": 1 - always - never,
    "same_option_in_all": same / n,
    "added_up": added / n,
    "right_by_letter": [right / count for right, count in by_letter],
}

# Confidence against accuracy, in eight groups from 25% to 100%.
groups = [[] for _ in range(8)]
confidences = []
for row, score in zip(standard["logprobs"], standard["scores"]):
    p = [math.exp(v) for v in row]
    confidence = max(p) / sum(p)  # among the four letters
    confidences.append(confidence)
    groups[min(7, int((confidence - 0.25) / 0.09375))].append((confidence, score))
bins = [{"share": len(g) / n, "confidence": sum(c for c, _ in g) / len(g), "right": sum(s for _, s in g) / len(g)} for g in groups]
summary["calibration"] = {"confidence": sum(confidences) / n, "gap": sum(b["share"] * abs(b["confidence"] - b["right"]) for b in bins), "groups": bins}

# The subjects, each with its interval.
by_subject = {}
for subject, score in zip(standard["subject"], standard["scores"]):
    by_subject.setdefault(standard["subjects"][subject], []).append(score)
rows = []
for subject, scores in by_subject.items():
    mean, error = evaluate.mean_and_error(scores)
    guess = 1.96 * math.sqrt(0.25 * 0.75 / len(scores))  # the interval of guessing on as many questions
    rows.append({"subject": subject, "n": len(scores), "mean": mean, "error": error, "like_guessing": abs(mean - 0.25) <= guess})
summary["by_subject"] = sorted(rows, key=lambda row: row["mean"])

# No options shown: each option's text as a continuation, compared by three rules.
cloze = load("qwen-cloze")
summary["cloze"] = {"seconds": cloze["seconds"], "rules": {name: {"mean": rule["mean"], "error": rule["error"]} for name, rule in cloze["rules"].items()}}
# A question that says "which of the following" cannot be answered without its options.
questions = [example["question"] for example in tasks.mmlu_examples("test")]
summary["cloze"]["questions_that_point_at_their_options"] = sum(bool(re.search(r"following|which of these|which one of", q, re.I)) for q in questions) / len(questions)

# Written replies to some of the questions, and the letter read in a chat turn on the same ones.
written = load("qwen-generate-chat-0shot")
taken, seen = [], Counter()
for q, subject in enumerate(standard["subject"]):  # the first questions of every subject, as --per-subject takes them
    if seen[subject] < written["n"] // len(standard["subjects"]):
        taken.append(q)
        seen[subject] += 1
chat = load("qwen-letter-chat-0shot")
assert [standard["answer"][q] for q in taken] == written["answer"]
readable = [score for score, choice in zip(written["scores"], written["choice"]) if choice >= 0]
difference, low, high = evaluate.paired_difference([chat["scores"][q] for q in taken], written["scores"])
summary["written"] = {
    "n": written["n"],
    "max_new_tokens": written["max_new_tokens"],
    "mean": written["mean"],
    "error": written["error"],
    "unreadable": written["unreadable"],
    "right_of_readable": sum(readable) / len(readable),
    "letter_in_chat_on_the_same": sum(chat["scores"][q] for q in taken) / len(taken),
    "letter_as_text_on_the_same": sum(standard["scores"][q] for q in taken) / len(taken),
    # the letter on the questions whose written reply could be read, which may be the easier ones
    "letter_as_text_on_the_readable": sum(standard["scores"][q] for q, choice in zip(taken, written["choice"]) if choice >= 0) / len(readable),
    "written_against_letter_on_the_readable": dict(zip(("difference", "low", "high"), evaluate.paired_difference(
        [standard["scores"][q] for q, choice in zip(taken, written["choice"]) if choice >= 0], readable))),
    "against_letter_in_chat": {"difference": difference, "low": low, "high": high},
    "first_words": Counter(sample["reply"].split()[0] for sample in written["samples"] if sample["reply"].split()).most_common(5),
    "samples": written["samples"][:12],
}

# A few questions in full, with the model's four probabilities in every order.
summary["shown"] = [
    {"index": int(q), "subject": standard["subjects"][standard["subject"][int(q)]], "question": shown["question"], "options": shown["options"], "answer": standard["answer"][int(q)],
     "orders": [[math.exp(v) for v in r["logprobs"][int(q)]] for r in orders],
     "cloze": {"logprobs": cloze["logprobs"][int(q)], "unconditional": cloze["unconditional"][int(q)], "lengths": cloze["lengths"][int(q)]}}
    for q, shown in standard["shown"].items()
]

with open(args.out, "w") as f:
    json.dump(summary, f, indent=1)
print(f"wrote {args.out}, {os.path.getsize(args.out) / 1e3:.0f} kB")
o = summary["orders"]
print(f"orders: right in all {o['right_in_all']:.1%}, in none {o['right_in_none']:.1%}, same option in all {o['same_option_in_all']:.1%}, added up {o['added_up']:.1%}")
print(f"calibration: says {summary['calibration']['confidence']:.1%}, off by {summary['calibration']['gap']:.1%}")
w = summary["written"]
print(f"written: {w['mean']:.1%} of {w['n']}, {w['unreadable']} unreadable, {w['right_of_readable']:.1%} of the readable right; the letter in chat on the same {w['letter_in_chat_on_the_same']:.1%}")
