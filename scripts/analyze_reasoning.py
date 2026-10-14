"""
Compare the models of docs/09c-reasoning.md problem by problem on GSM8K's
test set: GRPO with room to think, and the student fine-tuned on a
reasoning model's thinking, each against the published model; and all
three on the first test problems, against the teacher, which answered
those. Of the replies that ran into their limit, it counts how many were
caught in a loop, their last 200 characters written before, for the
student and for the teacher.

    uv run scripts/analyze_reasoning.py   # reads runs/eval/, runs/reasoning/, out/eval/ and data/reasoning/; writes runs/reasoning/summary.json

See docs/09c-reasoning.md.
"""

import json

from minllm import evaluate, pretrained

MODELS = {"published": "gsm8k-qwen", "after GRPO with room to think": "gsm8k-grpo-qwen", "fine-tuned on the teacher's thinking": "gsm8k-sft-gsm8k-traces"}
TAIL = 200


def load(name):
    with open(f"runs/eval/{name}.json") as f:
        return json.load(f)


def looping(text):
    """Whether a text ends in what it already wrote: its last TAIL characters appear before them."""
    return len(text) > 2 * TAIL and text[-TAIL:] in text[:-TAIL]


summary = {"comparisons": {}, "runs": {}}
for label, (a, b) in {
    "GRPO with room to think, against the published model": ("gsm8k-qwen", "gsm8k-grpo-qwen"),
    "the student, against the published model": ("gsm8k-qwen", "gsm8k-sft-gsm8k-traces"),
}.items():
    difference, low, high = evaluate.paired_difference(load(a)["scores"], load(b)["scores"])
    summary["comparisons"][label] = {"from": a, "to": b, "difference": difference, "low": low, "high": high}
    print(f"{label}: {difference:+.1%}, 95% interval {low:+.1%} to {high:+.1%}")

# The replies that ran into the limit, counted as scripts/evaluate.py counts them, and the loops among them.
tok = pretrained.load_tokenizer()
for label, name in MODELS.items():
    r = load(name)
    with open(f"out/eval/{name}.jsonl", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    cut = [len(tok.encode_ordinary(row["reply"])) >= r["max_new_tokens"] - 2 for row in rows]
    ended = [row["score"] for row, c in zip(rows, cut) if not c]
    summary["runs"][name] = {
        "label": label, "mean": r["mean"], "error": r["error"], "n": r["n"], "max_new_tokens": r["max_new_tokens"],
        "cut_off": sum(cut), "cut_off_looping": sum(looping(row["reply"]) for row, c in zip(rows, cut) if c), "mean_of_ended": sum(ended) / len(ended),
    }
    print(name, summary["runs"][name])

# The first test problems, which the teacher answered: one reply each, sampled at its advised temperature.
teacher = json.load(open("runs/reasoning/teacher-test.json"))
n = teacher["n"]
with open("data/reasoning/replies-test.jsonl", encoding="utf-8") as f:
    replies = {r["index"]: r for r in map(json.loads, f)}
unfinished = [replies[i]["text"] for i in range(n) if not replies[i]["ended"]]
summary["first"] = {"n": n, "teacher": {"mean": teacher["right"], "max_new_tokens": teacher["max_new_tokens"], "temperature": teacher["temperature"], "cut_off": len(unfinished), "cut_off_looping": sum(map(looping, unfinished))}}
for name in MODELS.values():
    summary["first"][name] = {"mean": sum(load(name)["scores"][:n]) / n}
print(f"first {n} test problems:", summary["first"])

with open("runs/reasoning/summary.json", "w") as f:
    json.dump(summary, f, indent=1)
print("wrote runs/reasoning/summary.json")
