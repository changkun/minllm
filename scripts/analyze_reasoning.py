"""
Compare the models of docs/09c-reasoning.md problem by problem on GSM8K's
test set: GRPO with room to think, and the student fine-tuned on a
reasoning model's thinking, each against the published model; and all
three on the first test problems, against the teacher, which answered
those. Of the replies that ran into their limit, it counts how many were
caught in a loop, their last 200 characters written before, for the
student and for the teacher; and of the replies cut at a budget, how many
were right made to answer there, and how many when left to run on.

    uv run scripts/analyze_reasoning.py   # reads runs/eval/, runs/reasoning/, out/eval/ and data/reasoning/; writes runs/reasoning/summary.json

See docs/09c-reasoning.md.
"""

import json

from minllm import evaluate, pretrained
from minllm.common import result_path

MODELS = {"0.5B as published": "gsm8k-qwen", "after GRPO with room to think": "gsm8k-grpo-qwen", "fine-tuned on the teacher's thinking": "gsm8k-sft-gsm8k-traces"}
TAIL = 200


def load(name):
    with open(result_path(f"runs/eval/{name}.json")) as f:
        return json.load(f)


def looping(text):
    """Whether a text ends in what it already wrote: its last TAIL characters appear before them."""
    return len(text) > 2 * TAIL and text[-TAIL:] in text[:-TAIL]


def kinds(replies):
    """How many (right, ended, looping) replies were right, wrong after they ended, or wrong and cut off in a loop or not."""
    count = lambda f: sum(1 for right, ended, loops in replies if f(right, ended, loops))
    return {
        "right": count(lambda right, ended, loops: right),
        "wrong_ended": count(lambda right, ended, loops: not right and ended),
        "wrong_cut_off_looping": count(lambda right, ended, loops: not right and not ended and loops),
        "wrong_cut_off_otherwise": count(lambda right, ended, loops: not right and not ended and not loops),
    }


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
        "replies": kinds([(row["score"] > 0, not c, looping(row["reply"])) for row, c in zip(rows, cut)]),
    }
    loops = [row["reply"] for row, c in zip(rows, cut) if c and looping(row["reply"])]
    if loops:
        summary["runs"][name]["loop_example"] = loops[0][-3 * TAIL :]  # how the first of them ends
    print(name, {k: v for k, v in summary["runs"][name].items() if k != "loop_example"})

# The first test problems, which the teacher answered: one reply each, sampled at its advised temperature.
teacher = json.load(open(result_path("runs/reasoning/teacher-test.json")))
n = teacher["n"]
with open("data/reasoning/replies-test.jsonl", encoding="utf-8") as f:
    replies = {r["index"]: r for r in map(json.loads, f)}
unfinished = [replies[i]["text"] for i in range(n) if not replies[i]["ended"]]
summary["first"] = {"n": n, "teacher": {
    "mean": teacher["right"], "max_new_tokens": teacher["max_new_tokens"], "temperature": teacher["temperature"], "cut_off": len(unfinished), "cut_off_looping": sum(map(looping, unfinished)),
    "replies": kinds([(replies[i]["right"], replies[i]["ended"], looping(replies[i]["text"])) for i in range(n)]),
}}
for name in MODELS.values():
    summary["first"][name] = {"mean": sum(load(name)["scores"][:n]) / n}
print(f"first {n} test problems:", summary["first"])

# The replies a budget cut, right when made to answer there against right when left to run on to the limit.
# A reply that ended within the budget keeps its own ending, so the difference is all in the replies cut.
summary["budgets"] = {}
for name in ("qwen", "grpo-qwen", "sft-gsm8k-traces"):
    r = json.load(open(result_path(f"runs/reasoning/test-time-{name}.json")))
    lengths, scores = r["greedy"]["lengths"], r["greedy"]["scores"]
    summary["budgets"][name] = {}
    for b, v in r["budgets"].items():
        cut = [i for i, n in enumerate(lengths) if n > int(b)]
        kept = sum(scores) - sum(scores[i] for i in cut)
        summary["budgets"][name][b] = {"cut": len(cut), "right_ran_on": sum(scores[i] for i in cut), "right_answered": round(v["right"] * r["n"]) - kept}
    print(name, "budgets:", summary["budgets"][name])

# The same greedy replies twice: stage 7's evaluation of all the test problems, and test_time.py's of the first ones, in other batches.
summary["greedy_twice"] = {}
for name, short in zip(MODELS.values(), ("qwen", "grpo-qwen", "sft-gsm8k-traces")):
    r = json.load(open(result_path(f"runs/reasoning/test-time-{short}.json")))
    once, again = load(name)["scores"][: r["n"]], r["greedy"]["scores"]
    summary["greedy_twice"][name] = {"n": r["n"], "evaluated": sum(once) / r["n"], "test_time": sum(again) / r["n"], "problems_differing": sum(bool(a) != bool(b) for a, b in zip(once, again))}
print("greedy twice:", summary["greedy_twice"])

with open(result_path("runs/reasoning/summary.json"), "w") as f:
    json.dump(summary, f, indent=1)
print("wrote", result_path("runs/reasoning/summary.json"))
