"""
Compare the models of docs/08b-dpo.md problem by problem on GSM8K's test
set: DPO on the verifier's pairs, with and without the preferred reply's own
loss, against the model they started from, against self-training on the
same problems (side topic 9a) and against GRPO on as many replies (stage
9); how long the replies of each are; how far every run moved its pairs'
replies; and one of the pairs, to show.

    uv run scripts/analyze_dpo.py   # reads runs/eval/, out/eval/, the logs in out/dpo-gsm8k*/ and data/preferences/gsm8k.jsonl; writes runs/dpo/summary.json

See docs/08b-dpo.md.
"""

import json
import os

from minllm import evaluate, pretrained
from minllm.common import result_path

MODELS = {
    "fine-tuned, the start": "gsm8k-sft-gsm8k-lora",
    "self-training": "gsm8k-self-gsm8k",
    "GRPO": "gsm8k-grpo-gsm8k",
    "DPO, one pass": "gsm8k-dpo-gsm8k-1",
    "DPO, two passes": "gsm8k-dpo-gsm8k",
    "DPO, two passes, lr 1e-5": "gsm8k-dpo-gsm8k-lr1e-5",
    "DPO, two passes, lr 2e-6": "gsm8k-dpo-gsm8k-lr2e-6",
    "DPO with the preferred reply's loss": "gsm8k-dpo-gsm8k-nll",
}
COMPARISONS = {f"{label}, against the start": ("gsm8k-sft-gsm8k-lora", name) for label, name in MODELS.items() if "DPO" in label} | {
    "GRPO, against DPO at lr 1e-5": ("gsm8k-dpo-gsm8k-lr1e-5", "gsm8k-grpo-gsm8k"),
}


def load(name):
    with open(result_path(f"runs/eval/{name}.json")) as f:
        return json.load(f)


missing = [name for name in MODELS.values() if not os.path.exists(result_path(f"runs/eval/{name}.json"))]
assert not missing, f"evaluate these models first, as docs/08b-dpo.md says: {', '.join(missing)}"

summary = {"comparisons": {}, "models": {}}
for label, (a, b) in COMPARISONS.items():
    difference, low, high = evaluate.paired_difference(load(a)["scores"], load(b)["scores"])
    summary["comparisons"][label] = {"from": a, "to": b, "difference": difference, "low": low, "high": high}
    print(f"{label}: {difference:+.1%}, 95% interval {low:+.1%} to {high:+.1%}")

# Of the test problems, those each DPO model solves where the start did not, and the other way round.
start = load("gsm8k-sft-gsm8k-lora")["scores"]
summary["changed"] = {}
for label, name in MODELS.items():
    if "DPO" in label:
        scores = load(name)["scores"]
        summary["changed"][name] = {"won": sum(b > a for a, b in zip(start, scores)), "lost": sum(b < a for a, b in zip(start, scores))}

tok = pretrained.load_tokenizer()
print(f"\n{'':38s} {'solved':>8s} {'tokens':>7s} {'cut off':>8s} {'answered':>9s}")
for label, name in MODELS.items():
    r = load(name)
    with open(f"out/eval/{name}.jsonl", encoding="utf-8") as f:
        replies = [json.loads(line)["reply"] for line in f]
    lengths = [len(tok.encode_ordinary(reply)) for reply in replies]
    cut_off = sum(n >= r["max_new_tokens"] - 2 for n in lengths)
    answered = sum("####" in reply for reply in replies)  # replies that end in an answer of the asked-for form
    summary["models"][label] = {"name": name, "mean": r["mean"], "error": r["error"], "tokens": sum(lengths) / len(lengths), "cut_off": cut_off, "answered": answered}
    print(f"{label:38s} {r['mean']:8.1%} {sum(lengths) / len(lengths):7.0f} {cut_off:8d} {answered:9d}")

# Every run at the end of its training: how much likelier its pairs' replies
# had become, over its last five steps, and its last held-out score.
summary["runs"] = {}
for name in ("dpo-1", "dpo", "dpo-lr1e-5", "dpo-lr2e-6", "dpo-nll"):
    with open(f"out/{name.replace('dpo', 'dpo-gsm8k', 1)}/log.jsonl") as f:
        log = [json.loads(line) for line in f]
    last = [r for r in log if "gain_preferred" in r][-5:]
    summary["runs"][name] = {
        "steps": max(r["step"] for r in log),
        "gain_preferred": sum(r["gain_preferred"] for r in last) / len(last),
        "gain_other": sum(r["gain_other"] for r in last) / len(last),
        "held_out": [r["held_out"] for r in log if "held_out" in r][-1],
    }
    print(f"{name:12s} last five steps: right replies {summary['runs'][name]['gain_preferred']:+.1f} nats, wrong ones {summary['runs'][name]['gain_other']:+.1f}; held-out {summary['runs'][name]['held_out']:.1%}")

# One pair to show: the shortest, whose wrong reply ended, as most did.
with open("data/preferences/gsm8k.jsonl", encoding="utf-8") as f:
    pairs = [json.loads(line) for line in f]
pair = min((p for p in pairs if all(p["finished"])), key=lambda p: len(p["replies"][0]) + len(p["replies"][1]))
summary["example"] = {"problem": pair["messages"][-1]["content"], "preferred": pair["replies"][0], "other": pair["replies"][1]}

with open(result_path("runs/dpo/summary.json"), "w") as f:
    json.dump(summary, f, indent=1)
print(f"\nwrote {result_path('runs/dpo/summary.json')}")
