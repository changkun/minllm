"""
Compare the runs of docs/06a-tool-calling.md problem by problem: the same
model with its calls run and with them failing, the model fine-tuned with
the tool against the one fine-tuned on the same problems without it, and
the published model with and without the tool in its prompt.

    uv run scripts/analyze_tools.py      # reads runs/eval/ and out/eval/, writes runs/eval/tools-summary.json

It also counts, from all conversations that scripts/evaluate_tools.py
saved in out/eval/, how often the model's turn after a result begins with
that result: whether it reads what the tool says.

See docs/06a-tool-calling.md.
"""

import json

from minllm import evaluate


def load(name):
    with open(f"runs/eval/{name}.json") as f:
        return json.load(f)


pairs = {
    "with the tool, against without it (stage 6)": ("gsm8k-sft-gsm8k-lora", "gsm8k-tools-sft-gsm8k-tools-lora"),
    "calls failing, against calls run": ("gsm8k-tools-sft-gsm8k-tools-lora", "gsm8k-tools-sft-gsm8k-tools-lora-no-execute"),
    "published, told about the tool, against not": ("gsm8k-qwen", "gsm8k-tools-qwen"),
}
summary = {"comparisons": {}, "runs": {}}
for label, (a, b) in pairs.items():
    difference, low, high = evaluate.paired_difference(load(a)["scores"], load(b)["scores"])
    summary["comparisons"][label] = {"from": a, "to": b, "difference": difference, "low": low, "high": high}
    print(f"{label}: {difference:+.1%}, 95% interval {low:+.1%} to {high:+.1%}")
for name in ["gsm8k-tools-sft-gsm8k-tools-lora", "gsm8k-tools-sft-gsm8k-tools-lora-no-execute", "gsm8k-tools-qwen"]:
    r = load(name)
    n = r["n"]
    written = sum(w[1] for w in r["written"])
    summary["runs"][name] = {
        "mean": r["mean"], "error": r["error"], "n": n,
        "with_a_call": sum(c > 0 for c in r["calls"]) / n, "calls_per_problem": sum(r["calls"]) / n, "malformed": sum(r["bad_calls"]),
        "written_calculations": written, "wrong_calculations": sum(w[1] - w[0] for w in r["written"]), "seconds": r["seconds"],
    }
    print(name, {k: round(v, 3) if isinstance(v, float) else v for k, v in summary["runs"][name].items()})
# Does the model go on from the tool's result?
for name in ["gsm8k-tools-sft-gsm8k-tools-lora", "gsm8k-tools-sft-gsm8k-tools-lora-no-execute"]:
    follows = results = 0
    with open(f"out/eval/{name}.jsonl", encoding="utf-8") as f:
        for line in f:
            conversation = json.loads(line)["conversation"]
            for k, m in enumerate(conversation[:-1]):
                if m["role"] == "user" and m["content"].startswith("<tool_response>"):
                    results += 1
                    follows += conversation[k + 1]["content"].strip().startswith(m["content"].split("\n")[1])
    summary["runs"][name] |= {"results": results, "next_turn_begins_with_it": follows}
    print(f"{name}: the next turn begins with the result after {follows} of {results}")
with open("runs/eval/tools-summary.json", "w") as f:
    json.dump(summary, f, indent=1)
print("wrote runs/eval/tools-summary.json")
