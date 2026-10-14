"""
Count the calculations that replies write out, and how many of them are
wrong: "16 - 3 - 4 = 7" is a calculation, and a wrong one.

    uv run scripts/count_calculations.py      # the GSM8K replies of stage 7, from out/eval/; writes runs/eval/calculations.json

A model without a tool does its arithmetic in its head, and this measures
how often that goes wrong. scripts/evaluate_tools.py counts the same for
replies written with a calculator. See docs/06a-tool-calling.md.
"""

import json

from minllm import tools
from minllm.common import result_path

result = {}
for name in ["gsm8k-qwen", "gsm8k-sft-gsm8k-lora", "gsm8k-sft-gsm8k"]:
    with open(f"out/eval/{name}.jsonl", encoding="utf-8") as f:
        replies = [json.loads(line)["reply"] for line in f]
    found = [tools.written_calculations(r) for r in replies]
    calculations = sum(map(len, found))
    wrong = sum(not right for f in found for *_, right in f)
    result[name] = {"replies": len(replies), "calculations": calculations, "wrong": wrong, "replies_with_a_wrong_one": sum(any(not right for *_, right in f) for f in found), "examples": [[e, w] for f in found for e, w, right in f if not right][:12]}
    print(f"{name}: {calculations} calculations written out, {wrong} wrong ({wrong / calculations:.1%}), in {result[name]['replies_with_a_wrong_one']} of {len(replies)} replies")
with open(result_path("runs/eval/calculations.json"), "w") as f:
    json.dump(result, f, indent=1)
print("wrote", result_path("runs/eval/calculations.json"))
