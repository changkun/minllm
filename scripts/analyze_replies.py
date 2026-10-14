"""
How the GSM8K replies of stages 7 and 9 are written: how long a typical
reply is, in characters, and how many give their answer the way GSM8K's
reference solutions do, in a line "#### <number>". The published model, the model
fine-tuned with LoRA in stage 6, and the same after stage 9's GRPO.

    uv run scripts/analyze_replies.py   # reads out/eval/, writes runs/eval/replies-summary.json

See docs/07-evaluation.md and docs/09-rlvr.md.
"""

import json
import statistics

MODELS = {"0.5B as published": "gsm8k-qwen", "fine-tuned, LoRA": "gsm8k-sft-gsm8k-lora", "after GRPO": "gsm8k-grpo-gsm8k"}

summary = {}
for label, name in MODELS.items():
    with open(f"out/eval/{name}.jsonl", encoding="utf-8") as f:
        replies = [json.loads(line)["reply"] for line in f]
    summary[name] = {
        "label": label,
        "n": len(replies),
        "median_characters": statistics.median(len(r) for r in replies),
        "with_answer_line": sum("####" in r for r in replies) / len(replies),
    }
    print(name, summary[name])
with open("runs/eval/replies-summary.json", "w") as f:
    json.dump(summary, f, indent=1)
print("wrote runs/eval/replies-summary.json")
