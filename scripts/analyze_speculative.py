"""
Summarize the measurements of docs/04b-speculative.md: for every pair of a
model and its draft, the speed of each case against the model alone, the
share of draft tokens kept, the tokens a pass of the model yields, and how
many greedy replies are the model's own, token for token, with the token
at which each of the others parts from it. What a round of draft and check
costs is counted in passes of the model alone: the tokens it yields over
how much faster it is.

    uv run scripts/analyze_speculative.py   # reads runs/bench/speculative-*.json, writes runs/bench/speculative-summary.json

Files written before speculative.generate left the stop token out of what
it returns count a reply that ends in one as different from the model's
own. Here the stop token is left out of every comparison.

See docs/04b-speculative.md.
"""

import json
import os

from minllm import chat, checkpoint, pretrained
from minllm.data import EOT
from minllm.tokenizer import Tokenizer

PAIRS = {
    "speculative-wide-mps": "the 42M story model, drafted by the 14M one",
    "speculative-qwen2.5-1.5b-instruct-mps": "Qwen2.5-1.5B, drafted by the 0.5B model",
    "speculative-qwen2.5-7b-instruct-mps": "Qwen2.5-7B, drafted by the 0.5B model",
    "speculative-qwen2.5-7b-instruct-draft-1.5b-mps": "Qwen2.5-7B, drafted by the 1.5B model",
    "speculative-wide-mps-float32": "the story models in 32-bit floats",
    "speculative-wide-cpu-bfloat16": "the story models on the CPU, in 16-bit floats",
    "speculative-qwen2.5-1.5b-instruct-mps-float32": "Qwen2.5-1.5B and the 0.5B draft in 32-bit floats",
}
STOPS = {"qwen": pretrained.load_tokenizer().special_tokens[chat.IM_END], "small": Tokenizer.load(checkpoint.TOKENIZER).special_tokens[EOT]}


def parted(alone, new, stop):
    """The first token at which a reply differs from the model's own, or None if it is the same; a stop token at the end is not counted."""
    new = new[:-1] if new and new[-1] == stop else new
    if new == alone:
        return None
    return next((i for i, (a, b) in enumerate(zip(alone, new)) if a != b), min(len(alone), len(new)))


summary = {}
for name, label in PAIRS.items():
    if not os.path.exists(f"runs/bench/{name}.json"):
        continue
    with open(f"runs/bench/{name}.json") as f:
        r = json.load(f)
    stop = STOPS["qwen" if os.path.exists(f"{r['target']}/config.json") else "small"]
    alone = next(c for c in r["cases"] if c["draft"] == "none" and c["temperature"] == 0)
    sampled = next(c for c in r["cases"] if c["draft"] == "none" and c["temperature"] > 0)
    cases = []
    for c in r["cases"]:
        if c["draft"] == "none":
            continue
        base = alone if c["temperature"] == 0 else sampled  # sampling is measured against the model sampling alone
        case = {
            "draft": c["draft"], "k": c["k"], "temperature": c["temperature"],
            "tokens_per_second": c["tokens_per_second"], "speedup": c["tokens_per_second"] / base["tokens_per_second"],
            "kept": c["kept"] / max(1, c["proposed"]), "tokens_per_pass": c["tokens"] / max(1, c["passes"]),
        }
        # What a round costs, counted in tokens of the model alone: the tokens it yields over how much faster it is.
        case["round_cost"] = case["tokens_per_pass"] / case["speedup"]
        if c["temperature"] == 0:
            parts = [parted(a, b, stop) for a, b in zip(alone["outputs"], c["outputs"])]
            case |= {"same": sum(p is None for p in parts), "parted_at": parts}
        cases.append(case)
    summary[name] = {"label": label, "target": r["target"], "draft": r["draft"], "device": r["device"], "n": r["n"], "max_new_tokens": r["max_new_tokens"],
                     "alone": alone["tokens_per_second"], "alone_sampling": sampled["tokens_per_second"], "cases": cases}
    best = max((c for c in cases if c["temperature"] == 0), key=lambda c: c["speedup"])
    same = [c["same"] for c in cases if "same" in c]
    print(f"{label}: alone {alone['tokens_per_second']:.1f} tokens/s; best {best['draft']} k={best['k']} at {best['speedup']:.2f}x; greedy replies the same: {min(same)} to {max(same)} of {r['n']}")

with open("runs/bench/speculative-summary.json", "w") as f:
    json.dump(summary, f, indent=1)
print("wrote runs/bench/speculative-summary.json")
