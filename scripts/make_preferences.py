"""
Collect preference data: two stories per prompt, and which one the annotator
prefers. The annotator here is the simulated one of minllm/preferences.py.

    uv run scripts/make_preferences.py

Writes data/preferences/train.jsonl and held_out.jsonl. To give your own
preferences instead, see scripts/label_preferences.py.
"""

import argparse
import json
import os
import random

import torch

from minllm import checkpoint, preferences, rl, tasks
from minllm.common import pick_device, result_path

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="out/sft-words", help="the model whose stories are compared")
parser.add_argument("--n", type=int, default=4000, help="number of comparisons")
parser.add_argument("--held-out", type=int, default=500, help="how many of them to keep out of training")
parser.add_argument("--out", default="data/preferences")
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

device = pick_device()
torch.manual_seed(args.seed)
rng = random.Random(args.seed)
model, tok, _ = checkpoint.load(args.model, device)

# Prompts from training stories; the test prompts of the later evaluations stay unseen.
examples = list(tasks.words_examples("data/tinystories/train.txt", tasks.content_words(), args.n, seed=args.seed + 100))
pairs = []
for start in range(0, len(examples), 256):
    chunk = examples[start : start + 256]
    # Two samples per prompt, at temperature 1: the model's own distribution.
    _, _, texts = rl.rollout(model, tok, [e["messages"] for e in chunk], samples=2, max_new_tokens=220)
    for i, e in enumerate(chunk):
        a, b = texts[2 * i], texts[2 * i + 1]
        pairs.append({"messages": e["messages"], "replies": [a, b], "preferred": preferences.prefer(a, b, rng)})
    print(f"\r{len(pairs)}/{len(examples)}", end="", flush=True)

os.makedirs(args.out, exist_ok=True)
preferences.save(f"{args.out}/train.jsonl", pairs[: -args.held_out])
preferences.save(f"{args.out}/held_out.jsonl", pairs[-args.held_out :])

# How good is the annotator's signal? Its taste is known, so this can be said.
tastes = [[preferences.taste(r) for r in p["replies"]] for p in pairs]
consistent = sum((t[p["preferred"]] >= t[1 - p["preferred"]]) for t, p in zip(tastes, pairs)) / len(pairs)
mean = sum(sum(t) for t in tastes) / (2 * len(tastes))
print(f"\r{len(pairs)} comparisons written to {args.out}")
print(f"mean taste of a story: {mean:+.3f}; the annotator picked the story of higher taste in {consistent:.1%} of comparisons")
os.makedirs(result_path("runs/rlhf"), exist_ok=True)
with open(result_path("runs/rlhf/preferences.json"), "w") as f:
    json.dump({"model": args.model, "comparisons": len(pairs), "held_out": args.held_out, "mean_taste": mean, "annotator_consistency": consistent}, f, indent=1)
