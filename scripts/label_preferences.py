"""
Give your own preferences: read two stories, say which one you like better.

    uv run scripts/label_preferences.py --n 50

Appends to data/preferences/human.jsonl, in the format that
scripts/train_reward.py reads (--train data/preferences/human.jsonl).
A reward model needs a few hundred comparisons to learn anything, and the
stories are short: about a minute for ten.
"""

import argparse
import json
import os
import random
import textwrap

import torch

from minllm import checkpoint, rl, tasks
from minllm.common import pick_device

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="out/sft-words")
parser.add_argument("--n", type=int, default=50)
parser.add_argument("--out", default="data/preferences/human.jsonl")
parser.add_argument("--seed", type=int, default=None)
args = parser.parse_args()

device = pick_device()
seed = args.seed if args.seed is not None else random.randrange(1 << 30)
torch.manual_seed(seed)
model, tok, _ = checkpoint.load(args.model, device)
examples = list(tasks.words_examples("data/tinystories/train.txt", tasks.content_words(), args.n, seed=seed))
_, _, texts = rl.rollout(model, tok, [e["messages"] for e in examples], samples=2, max_new_tokens=220)

os.makedirs(os.path.dirname(args.out), exist_ok=True)
print("Type 1 or 2 for the story you prefer, s to skip, q to stop.\n")
with open(args.out, "a", encoding="utf-8") as f:
    for i, e in enumerate(examples):
        replies = [texts[2 * i], texts[2 * i + 1]]
        print(f"--- {i + 1}/{len(examples)}: {e['messages'][-1]['content']}\n")
        for number, story in enumerate(replies, 1):
            print(f"[{number}]\n{textwrap.fill(story, 100)}\n")
        answer = ""
        while answer not in ("1", "2", "s", "q"):
            try:
                answer = input("> ").strip().lower()
            except EOFError:
                answer = "q"
        if answer == "q":
            break
        if answer in ("1", "2"):
            f.write(json.dumps({"messages": e["messages"], "replies": replies, "preferred": int(answer) - 1}, ensure_ascii=False) + "\n")
            f.flush()
print(f"saved to {args.out}")
