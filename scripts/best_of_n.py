"""
Rejection sampling: write several stories, keep the one the reward model
likes best. The simplest use of a reward model, with no training at all.

    uv run scripts/best_of_n.py

Writes runs/rlhf/best_of_n.json. See docs/08-rlhf.md.
"""

import argparse
import json
import os

import torch

from minllm import checkpoint, preferences, reward, rl, tasks
from minllm.common import pick_device, result_path

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="out/sft-words")
parser.add_argument("--reward", default="out/reward")
parser.add_argument("--prompts", type=int, default=300)
parser.add_argument("--n", type=int, default=16, help="the largest number of stories per prompt")
parser.add_argument("--out", default="runs/rlhf/best_of_n.json")
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()
args.out = result_path(args.out)  # a path under runs/ goes to the results folder

device = pick_device()
torch.manual_seed(args.seed)
model, tok, _ = checkpoint.load(args.model, device)
reward_model, _ = reward.load(args.reward, device)
examples = tasks.test_examples("taste", args.prompts)
conversations = [e["messages"] for e in examples]

texts = []
for start in range(0, len(conversations), 32):
    texts += rl.rollout(model, tok, conversations[start : start + 32], samples=args.n, max_new_tokens=220)[2]
repeated = [c for c in conversations for _ in range(args.n)]
rewards = torch.tensor(reward.score(reward_model, tok, repeated, texts)).view(-1, args.n)
tastes = torch.tensor([preferences.taste(t) for t in texts]).view(-1, args.n)

results = {"prompts": len(conversations), "n": [], "picked_by_reward_model": [], "picked_by_taste": [], "any": tastes.mean().item()}
results["tastes_of_first"] = tastes[:, 0].tolist()  # per prompt, for confidence intervals
print(f"mean taste of a story: {results['any']:+.3f}")
print(f"{'n':>3} {'reward model picks':>19} {'the annotator would pick':>25}")
n = 1
while n <= args.n:
    picked = tastes[:, :n].gather(1, rewards[:, :n].argmax(dim=1, keepdim=True)).mean().item()
    best = tastes[:, :n].max(dim=1).values.mean().item()
    results["n"].append(n)
    results["picked_by_reward_model"].append(picked)
    results["picked_by_taste"].append(best)
    if n == 8:
        results["tastes_of_best_of_8"] = tastes[:, :n].gather(1, rewards[:, :n].argmax(dim=1, keepdim=True)).view(-1).tolist()
    print(f"{n:>3} {picked:>+19.3f} {best:>+25.3f}")
    n *= 2

os.makedirs(os.path.dirname(args.out), exist_ok=True)
with open(args.out, "w") as f:
    json.dump(results, f, indent=1)
print(f"wrote {args.out}")
