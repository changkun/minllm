"""
Train a reward model on preference comparisons.

    uv run scripts/train_reward.py
    uv run scripts/train_reward.py --pairs 500 --out out/reward-500     # with fewer comparisons

See docs/08-rlhf.md.
"""

import argparse
import json
import os
import random
import time

import torch

from minllm import checkpoint, preferences, reward
from minllm.common import free_memory, pick_device

parser = argparse.ArgumentParser()
parser.add_argument("--base", default="out/sft-words", help="the language model whose weights the reward model starts from")
parser.add_argument("--train", default="data/preferences/train.jsonl")
parser.add_argument("--held-out", default="data/preferences/held_out.jsonl")
parser.add_argument("--pairs", type=int, default=None, help="use only this many training comparisons")
parser.add_argument("--out", default="out/reward")
parser.add_argument("--epochs", type=int, default=3)
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

device = pick_device()
torch.manual_seed(args.seed)
rng = random.Random(args.seed)
body, tok, _ = checkpoint.load(args.base, device)
model = reward.RewardModel(body).to(device).train()
train = preferences.load(args.train)[: args.pairs]
held_out = preferences.load(args.held_out)
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
max_len = body.cfg.max_seq_len


def rewards_of(pairs):
    """The rewards of the preferred and of the other reply of every pair."""
    conversations = [p["messages"] for p in pairs] * 2
    replies = [p["replies"][p["preferred"]] for p in pairs] + [p["replies"][1 - p["preferred"]] for p in pairs]
    x, lengths = reward.encode(tok, conversations, replies, device, max_len)
    r = model(x, lengths)
    return r[: len(pairs)], r[len(pairs) :]


@torch.no_grad()
def evaluate(pairs):
    """
    Three numbers on comparisons the model was not trained on: its loss, how
    often it agrees with the annotator's choice, and how often it agrees with
    the annotator's taste, which is the choice with the noise removed.
    """
    loss = agree = truth = known = 0
    for i in range(0, len(pairs), 64):
        chunk = pairs[i : i + 64]
        preferred, other = rewards_of(chunk)
        loss += reward.comparison_loss(preferred, other).item() * len(chunk)
        agree += (preferred > other).sum().item()
        for p, a, b in zip(chunk, preferred.tolist(), other.tolist()):
            ta, tb = preferences.taste(p["replies"][p["preferred"]]), preferences.taste(p["replies"][1 - p["preferred"]])
            if ta != tb:
                known += 1
                truth += (a > b) == (ta > tb)
    return loss / len(pairs), agree / len(pairs), truth / known


start = time.time()
steps = 0
loss, agree, truth = evaluate(held_out)
print(f"device {device}, {len(train)} comparisons, {args.epochs} epochs")
print(f"before: loss {loss:.4f}, agrees with the annotator {agree:.1%}")
for epoch in range(args.epochs):
    rng.shuffle(train)
    for i in range(0, len(train), args.batch_size):
        preferred, other = rewards_of(train[i : i + args.batch_size])
        batch_loss = reward.comparison_loss(preferred, other)
        optimizer.zero_grad(set_to_none=True)
        batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        steps += 1
        if steps % 8 == 0:
            free_memory(device)
    loss, agree, truth = evaluate(held_out)
    print(f"epoch {epoch + 1}: held-out loss {loss:.4f}, agrees with the annotator {agree:.1%}, with their taste {truth:.1%}")

os.makedirs(args.out, exist_ok=True)
torch.save({"model": model.state_dict()}, f"{args.out}/reward.pt")
meta = {"base": args.base, "pairs": len(train), "held_out_loss": loss, "agrees_with_annotator": agree, "agrees_with_taste": truth}

# Reward against taste for every held-out story: the picture of how well the hidden taste was recovered.
model.eval()
stories = [(p["messages"], r) for p in held_out for r in p["replies"]]
meta["held_out_rewards"] = [round(r, 4) for r in reward.score(model, tok, [m for m, _ in stories], [r for _, r in stories])]
meta["held_out_tastes"] = [round(preferences.taste(r), 4) for _, r in stories]
with open(f"{args.out}/meta.json", "w") as f:
    json.dump(meta | {"minutes": (time.time() - start) / 60, "args": vars(args)}, f)
print(f"saved to {args.out}")
