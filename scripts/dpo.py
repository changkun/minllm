"""
Direct preference optimization (Rafailov et al., 2023, arXiv:2305.18290):
train the policy on the comparisons themselves. No reward model, and no
sampling during training.

    uv run scripts/dpo.py

See docs/08-rlhf.md.
"""

import argparse
import copy
import json
import os
import random
import time

import torch
import torch.nn.functional as F

from minllm import chat, checkpoint, preferences, rl
from minllm.common import free_memory, pick_device

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="out/sft-words", help="the policy's starting point, and the reference")
parser.add_argument("--train", default="data/preferences/train.jsonl")
parser.add_argument("--out", default="out/dpo")
parser.add_argument("--epochs", type=int, default=1)
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--lr", type=float, default=5e-5)
parser.add_argument("--beta", type=float, default=0.1, help="how strongly to hold the policy to the reference: larger is tighter")
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

device = pick_device()
torch.manual_seed(args.seed)
rng = random.Random(args.seed)
policy, tok, _ = checkpoint.load(args.model, device)
policy.train()
start_policy = copy.deepcopy(policy).eval().requires_grad_(False)
optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)
pairs = preferences.load(args.train)
end = tok.special_tokens[chat.IM_END]
limit = policy.cfg.max_seq_len


def reply_logprob(model, prompts, replies):
    """log p(reply | prompt) for every pair: the sum of its tokens' log-probabilities."""
    x, y, rows = rl.batch(prompts, replies, device)
    return rl.per_reply(rl.token_logprobs(rl.target_logits(model, x, y), y), rows, len(prompts))


os.makedirs(args.out, exist_ok=True)
log = open(f"{args.out}/log.jsonl", "w")
print(f"device {device}, {len(pairs)} comparisons, {args.epochs} epochs, beta {args.beta}")
start, step = time.time(), 0
for epoch in range(args.epochs):
    rng.shuffle(pairs)
    for i in range(0, len(pairs) - args.batch_size + 1, args.batch_size):
        chunk = pairs[i : i + args.batch_size]
        prompts = [chat.encode(tok, p["messages"]) for p in chunk] * 2
        texts = [p["replies"][p["preferred"]] for p in chunk] + [p["replies"][1 - p["preferred"]] for p in chunk]
        replies = [(tok.encode_ordinary(t) + [end])[: limit - len(p)] for p, t in zip(prompts, texts)]

        # How much likelier has each reply become than it was under the
        # starting model? DPO asks that this grow more for the preferred
        # reply than for the other one.
        with torch.no_grad():
            before = reply_logprob(start_policy, prompts, replies)
        gain = reply_logprob(policy, prompts, replies) - before
        preferred, other = gain[: len(chunk)], gain[len(chunk) :]
        loss = -F.logsigmoid(args.beta * (preferred - other)).mean()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()
        step += 1
        accuracy = (preferred > other).float().mean().item()
        log.write(json.dumps({"step": step, "loss": loss.item(), "accuracy": accuracy, "elapsed": time.time() - start}) + "\n")
        if step % 4 == 0:
            free_memory(device)
        if step % 20 == 0 or step == 1:
            print(f"step {step:4d} | loss {loss.item():.4f} | preferred reply gained more in {accuracy:.0%} of pairs")

minutes = (time.time() - start) / 60
meta = {"family": "small", "base": args.model, "method": "dpo", "step": step, "val_loss": 0.0, "minutes": minutes, "args": vars(args)}
checkpoint.save(args.out, policy, meta)
print(f"done in {minutes:.1f} minutes, saved to {args.out}")
