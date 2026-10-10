"""
RLHF by policy gradient: the model writes stories, the reward model scores
them, and PPO makes the well-scored ones more likely, while a KL penalty
holds the model near where it started.

    uv run scripts/ppo.py
    uv run scripts/ppo.py --kl 0 --out out/ppo-no-kl     # without the penalty

See docs/08-rlhf.md.
"""

import argparse
import copy
import json
import os
import random
import time

import torch

from minllm import checkpoint, preferences, reward, rl, tasks
from minllm.common import free_memory, pick_device

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="out/sft-words", help="the policy's starting point, and the reference it is held near")
parser.add_argument("--reward", default="out/reward")
parser.add_argument("--out", default="out/ppo")
parser.add_argument("--iterations", type=int, default=60)
parser.add_argument("--prompts", type=int, default=128, help="stories sampled per iteration")
parser.add_argument("--epochs", type=int, default=3, help="passes over each iteration's stories")
parser.add_argument("--minibatch", type=int, default=64)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--clip", type=float, default=0.2)
parser.add_argument("--kl", type=float, default=0.1, help="weight of the KL penalty towards the starting model")
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

device = pick_device()
torch.manual_seed(args.seed)
rng = random.Random(args.seed)
policy, tok, meta = checkpoint.load(args.model, device)
policy.train()
start_policy = copy.deepcopy(policy).eval().requires_grad_(False)  # the reference
reference = lambda x, y: rl.target_logits(start_policy, x, y)
reward_model, _ = reward.load(args.reward, device)
optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)

n = args.iterations * args.prompts
stream = tasks.words_examples("data/tinystories/train.txt", tasks.content_words(), n, seed=args.seed + 200)
os.makedirs(args.out, exist_ok=True)
log = open(f"{args.out}/log.jsonl", "w")
print(f"device {device}, {args.iterations} iterations of {args.prompts} stories, kl {args.kl}")
start = time.time()
for iteration in range(1, args.iterations + 1):
    examples = [next(stream) for _ in range(args.prompts)]
    conversations = [e["messages"] for e in examples]

    # 1. Act: the current policy writes one story per prompt.
    prompts, replies, texts = rl.rollout(policy, tok, conversations, max_new_tokens=220)
    # 2. Reward: the learned reward model scores them. The annotator's real
    #    taste is recorded too, but only to watch: training never sees it.
    rewards = torch.tensor(reward.score(reward_model, tok, conversations, texts), device=device)
    taste = sum(preferences.taste(t) for t in texts) / len(texts)
    words = sum(tasks.words_score(e, t) for e, t in zip(examples, texts)) / len(texts)
    # 3. Advantage: how much better than the batch's average each story did.
    advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-6)
    # 4. Improve: a few passes of PPO over these stories.
    old = rl.reply_logprobs(policy, prompts, replies, device)
    order = list(range(args.prompts))
    for _ in range(args.epochs):
        rng.shuffle(order)
        for i in range(0, args.prompts, args.minibatch):
            pick = order[i : i + args.minibatch]
            stats = rl.update(
                policy,
                optimizer,
                [prompts[j] for j in pick],
                [replies[j] for j in pick],
                advantages[pick],
                device,
                old_logp=[old[j] for j in pick],
                clip=args.clip,
                kl_coef=args.kl,
                reference=reference,
            )
    free_memory(device)

    length = sum(len(r) for r in replies) / len(replies)
    record = {"iteration": iteration, "reward": rewards.mean().item(), "taste": taste, "words": words, "kl": stats["kl"], "tokens": length}
    log.write(json.dumps(record | {"elapsed": time.time() - start}) + "\n")
    log.flush()
    if iteration % 5 == 0 or iteration == 1:
        print(f"iteration {iteration:3d} | reward {record['reward']:+.3f} | taste {taste:+.3f} | words {words:.2f} | kl {stats['kl']:.4f} | {length:.0f} tokens")

minutes = (time.time() - start) / 60
meta = {"family": "small", "base": args.model, "method": "ppo", "step": args.iterations, "val_loss": 0.0, "minutes": minutes, "args": vars(args)}
checkpoint.save(args.out, policy, meta)
print(f"done in {minutes:.1f} minutes, saved to {args.out}")
