"""
Direct preference optimization (Rafailov et al., 2023, arXiv:2305.18290):
train the policy on the comparisons themselves. No reward model, and no
sampling during training.

    uv run scripts/dpo.py                               # stage 8: stories, by the annotator's comparisons
    uv run scripts/dpo.py --model out/sft-gsm8k-lora --train data/preferences/gsm8k.jsonl --out out/dpo-gsm8k

The second trains the 0.5B model on pairs of a right and a wrong reply,
made by scripts/make_preferences.py --task gsm8k. See docs/08-rlhf.md and
docs/08b-dpo.md.
"""

import argparse
import json
import os
import random
import time

import torch

from minllm import chat, checkpoint, dpo, evaluate, lora, preferences, rl, tasks
from minllm.common import free_memory, pick_device
from minllm.pretrained import SYSTEM_PROMPT

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="out/sft-words", help="the policy's starting point, and the reference")
parser.add_argument("--train", default="data/preferences/train.jsonl")
parser.add_argument("--out", default="out/dpo")
parser.add_argument("--epochs", type=int, default=1)
parser.add_argument("--batch-size", type=int, default=32, help="pairs per step")
parser.add_argument("--chunk", type=int, default=None, help="pairs per forward pass, to fit in memory; the whole batch by default")
parser.add_argument("--lr", type=float, default=5e-5)
parser.add_argument("--beta", type=float, default=0.1, help="how strongly to hold the policy to the reference: larger is tighter")
parser.add_argument("--nll", type=float, default=0.0, help="weight of the preferred reply's own loss per token, added to DPO's (Pang et al., 2024)")
parser.add_argument("--eval-every", type=int, default=0, help="for GSM8K pairs: score 200 held-out training problems every this many steps")
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

device = pick_device()
device_type = torch.device(device).type
torch.manual_seed(args.seed)
rng = random.Random(args.seed)
policy, tok, meta = checkpoint.load(args.model, device)
policy.train()
small = meta["family"] == "small"
trainable = [p for p in policy.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(trainable, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)
autocast = torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=device_type != "cpu" and not small)
chunk = args.chunk or args.batch_size

# Every pair as token ids. A reply that ran into the token limit, which a
# wrong one may have, did not end, and gets no <|im_end|>.
pairs = preferences.load(args.train)
end = tok.special_tokens[chat.IM_END]
limit = policy.cfg.max_seq_len
prompts = [chat.encode(tok, p["messages"]) for p in pairs]
encode = lambda p, k, prompt: (tok.encode_ordinary(p["replies"][k]) + [end] * p.get("finished", [True, True])[k])[: limit - len(prompt)]
preferred = [encode(p, p["preferred"], prompt) for p, prompt in zip(pairs, prompts)]
other = [encode(p, 1 - p["preferred"], prompt) for p, prompt in zip(pairs, prompts)]


def hidden(x):
    with autocast:
        return policy.hidden(x)


def reply_logprob(rows):
    """log p(reply | prompt) of the preferred and then the other reply of every pair in rows, and the preferred replies' lengths."""
    x, y, at = rl.batch([prompts[i] for i in rows] * 2, [preferred[i] for i in rows] + [other[i] for i in rows], device)
    return rl.per_reply(rl.token_logprobs(rl.target_logits(policy, x, y, hidden), y), at, 2 * len(rows))


# The reference is the starting model, which is the policy before its first
# step: one pass over the pairs gives every log-probability it will be asked for.
with torch.no_grad():
    policy.eval()
    reference = torch.cat([reply_logprob(range(i, min(i + chunk, len(pairs)))).view(2, -1) for i in range(0, len(pairs), chunk)], dim=1)
    policy.train()
    free_memory(device)

held_out = None
if args.eval_every:
    held_out = tasks.gsm8k_examples("train", SYSTEM_PROMPT)[-200:]  # the problems no pair was drawn from


def held_out_score():
    """Greedy replies to the held-out problems: their score, length in tokens, and how many ran into the limit."""
    policy.eval()
    replies = evaluate.replies(policy, tok, held_out, tasks.MAX_NEW_TOKENS["gsm8k"], batch_size=64)
    policy.train()
    free_memory(device)
    lengths = [len(tok.encode_ordinary(r)) for r in replies]
    score = sum(tasks.SCORE["gsm8k"](e, r) for e, r in zip(held_out, replies)) / len(held_out)
    return {"held_out": score, "held_out_tokens": sum(lengths) / len(lengths), "held_out_cut_off": sum(n >= tasks.MAX_NEW_TOKENS["gsm8k"] - 2 for n in lengths)}


os.makedirs(args.out, exist_ok=True)
log = open(f"{args.out}/log.jsonl", "w")
print(f"device {device}, {len(pairs)} comparisons, {args.epochs} epochs, beta {args.beta}" + (f", nll {args.nll}" if args.nll else ""))
start, step = time.time(), 0
if held_out:
    record = {"step": 0} | held_out_score()
    log.write(json.dumps(record | {"elapsed": time.time() - start}) + "\n")
    print(f"step    0 | held-out {record['held_out']:.3f}, {record['held_out_tokens']:.0f} tokens")
order = list(range(len(pairs)))
for epoch in range(args.epochs):
    rng.shuffle(order)
    for i in range(0, len(order) - args.batch_size + 1, args.batch_size):
        batch = order[i : i + args.batch_size]
        optimizer.zero_grad(set_to_none=True)
        stats = {"loss": 0.0, "accuracy": 0.0, "gain_preferred": 0.0, "gain_other": 0.0}
        for j in range(0, len(batch), chunk):
            rows = batch[j : j + chunk]
            logp = reply_logprob(rows).view(2, -1)
            # How much likelier has each reply become than it was under the
            # starting model? DPO asks that this grow more for the preferred
            # reply than for the other one.
            gain = logp - reference[:, rows]
            loss = dpo.loss(gain[0], gain[1], args.beta)
            if args.nll:
                lengths = torch.tensor([len(preferred[k]) for k in rows], device=device)
                loss = loss + args.nll * (-logp[0] / lengths).mean()
            share = len(rows) / len(batch)
            (loss * share).backward()
            stats["loss"] += loss.item() * share
            stats["accuracy"] += (gain[0] > gain[1]).float().mean().item() * share
            stats["gain_preferred"] += gain[0].mean().item() * share
            stats["gain_other"] += gain[1].mean().item() * share
            if not small:
                free_memory(device)
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        step += 1
        record = {"step": step} | stats
        if held_out and step % args.eval_every == 0:
            record |= held_out_score()
        log.write(json.dumps(record | {"elapsed": time.time() - start}) + "\n")
        log.flush()
        if step % 4 == 0:
            free_memory(device)
        if step % 20 == 0 or step == 1 or "held_out" in record:
            line = f"step {step:4d} | loss {stats['loss']:.4f} | preferred reply gained more in {stats['accuracy']:.0%} of pairs | gains {stats['gain_preferred']:+.1f} and {stats['gain_other']:+.1f}"
            print(line + (f" | held-out {record['held_out']:.3f}, {record['held_out_tokens']:.0f} tokens" if "held_out" in record else ""))

if held_out and step % args.eval_every:
    record = {"step": step} | held_out_score()
    log.write(json.dumps(record | {"elapsed": time.time() - start}) + "\n")
    print(f"step {step:4d} | held-out {record['held_out']:.3f}, {record['held_out_tokens']:.0f} tokens")
minutes = (time.time() - start) / 60
meta |= {"method": "dpo", "step": step, "val_loss": 0.0, "minutes": minutes, "args": vars(args)}
if not lora.modules(policy):
    meta |= {"base": args.model}
checkpoint.save(args.out, policy, meta)
print(f"done in {minutes:.1f} minutes, saved to {args.out}")
