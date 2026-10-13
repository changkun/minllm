"""
RLVR with GRPO (Shao et al., 2024, arXiv:2402.03300): the model answers each
prompt several times, a program checks every answer, and the answers that did
better than the others to the same prompt are made more likely.

    uv run scripts/grpo.py --task words     # the 14M model learns to use the requested words
    uv run scripts/grpo.py --task gsm8k     # the 0.5B model gets better at math problems
    uv run scripts/grpo.py --task gsm8k-tools   # the same, with a calculator it calls during the rollouts

See docs/09-rlvr.md, and docs/09b-tool-rl.md for the calculator.
"""

import argparse
import json
import os
import random
import time

import torch

from minllm import chat, checkpoint, evaluate, lora, rl, tasks, tools
from minllm.common import free_memory, pick_device
from minllm.pretrained import SYSTEM_PROMPT

parser = argparse.ArgumentParser()
parser.add_argument("--task", choices=["words", "gsm8k", "gsm8k-tools"], required=True)
parser.add_argument("--model", default=None, help="the policy's starting point")
parser.add_argument("--out", default=None)
parser.add_argument("--iterations", type=int, default=None)
parser.add_argument("--prompts", type=int, default=None, help="prompts per iteration")
parser.add_argument("--samples", type=int, default=8, help="replies per prompt: the group that is compared")
parser.add_argument("--lr", type=float, default=None)
parser.add_argument("--kl", type=float, default=0.0, help="weight of a KL penalty towards the starting model")
parser.add_argument("--lora", type=int, default=0, help="add fresh LoRA adapters of this rank instead of training every weight")
parser.add_argument("--must-finish", action="store_true", help="no reward for a reply that ran into the token limit instead of ending")
parser.add_argument("--max-new-tokens", type=int, default=None, help="the limit on a sampled reply's length; the task's own by default")
parser.add_argument("--eval-every", type=int, default=None)
parser.add_argument("--eval-n", type=int, default=None, help="held-out examples per evaluation")
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

small, with_tools = args.task == "words", args.task == "gsm8k-tools"
args.model = args.model or {"words": "out/sft-words", "gsm8k": "out/sft-gsm8k-lora", "gsm8k-tools": "out/sft-gsm8k-tools-lora"}[args.task]
args.out = args.out or f"out/grpo-{args.task}"
args.iterations = args.iterations or (150 if small else 60)
args.prompts = args.prompts or (32 if small else 8)
args.lr = args.lr or 5e-5
args.eval_every = args.eval_every or (25 if small else 20)
args.eval_n = args.eval_n or (500 if small else 200)

device = pick_device()
device_type = torch.device(device).type
torch.manual_seed(args.seed)
rng = random.Random(args.seed)
model, tok, meta = checkpoint.load(args.model, device)
model.train()
if args.lora:
    meta = {"family": meta["family"], "base": args.model, "lora": {"r": args.lora, "alpha": 2 * args.lora}}
    lora.add(model, **meta["lora"])
trainable = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(trainable, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)
autocast = torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=device_type != "cpu" and not small)


def hidden(x):
    with autocast:
        return model.hidden(x)


reference = None
if args.kl:
    assert lora.modules(model), "the KL reference is the model with its adapters switched off, so it needs LoRA"

    def reference(x, y):
        lora.enable(model, False)
        logits = rl.target_logits(model, x, y, hidden)
        lora.enable(model, True)
        return logits


# Training prompts, and held-out ones to watch progress on. The held-out ones
# are not the test set: that is kept for scripts/evaluate.py, after training.
if small:
    vocabulary, n = tasks.content_words(), args.iterations * args.prompts
    stream = tasks.words_examples("data/tinystories/train.txt", vocabulary, n, seed=args.seed + 300)
    draw = lambda: [next(stream) for _ in range(args.prompts)]
    held_out = list(tasks.words_examples("data/tinystories/valid.txt", vocabulary, args.eval_n, seed=11))
elif with_tools:
    # the system's turn with the calculator, and the problem; the solution is never shown
    pool = [{"messages": e["messages"][:2], "number": e["number"]} for e in tools.gsm8k_conversations("train", SYSTEM_PROMPT)]
    pool, held_out = pool[:-200], pool[-200:][: args.eval_n]
    draw = lambda: rng.sample(pool, args.prompts)
else:
    pool = tasks.gsm8k_examples("train", SYSTEM_PROMPT)
    pool, held_out = pool[:-200], pool[-200:][: args.eval_n]  # the 200 problems that fine-tuning held out too
    draw = lambda: rng.sample(pool, args.prompts)
score = tasks.SCORE["gsm8k" if with_tools else args.task]
# with the calculator, the limit is on each of the model's turns, as in scripts/evaluate_tools.py
max_new_tokens = args.max_new_tokens or (320 if with_tools else tasks.MAX_NEW_TOKENS[args.task])
end = tok.special_tokens[chat.IM_END]


def held_out_score():
    """Greedy replies to the held-out prompts, scored. The number that says whether training helps."""
    model.eval()
    if with_tools:
        finished, _ = tools.run(model, tok, [e["messages"] for e in held_out], max_new_tokens, batch_size=64)
        replies = [tools.answer_text(c, 2) for c in finished]
    else:
        replies = evaluate.replies(model, tok, held_out, tasks.MAX_NEW_TOKENS[args.task], batch_size=256 if small else 64)
    model.train()
    free_memory(device)
    return sum(score(e, r) for e, r in zip(held_out, replies)) / len(held_out)


os.makedirs(args.out, exist_ok=True)
log = open(f"{args.out}/log.jsonl", "w")
print(f"device {device}, {args.model} on {args.task}: {args.iterations} iterations of {args.prompts} prompts x {args.samples} replies")
print(f"training {sum(p.numel() for p in trainable) / 1e6:.1f}M parameters, lr {args.lr:g}, kl {args.kl}")
start = time.time()
value = held_out_score()
print(f"iteration   0 | held-out {value:.3f}")
log.write(json.dumps({"iteration": 0, "held_out": value, "elapsed": 0.0}) + "\n")
for iteration in range(1, args.iterations + 1):
    examples = draw()
    # 1. Act: several replies to every prompt, sampled from the current policy. With the
    # calculator a reply is several turns, and the calls in it are run as it is sampled.
    masks = None
    with autocast:
        if with_tools:
            prompts, replies, masks, conversations, counts = tools.rollout(model, tok, [e["messages"] for e in examples], args.samples, max_new_tokens)
            texts = [tools.answer_text(c, 2) for c in conversations]
        else:
            prompts, replies, texts = rl.rollout(model, tok, [e["messages"] for e in examples], args.samples, max_new_tokens)
    # 2. Verify: a program scores every reply. No human and no learned model is involved.
    rewards = torch.tensor([score(examples[i // args.samples], text) for i, text in enumerate(texts)], device=device)
    finished = torch.tensor([reply[-1] == end for reply in replies], device=device)
    if args.must_finish:
        rewards = rewards * finished
    # 3. Compare within each prompt's group, and 4. take one policy-gradient step.
    advantages = rl.group_advantages(rewards, args.samples)
    # Replies with calls run to over a thousand tokens: their gradients are taken two at a time.
    chunk = 64 if small else 2 if with_tools else 4
    stats = rl.update(model, optimizer, prompts, replies, advantages, device, kl_coef=args.kl, reference=reference, chunk=chunk, hidden=hidden, masks=masks)
    free_memory(device)

    groups = rewards.view(-1, args.samples)
    record = {
        "iteration": iteration,
        "reward": rewards.mean().item(),
        "informative_groups": (groups.std(dim=1) > 0).float().mean().item(),  # groups whose replies differ in reward
        "tokens": sum(len(r) for r in replies) / len(replies),
        "finished": finished.float().mean().item(),  # replies that ended by themselves
        "kl": stats["kl"],
        "elapsed": time.time() - start,
    }
    if with_tools:
        record |= {"calls": sum(c["calls"] for c in counts) / len(counts), "bad_calls": sum(c["bad_calls"] for c in counts), "with_call": sum(c["calls"] > 0 for c in counts) / len(counts)}
        record["tokens"] = sum(sum(m) for m in masks) / len(masks)  # the model's own tokens
    if iteration % args.eval_every == 0 or iteration == args.iterations:
        record["held_out"] = held_out_score()
    log.write(json.dumps(record) + "\n")
    log.flush()
    if "held_out" in record or iteration % 5 == 0 or iteration == 1:
        line = f"iteration {iteration:3d} | reward {record['reward']:.3f} | informative groups {record['informative_groups']:.0%} | {record['tokens']:.0f} tokens, {record['finished']:.0%} finished"
        if with_tools:
            line += f" | {record['calls']:.1f} calls"
        print(line + (f" | held-out {record['held_out']:.3f}" if "held_out" in record else ""))

minutes = (time.time() - start) / 60
meta |= {"method": "grpo", "task": args.task, "step": args.iterations, "val_loss": 0.0, "minutes": minutes, "args": vars(args)}
if not lora.modules(model):
    meta |= {"base": args.model}
checkpoint.save(args.out, model, meta)
print(f"done in {minutes:.1f} minutes, saved to {args.out}")
