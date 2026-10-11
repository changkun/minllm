"""
Self-training with a verifier: the model answers training prompts several
times, a program keeps the answers that are right, the model is fine-tuned on
those, and the better model answers again.

    uv run scripts/self_train.py --task words     # the 14M model learns to use the requested words
    uv run scripts/self_train.py --task gsm8k     # the 0.5B model gets better at math problems

The defaults spend as many replies as scripts/grpo.py does on the same task,
from the same starting model. See docs/09a-self-training.md.
"""

import argparse
import json
import os
import random
import time

import torch
import torch.nn.functional as F

from minllm import chat, checkpoint, evaluate, lora, rl, selftrain, tasks
from minllm.common import free_memory, pick_device
from minllm.pretrained import SYSTEM_PROMPT

parser = argparse.ArgumentParser()
parser.add_argument("--task", choices=["words", "gsm8k"], required=True)
parser.add_argument("--model", default=None, help="the starting point")
parser.add_argument("--out", default=None)
parser.add_argument("--rounds", type=int, default=None)
parser.add_argument("--prompts", type=int, default=None, help="training prompts per round")
parser.add_argument("--samples", type=int, default=None, help="replies per prompt")
parser.add_argument("--per-prompt", type=int, default=2, help="right replies kept per prompt, at most")
parser.add_argument("--lr", type=float, default=None)
parser.add_argument("--lora", type=int, default=0, help="add fresh LoRA adapters of this rank instead of training the weights the model arrives with")
parser.add_argument("--batch-size", type=int, default=None, help="kept replies per fine-tuning step")
parser.add_argument("--max-new-tokens", type=int, default=None, help="the limit on a sampled reply's length")
parser.add_argument("--eval-n", type=int, default=None, help="held-out examples per evaluation")
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

# Fine-tuning's learning rates of stage 6, halved, since the model is already trained on the task.
small = args.task == "words"
args.model = args.model or ("out/sft-words" if small else "out/sft-gsm8k-lora")
args.out = args.out or f"out/self-{args.task}"
args.rounds = args.rounds or (6 if small else 2)
args.prompts = args.prompts or (1600 if small else 480)
args.samples = args.samples or (4 if small else 8)
args.lr = args.lr or (1.5e-4 if small else 1e-4)
args.batch_size = args.batch_size or (32 if small else 8)
args.max_new_tokens = args.max_new_tokens or (tasks.MAX_NEW_TOKENS["words"] if small else 256)
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


# Training prompts and held-out ones, drawn as scripts/grpo.py draws them.
if small:
    vocabulary = tasks.content_words()
    stream = tasks.words_examples("data/tinystories/train.txt", vocabulary, args.rounds * args.prompts, seed=args.seed + 300)
    draw = lambda: [next(stream) for _ in range(args.prompts)]
    held_out = list(tasks.words_examples("data/tinystories/valid.txt", vocabulary, args.eval_n, seed=11))
else:
    pool = tasks.gsm8k_examples("train", SYSTEM_PROMPT)
    pool, held_out = pool[:-200], pool[-200:][: args.eval_n]
    draw = lambda: rng.sample(pool, args.prompts)
score = tasks.SCORE[args.task]
end = tok.special_tokens[chat.IM_END]
chunk = (64 if small else 8) * args.samples  # replies generated at once


def held_out_score():
    """Greedy replies to the held-out prompts, scored."""
    model.eval()
    replies = evaluate.replies(model, tok, held_out, tasks.MAX_NEW_TOKENS[args.task], batch_size=256 if small else 64)
    model.train()
    free_memory(device)
    return sum(score(e, r) for e, r in zip(held_out, replies)) / len(held_out)


os.makedirs(args.out, exist_ok=True)
log = open(f"{args.out}/log.jsonl", "w")
print(f"device {device}, {args.model} on {args.task}: {args.rounds} rounds of {args.prompts} prompts x {args.samples} replies")
print(f"training {sum(p.numel() for p in trainable) / 1e6:.1f}M parameters, lr {args.lr:g}")
start = time.time()
value = held_out_score()
print(f"round 0 | held-out {value:.3f}")
log.write(json.dumps({"round": 0, "held_out": value, "elapsed": 0.0}) + "\n")
for round in range(1, args.rounds + 1):
    examples = draw()
    # 1. Answer: several replies to every prompt, sampled from the current model.
    prompts, replies, texts = [], [], []
    for i in range(0, len(examples), chunk // args.samples):
        with autocast:
            p, r, t = rl.rollout(model, tok, [e["messages"] for e in examples[i : i + chunk // args.samples]], args.samples, args.max_new_tokens)
        prompts, replies, texts = prompts + p, replies + r, texts + t
        free_memory(device)
    # 2. Verify, and 3. keep the replies that are right.
    scores = [score(examples[i // args.samples], text) for i, text in enumerate(texts)]
    finished = [reply[-1] == end for reply in replies]
    kept = selftrain.keep(texts, scores, finished, args.samples, args.per_prompt)
    # 4. Fine-tune on them, once through: the loss of stage 6, on the model's own replies.
    rng.shuffle(kept)
    loss_sum = steps = 0
    for i in range(0, len(kept), args.batch_size):
        batch = kept[i : i + args.batch_size]
        x, y, _ = rl.batch([prompts[j] for j in batch], [replies[j] for j in batch], device)
        loss = F.cross_entropy(rl.target_logits(model, x, y, hidden), y[y != -100])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        loss_sum, steps = loss_sum + loss.item(), steps + 1
        if steps % 4 == 0:
            free_memory(device)

    right = [finished[i] and scores[i] == 1 for i in range(len(texts))]
    record = {
        "round": round,
        "right": sum(right) / len(right),  # sampled replies that are finished and entirely right
        "prompts_with_a_right_reply": sum(any(right[i : i + args.samples]) for i in range(0, len(right), args.samples)) / len(examples),
        "kept": len(kept),
        "steps": steps,
        "loss": loss_sum / max(steps, 1),
        "tokens": sum(len(r) for r in replies) / len(replies),
        "finished": sum(finished) / len(finished),
        "held_out": held_out_score(),
        "elapsed": time.time() - start,
    }
    log.write(json.dumps(record) + "\n")
    log.flush()
    print(f"round {round} | {record['right']:.1%} of {len(texts)} replies right, {record['prompts_with_a_right_reply']:.1%} of prompts have one | kept {len(kept)}, {steps} steps, loss {record['loss']:.3f} | {record['tokens']:.0f} tokens, {record['finished']:.0%} finished | held-out {record['held_out']:.3f}")

minutes = (time.time() - start) / 60
meta |= {"method": "self-training", "task": args.task, "step": args.rounds, "val_loss": 0.0, "minutes": minutes, "args": vars(args)}
if not lora.modules(model):
    meta |= {"base": args.model}
checkpoint.save(args.out, model, meta)
print(f"done in {minutes:.1f} minutes, saved to {args.out}")
