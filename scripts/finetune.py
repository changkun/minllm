"""
Fine-tune a model on a task: show it conversations and train it on the replies.

    uv run scripts/finetune.py --task words               # the 14M model learns to follow an instruction
    uv run scripts/finetune.py --task gsm8k --lora 16     # the 0.5B model learns grade-school math, through LoRA
    uv run scripts/finetune.py --task gsm8k               # the same, training every weight

See docs/06-finetuning.md.
"""

import argparse
import json
import math
import os
import random
import time

import torch
import torch.nn.functional as F

from minllm import chat, checkpoint, lora, tasks
from minllm.common import free_memory, pick_device
from minllm.data import EOT
from minllm.inference import generate
from minllm.pretrained import SYSTEM_PROMPT

parser = argparse.ArgumentParser()
parser.add_argument("--task", choices=["words", "gsm8k"], required=True)
parser.add_argument("--base", default=None, help="the model to start from: a directory, or qwen")
parser.add_argument("--out", default=None)
parser.add_argument("--lora", type=int, default=0, help="rank of the LoRA adapters; 0 trains every weight")
parser.add_argument("--lora-alpha", type=int, default=None, help="defaults to twice the rank")
parser.add_argument("--steps", type=int, default=None)
parser.add_argument("--batch-size", type=int, default=None)
parser.add_argument("--lr", type=float, default=None, help="peak learning rate")
parser.add_argument("--warmup", type=int, default=50)
parser.add_argument("--max-len", type=int, default=None, help="longer examples are left out")
parser.add_argument("--eval-every", type=int, default=200)
parser.add_argument("--eval-batches", type=int, default=20)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--no-compile", action="store_true")
args = parser.parse_args()

# Defaults per task. Fine-tuning moves a trained model a little, so its
# learning rates are far below the 1e-3 of pretraining, and lower still when
# every weight of a large model is trained. LoRA's fresh matrices take more.
small = args.task == "words"
args.base = args.base or ("out/base" if small else "qwen")
args.out = args.out or f"out/sft-{args.task}" + ("-lora" if args.lora else "")
args.steps = args.steps or (2000 if small else 900)
args.batch_size = args.batch_size or (64 if small else 8)
args.lr = args.lr or (3e-4 if small else 2e-4 if args.lora else 1e-5)
args.max_len = args.max_len or (256 if small else 384)

device = pick_device()
torch.manual_seed(args.seed)
rng = random.Random(args.seed)
model, tok, base_meta = checkpoint.load(args.base, device)
model.train()

# ------------------------------------------------------------------- data

if small:
    vocabulary = tasks.content_words()
    train = list(tasks.words_examples("data/tinystories/train.txt", vocabulary, args.steps * args.batch_size))
    held_out = list(tasks.words_examples("data/tinystories/valid.txt", vocabulary, 2000, seed=1))
else:
    examples = tasks.gsm8k_examples("train", SYSTEM_PROMPT)
    train, held_out = examples[:-200], examples[-200:]  # the test split stays untouched for evaluation


def encode(examples):
    """Each example as (ids, labels): the conversation with its reference reply, and which tokens to learn."""
    pairs = [chat.training_example(tok, e["messages"] + [{"role": "assistant", "content": e["answer"]}]) for e in examples]
    return [pair for pair in pairs if len(pair[0]) <= args.max_len]


def batch(pairs):
    """
    Stack examples of different lengths into inputs x and targets y. Short
    ones are padded on the right, where the causal mask already keeps the
    padding out of sight of every real token, and padded targets are -100.
    """
    width = math.ceil(max(len(ids) for ids, _ in pairs) / 32) * 32  # few distinct shapes, for the compiler
    x = torch.zeros(len(pairs), width, dtype=torch.long)
    y = torch.full((len(pairs), width), -100, dtype=torch.long)
    for i, (ids, labels) in enumerate(pairs):
        x[i, : len(ids) - 1] = torch.tensor(ids[:-1])
        y[i, : len(ids) - 1] = torch.tensor(labels[1:])  # the target at a position is the next token
    return x.to(device), y.to(device)


train_pairs, held_out_pairs = encode(train), encode(held_out)
held_out_batches = [held_out_pairs[i : i + args.batch_size] for i in range(0, args.eval_batches * args.batch_size, args.batch_size)]

# ------------------------------------------------------------------ model

meta = {"family": base_meta["family"], "base": args.base, "task": args.task}
n_total = model.num_params()
if args.lora:
    meta["lora"] = {"r": args.lora, "alpha": args.lora_alpha or 2 * args.lora}
    trainable = lora.add(model, **meta["lora"])
else:
    trainable = list(model.parameters())
optimizer = torch.optim.AdamW(trainable, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)

hidden = model.hidden if args.no_compile else torch.compile(model.hidden)
device_type = torch.device(device).type
autocast = torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=device_type != "cpu")


def loss_on(pairs):
    x, y = batch(pairs)
    with autocast:
        h = hidden(x)
        # The output layer is the most expensive part of a step, in memory
        # above all, and most positions have no target: the prompt, the
        # padding. So it is applied only where there is something to learn.
        learn = y != -100
        logits = model.lm_head(h[learn])
    return F.cross_entropy(logits.float(), y[learn])


def lr_at(step):
    if step < args.warmup:
        return args.lr * (step + 1) / args.warmup
    progress = (step - args.warmup) / max(1, args.steps - args.warmup)
    return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


@torch.no_grad()
def evaluate():
    return sum(loss_on(pairs).item() for pairs in held_out_batches) / len(held_out_batches)


@torch.no_grad()
def sample():
    """The model's reply to one held-out prompt."""
    prompt = chat.encode(tok, held_out[0]["messages"])
    stop = [tok.special_tokens[chat.IM_END], tok.special_tokens[EOT]]
    with autocast:
        out = generate(model, [prompt], 200, temperature=0.8 if small else 0, top_k=50, stop=stop)[0]
    return tok.decode(out).replace("\n", " ")


os.makedirs(args.out, exist_ok=True)
log_path = f"{args.out}/log.jsonl"
open(log_path, "w").close()


def log(**record):
    with open(log_path, "a") as f:
        f.write(json.dumps(record) + "\n")


n_trainable = sum(p.numel() for p in trainable)
print(f"device {device}, {args.base} on {args.task}: {len(train_pairs):,} examples, {args.steps} steps of {args.batch_size}")
print(f"training {n_trainable / 1e6:.1f}M of {n_total / 1e6:.1f}M parameters ({n_trainable / n_total:.1%}), lr {args.lr:g}")
print(f"prompt: {held_out[0]['messages'][-1]['content'][:100]}")

start = interval_start = time.time()
interval_loss, interval_tokens, step, peak_memory = 0.0, 0, 0, 0.0
while True:
    if step % args.eval_every == 0 or step == args.steps:
        val_loss = evaluate()
        print(f"step {step:5d} | val loss {val_loss:.4f} | {sample()[:160]}")
        log(step=step, val_loss=val_loss, elapsed=time.time() - start)
        interval_start = time.time()
    if step == args.steps:
        break

    for group in optimizer.param_groups:
        group["lr"] = lr_at(step)
    pairs = rng.sample(train_pairs, args.batch_size)
    loss = loss_on(pairs)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(trainable, 1.0)
    optimizer.step()
    step += 1
    if step % 4 == 0:
        peak_memory = max(peak_memory, free_memory(device))

    interval_loss += loss.item()
    interval_tokens += sum(len(ids) for ids, _ in pairs)
    if step % 50 == 0:
        now = time.time()
        speed = interval_tokens / (now - interval_start)
        print(f"step {step:5d} | loss {interval_loss / 50:.4f} | lr {lr_at(step - 1):.2e} | {speed / 1e3:.1f}k tok/s")
        log(step=step, train_loss=interval_loss / 50, lr=lr_at(step - 1), tokens_per_second=speed)
        interval_start, interval_loss, interval_tokens = now, 0.0, 0

minutes = (time.time() - start) / 60
meta |= {"step": step, "val_loss": val_loss, "minutes": minutes, "trainable_parameters": n_trainable, "memory_gb": peak_memory}
checkpoint.save(args.out, model, meta)
print(f"done in {minutes:.1f} minutes, saved to {args.out}")
