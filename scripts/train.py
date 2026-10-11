"""
Pretrain the model on TinyStories: read a window of text, predict every next
token, nudge the weights to make the right token more likely. Repeat.

    uv run scripts/train.py                     # the default run
    uv run scripts/train.py --resume            # continue an interrupted run
    uv run scripts/train.py --steps 200 --dim 128 --n-layers 2 --out out/debug
    uv run scripts/train.py --n-experts 8 --mlp-dim 512 --out out/moe    # a mixture of experts

See docs/03-pretraining.md, and docs/02a-moe.md for the mixture of experts.
"""

import argparse
import json
import math
import os
import time
from dataclasses import asdict

import torch

from minllm.common import load_checkpoint, pick_device
from minllm.data import EOT, get_batch, load_tokens
from minllm.model import Config, Transformer
from minllm.tokenizer import Tokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--data", default="data/tinystories")
parser.add_argument("--tokenizer", default="out/tokenizer.json")
parser.add_argument("--out", default="out/base")
parser.add_argument("--resume", action="store_true", help="continue from the checkpoint in --out")
# model
parser.add_argument("--dim", type=int, default=Config.dim)
parser.add_argument("--n-layers", type=int, default=Config.n_layers)
parser.add_argument("--n-heads", type=int, default=Config.n_heads)
parser.add_argument("--n-kv-heads", type=int, default=Config.n_kv_heads)
parser.add_argument("--mlp-dim", type=int, default=Config.mlp_dim)
parser.add_argument("--seq-len", type=int, default=256, help="tokens per training window")
# mixture of experts
parser.add_argument("--n-experts", type=int, default=0, help="above 0, every MLP becomes this many experts of width --mlp-dim")
parser.add_argument("--n-active", type=int, default=Config.n_active, help="experts that process each token")
parser.add_argument("--balance", type=float, default=0.01, help="weight of the loss that spreads the tokens over the experts")
parser.add_argument("--capacity", type=float, default=1.5, help="tokens an expert has room for in a training step, as a multiple of an even split")
# optimization
parser.add_argument("--steps", type=int, default=6000)
parser.add_argument("--batch-size", type=int, default=64)
parser.add_argument("--lr", type=float, default=1e-3, help="peak learning rate")
parser.add_argument("--warmup", type=int, default=200, help="steps over which the learning rate ramps up")
parser.add_argument("--weight-decay", type=float, default=0.1)
parser.add_argument("--grad-clip", type=float, default=1.0)
parser.add_argument("--seed", type=int, default=0)
# evaluation and logging
parser.add_argument("--log-every", type=int, default=50)
parser.add_argument("--eval-every", type=int, default=500)
parser.add_argument("--eval-batches", type=int, default=40)
# speed
parser.add_argument("--no-compile", action="store_true", help="skip torch.compile")
parser.add_argument("--no-amp", action="store_true", help="compute in 32-bit floats throughout")
args = parser.parse_args()

device = pick_device()
torch.manual_seed(args.seed)
os.makedirs(args.out, exist_ok=True)
ckpt_path, log_path = f"{args.out}/ckpt.pt", f"{args.out}/log.jsonl"
if not args.resume:
    if os.path.exists(ckpt_path):
        raise SystemExit(f"{ckpt_path} exists: pass --resume to continue it, or pick another --out")
    open(log_path, "w").close()

tok = Tokenizer.load(args.tokenizer)
train_tokens = load_tokens(f"{args.data}/train.bin")
valid_tokens = load_tokens(f"{args.data}/valid.bin")

if args.resume:
    model, ckpt = load_checkpoint(ckpt_path, device)
    cfg = model.cfg
    args.seq_len = cfg.max_seq_len
else:
    cfg = Config(
        vocab_size=tok.vocab_size,
        max_seq_len=args.seq_len,
        dim=args.dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        n_kv_heads=args.n_kv_heads,
        mlp_dim=args.mlp_dim,
        n_experts=args.n_experts,
        n_active=args.n_active,
    )
    model = Transformer(cfg).to(device)
mixtures = model.mixtures()

# AdamW. Weight decay shrinks the weights a little every step, which keeps the
# matrices from growing without bound; the gains of the norm layers are exempt.
matrices = [p for p in model.parameters() if p.dim() >= 2]
gains = [p for p in model.parameters() if p.dim() < 2]
optimizer = torch.optim.AdamW(
    [{"params": matrices, "weight_decay": args.weight_decay}, {"params": gains, "weight_decay": 0.0}],
    lr=args.lr,
    betas=(0.9, 0.95),
)
step = 0
if args.resume:
    optimizer.load_state_dict(ckpt["optimizer"])
    step = ckpt["step"]

# Two things that make a step faster without changing what it computes:
# torch.compile fuses the model's many small operations into few large ones,
# and autocast runs the arithmetic in 16-bit floats while the weights stay 32-bit.
forward = model if args.no_compile else torch.compile(model)
device_type = torch.device(device).type
use_amp = device_type != "cpu" and not args.no_amp
autocast = torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=use_amp)


def set_capacity(capacity):
    for m in mixtures:
        m.capacity = capacity


def lr_at(step):
    """Ramp up linearly, then follow a cosine down to a tenth of the peak."""
    if step < args.warmup:
        return args.lr * (step + 1) / args.warmup
    progress = (step - args.warmup) / max(1, args.steps - args.warmup)
    return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


@torch.no_grad()
def evaluate():
    """Mean loss on held-out stories. The batches are the same on every call."""
    generator = torch.Generator().manual_seed(0)
    total = 0.0
    set_capacity(cfg.n_experts / cfg.n_active if mixtures else None)  # room for every token: nothing is dropped
    for _ in range(args.eval_batches):
        x, y = get_batch(valid_tokens, args.batch_size, args.seq_len, device, generator)
        with autocast:
            total += forward(x, y)[1].item()
    return total / args.eval_batches


def sample():
    set_capacity(None)
    prompt = torch.tensor([tok.encode(EOT + "Once upon a time")], device=device)
    ids = model.generate(prompt, max_new_tokens=80, temperature=0.8, top_k=50)[0, 1:].tolist()
    return tok.decode(ids).split(EOT)[0].replace("\n", " ")


def log(**record):
    with open(log_path, "a") as f:
        f.write(json.dumps(record) + "\n")


tokens_per_step = args.batch_size * args.seq_len
print(f"device {device}, compile {not args.no_compile}, amp {use_amp}")
print(f"model: {model.num_params() / 1e6:.1f}M parameters, {cfg}")
if mixtures:
    active = model.num_params() - sum(p.numel() for m in mixtures for e in m.experts[cfg.n_active :] for p in e.parameters())
    print(f"experts: {cfg.n_active} of {cfg.n_experts} per token, {active / 1e6:.1f}M parameters active, balance {args.balance}, capacity {args.capacity}")
print(f"data: {len(train_tokens) / 1e6:.0f}M training tokens, {tokens_per_step} per step, ", end="")
print(f"{args.steps * tokens_per_step / len(train_tokens):.1f} passes over the data in {args.steps} steps")

start = interval_start = time.time()
interval_loss = 0.0
while True:
    if step % args.eval_every == 0 or step == args.steps:
        val_loss = evaluate()
        print(f"step {step:6d} | val loss {val_loss:.4f} | {sample()}")
        log(step=step, val_loss=val_loss, elapsed=time.time() - start)
        if step > 0:
            state = {"config": asdict(cfg), "model": model.state_dict(), "optimizer": optimizer.state_dict()}
            torch.save(state | {"step": step, "val_loss": val_loss}, ckpt_path)
        interval_start = time.time()  # leave evaluation out of the speed measurement
    if step == args.steps:
        break

    for group in optimizer.param_groups:
        group["lr"] = lr_at(step)
    x, y = get_batch(train_tokens, args.batch_size, args.seq_len, device)
    set_capacity(args.capacity)
    with autocast:
        _, loss = forward(x, y)
        # A mixture of experts is trained on one more term, which asks the
        # routers to spread the tokens. The loss that is logged stays the
        # cross-entropy, so that it compares with a dense model's.
        objective = loss + args.balance * model.balance_loss() if mixtures else loss
    optimizer.zero_grad(set_to_none=True)
    objective.backward()  # d loss / d weight, for every weight
    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)  # one bad batch cannot throw the weights far
    optimizer.step()
    step += 1

    interval_loss += loss.item()
    if step % args.log_every == 0:
        now = time.time()
        speed = args.log_every * tokens_per_step / (now - interval_start)
        train_loss = interval_loss / args.log_every
        print(f"step {step:6d} | loss {train_loss:.4f} | lr {lr_at(step - 1):.2e} | {speed / 1e3:.0f}k tok/s")
        record = dict(step=step, train_loss=train_loss, lr=lr_at(step - 1), tokens_per_second=speed)
        if mixtures:
            # From the last batch: the balance loss, the routed work that found no room,
            # and each expert's share of the work in every block.
            record["balance"] = model.balance_loss().item()
            record["dropped"] = sum(m.dropped.item() for m in mixtures) / len(mixtures)
            record["expert_share"] = [[round(v, 4) for v in m.share.tolist()] for m in mixtures]
        log(**record)
        interval_start, interval_loss = now, 0.0

print(f"done in {(time.time() - start) / 60:.1f} minutes, checkpoint at {ckpt_path}")
