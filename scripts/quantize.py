"""
Quantize a model: store its matrices in a few bits per weight, and measure
what that costs on held-out text.

    uv run scripts/quantize.py --model qwen --bits 4        # writes out/qwen-q4
    uv run scripts/quantize.py --model out/base --bits 8    # writes out/base-q8
    uv run scripts/quantize.py --model qwen --sweep         # the loss at every bit width, writes runs/quant/

See docs/10a-quantization.md.
"""

import argparse
import json
import math
import os

import torch
import torch.nn.functional as F

from minllm import chat, checkpoint, quantize, tasks
from minllm.common import free_memory, pick_device
from minllm.data import get_batch, load_tokens
from minllm.pretrained import SYSTEM_PROMPT

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True, help="qwen, or the directory of a trained model")
parser.add_argument("--bits", type=int, default=4, choices=[2, 4, 8], help="bits per weight: the widths that pack evenly into 32-bit words")
parser.add_argument("--group", type=int, default=64, help="weights that share a scale and an offset; 0 is a whole row")
parser.add_argument("--embed-bits", type=int, default=None, help="bits for the embedding table: --bits by default, 0 keeps it as it is")
parser.add_argument("--out", default=None)
parser.add_argument("--sweep", action="store_true", help="measure many settings without writing a model")
args = parser.parse_args()

device = pick_device()
device_type = torch.device(device).type
autocast = torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=device_type != "cpu")
name = os.path.basename(args.model.rstrip("/"))
model, tok, meta = checkpoint.load(args.model, "cpu")
family = meta["family"]

# Held-out text, the same that the training scripts watch: 40 batches of
# stories for the small model, and for Qwen the reference solutions of 160
# GSM8K problems that fine-tuning never trained on.
if family == "small":
    generator = torch.Generator().manual_seed(0)
    tokens = load_tokens("data/tinystories/valid.bin")
    batches = [get_batch(tokens, 64, model.cfg.max_seq_len, "cpu", generator) for _ in range(40)]
else:
    examples = tasks.gsm8k_examples("train", SYSTEM_PROMPT)[-200:]
    pairs = [chat.training_example(tok, e["messages"] + [{"role": "assistant", "content": e["answer"]}]) for e in examples]
    pairs = [pair for pair in pairs if len(pair[0]) <= 384][:160]
    batches = []
    for i in range(0, len(pairs), 8):
        width = math.ceil(max(len(ids) for ids, _ in pairs[i : i + 8]) / 32) * 32
        x, y = torch.zeros(8, width, dtype=torch.long), torch.full((8, width), -100, dtype=torch.long)
        for row, (ids, labels) in enumerate(pairs[i : i + 8]):
            x[row, : len(ids) - 1], y[row, : len(ids) - 1] = torch.tensor(ids[:-1]), torch.tensor(labels[1:])
        batches.append((x, y))


@torch.no_grad()
def held_out_loss(model):
    total = 0.0
    for x, y in batches:
        x, y = x.to(device), y.to(device)
        with autocast:
            h = model.hidden(x)
            learn = y != -100
            total += F.cross_entropy(model.lm_head(h[learn]).float(), y[learn]).item()
    free_memory(device)
    return total / len(batches)


def megabytes(path):
    return os.path.getsize(path) / 1e6


model = model.to(device)
parameters = model.num_params()
before = held_out_loss(model)
print(f"{args.model}: {parameters / 1e6:.1f}M parameters, held-out loss {before:.4f}")

if args.sweep:
    # Round the weights in place, measure, and put the originals back.
    original = {k: v.clone() for k, v in model.state_dict().items()}
    matrices = sum(p.numel() for p in model.parameters() if p.dim() == 2)

    def measure(bits, group, embed_bits=None):
        model.load_state_dict(original)
        return held_out_loss(quantize.rounded(model, bits, group, embed_bits))

    rows = []
    for group in [0, 64, 16]:
        for bits in [8, 6, 5, 4, 3, 2]:
            loss = measure(bits, group)
            # a weight costs its code, and its share of the group's 16-bit scale and offset
            cost = sum(p.numel() * (bits + 32 / (group or p.shape[1])) for p in model.parameters() if p.dim() == 2) / matrices
            rows.append({"bits": bits, "group": group, "bits_per_weight": cost, "loss": loss})
            print(f"{bits} bits, {'rows' if not group else f'groups of {group}':13s} {cost:5.2f} bits per weight | loss {loss:.4f} ({loss - before:+.4f})")
    parts = {"blocks": measure(4, 64, embed_bits=0), "embedding": measure(0, 64, embed_bits=4)}
    for part, loss in parts.items():
        print(f"4 bits in groups of 64, only the {part:9s} | loss {loss:.4f} ({loss - before:+.4f})")
    os.makedirs("runs/quant", exist_ok=True)
    with open(f"runs/quant/sweep-{name}.json", "w") as f:
        json.dump({"model": args.model, "parameters": parameters, "loss": before, "settings": rows, "parts_at_4_bits": parts}, f, indent=1)
    print(f"wrote runs/quant/sweep-{name}.json")
    raise SystemExit

out = args.out or f"out/{name}-q{args.bits}"
quant = {"bits": args.bits, "group": args.group, "embed_bits": args.embed_bits}
checkpoint.save(out, quantize.apply(model.cpu(), **quant), {"family": family, "base": args.model, "quant": quant})
del model

# Read the directory back, as anyone who was given it would, and measure that.
model, _, _ = checkpoint.load(out, device)
after = held_out_loss(model)
size = megabytes(f"{out}/model.safetensors")
print(f"{out}: {size:.1f} MB, {size * 8e6 / parameters:.2f} bits per parameter, held-out loss {after:.4f} ({after - before:+.4f})")
with open(f"{out}/meta.json") as f:
    meta = json.load(f)
with open(f"{out}/meta.json", "w") as f:
    json.dump(meta | {"parameters": parameters, "megabytes": size, "loss": after, "loss_before": before}, f, indent=1)
