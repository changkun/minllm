"""
Save what a teacher predicts at a few positions of a sentence, next to the
token that actually came next, for the figure in docs/03a-distillation.md.
Also measures, on held-out stories, how much the teacher's predictions say
beyond the label.

    uv run scripts/dump_targets.py            # the teacher out/wide, writes runs/distill/targets.json
"""

import argparse
import json
import os

import torch
import torch.nn.functional as F

from minllm.common import load_checkpoint, pick_device
from minllm.data import EOT, get_batch, load_tokens
from minllm.tokenizer import Tokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--teacher", default="out/wide")
parser.add_argument("--tokenizer", default="out/tokenizer.json")
parser.add_argument("--text", default="Once upon a time, there was a little girl named Lily. She loved to play with her red ball.")
parser.add_argument("--after", default="little,named,play with her", help="show the prediction after each of these pieces of the text")
parser.add_argument("--top", type=int, default=6)
parser.add_argument("--out", default="runs/distill/targets.json")
args = parser.parse_args()

device = pick_device()
tok = Tokenizer.load(args.tokenizer)
teacher, ckpt = load_checkpoint(f"{args.teacher}/ckpt.pt", device)
teacher.eval()

positions = []
for piece in args.after.split(","):
    prefix = args.text[: args.text.index(piece) + len(piece)]
    ids = [tok.special_tokens[EOT]] + tok.encode_ordinary(prefix)
    label = tok.encode_ordinary(args.text)[len(ids) - 1]  # the token that follows the prefix in the text
    with torch.no_grad():
        probs = F.softmax(teacher(torch.tensor([ids], device=device))[0][0, -1].float(), dim=-1)
    best = probs.topk(args.top)
    positions.append({
        "context": prefix,
        "label": tok.decode([label]),
        "label_probability": probs[label].item(),
        "top": [{"token": tok.decode([t]), "probability": p} for p, t in zip(best.values.tolist(), best.indices.tolist())],
    })
    print(f"{prefix!r} -> {tok.decode([label])!r} ({probs[label]:.1%}) | " + ", ".join(f"{tok.decode([t])!r} {p:.0%}" for p, t in zip(best.values.tolist(), best.indices.tolist())))

# Over held-out stories: how spread out the teacher's predictions are, and how they relate to the labels.
x, y = get_batch(load_tokens("data/tinystories/valid.bin"), 100, teacher.cfg.max_seq_len, device, torch.Generator().manual_seed(0))
with torch.no_grad():
    logp = torch.cat([F.log_softmax(teacher(rows)[0].float(), dim=-1) for rows in x.split(25)])
p = logp.exp()
entropy = -(p * logp).sum(-1)
stats = {
    "tokens": y.numel(),
    "loss": -logp.gather(-1, y.unsqueeze(-1)).mean().item(),  # the teacher's own loss on these labels
    "entropy": entropy.mean().item(),  # nats: how undecided the teacher is, on average
    "label_is_first_choice": (logp.argmax(-1) == y).float().mean().item(),
    "choices": entropy.exp().mean().item(),  # as if it chose evenly among this many tokens
    "above_one_percent": (p > 0.01).sum(-1).float().mean().item(),  # tokens the teacher gives more than 1%
}
print(f"on {stats['tokens']:,} held-out tokens: loss {stats['loss']:.3f}, entropy {stats['entropy']:.3f} nats, the label is the teacher's first choice {stats['label_is_first_choice']:.1%} of the time, {stats['above_one_percent']:.1f} tokens get more than 1%")

os.makedirs(os.path.dirname(args.out), exist_ok=True)
with open(args.out, "w") as f:
    json.dump({"teacher": args.teacher, "step": ckpt["step"], "text": args.text, "positions": positions, "held_out": stats}, f, indent=1, ensure_ascii=False)
print(f"wrote {args.out}")
