"""
Save a model's whole prediction for the next token after a few prompts, for
the interactive figures of the website: the sampling controls of stage 4 and
the teacher's temperature in docs/03a-distillation.md.

    uv run scripts/dump_next_token.py                                   # out/base, writes runs/base/next-token.json
    uv run scripts/dump_next_token.py --model out/wide --out runs/distill/next-token.json
"""

import argparse
import json
import os

import torch

from minllm.common import load_checkpoint, pick_device, result_path
from minllm.data import EOT
from minllm.tokenizer import Tokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="out/base")
parser.add_argument("--tokenizer", default="out/tokenizer.json")
parser.add_argument("--prompts", nargs="+", default=["Once upon a time, there was a little", "Once upon a time, there was a little girl named", "She loved to play with her", "The tiny robot was sad because"])
parser.add_argument("--out", default="runs/base/next-token.json")
args = parser.parse_args()
args.out = result_path(args.out)  # a path under runs/ goes to the results folder

device = pick_device()
tok = Tokenizer.load(args.tokenizer)
model, ckpt = load_checkpoint(f"{args.model}/ckpt.pt", device)
model.eval()

prompts = []
for prompt in args.prompts:
    ids = [tok.special_tokens[EOT]] + tok.encode_ordinary(prompt)
    with torch.no_grad():
        logits = model(torch.tensor([ids], device=device))[0][0, -1].float().cpu()
    # every token's logit, relative to the largest, so that any temperature can be applied later
    logits = logits - logits.max()
    best = logits.topk(5).indices.tolist()
    prompts.append({"prompt": prompt, "logits": [round(v, 3) for v in logits.tolist()]})
    print(f"{prompt!r} -> " + ", ".join(f"{tok.decode([t])!r} {p:.0%}" for t, p in zip(best, logits.softmax(-1)[best].tolist())))

os.makedirs(os.path.dirname(args.out), exist_ok=True)
with open(args.out, "w") as f:
    json.dump({"model": args.model, "step": ckpt["step"], "vocab_size": model.cfg.vocab_size, "prompts": prompts}, f)
print(f"wrote {args.out}, {os.path.getsize(args.out) / 1e3:.0f} kB")
