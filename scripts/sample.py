"""
Write stories with a trained model.

    uv run scripts/sample.py
    uv run scripts/sample.py --prompt "The tiny robot was sad because" --temperature 0.5
"""

import argparse

import torch

from minllm.common import load_checkpoint, pick_device
from minllm.data import EOT
from minllm.tokenizer import Tokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", default="out/base/ckpt.pt")
parser.add_argument("--tokenizer", default="out/tokenizer.json")
parser.add_argument("--prompt", default="Once upon a time")
parser.add_argument("--n", type=int, default=3, help="number of stories")
parser.add_argument("--max-tokens", type=int, default=200)
parser.add_argument("--temperature", type=float, default=0.8, help="0 always picks the likeliest token")
parser.add_argument("--top-k", type=int, default=50, help="sample only among the k likeliest tokens")
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

device = pick_device()
torch.manual_seed(args.seed)
tok = Tokenizer.load(args.tokenizer)
model, ckpt = load_checkpoint(args.ckpt, device)
print(f"{args.ckpt}: step {ckpt['step']}, val loss {ckpt['val_loss']:.4f}\n")

# A story starts after <|endoftext|>, exactly as in the training data.
prompt = [tok.special_tokens[EOT]] + tok.encode_ordinary(args.prompt)
batch = torch.tensor([prompt] * args.n, device=device)
out = model.generate(batch, args.max_tokens, args.temperature, args.top_k)
for ids in out[:, 1:].tolist():
    print(tok.decode(ids).split(EOT)[0].strip(), end="\n\n---\n\n")
