"""
Write stories with a trained model.

    uv run scripts/sample.py
    uv run scripts/sample.py --prompt "The tiny robot was sad because" --temperature 0.5
    uv run scripts/sample.py --n 1      # a single story is printed token by token

See docs/04-inference.md.
"""

import argparse

import torch

from minllm.common import load_checkpoint, pick_device
from minllm.data import EOT
from minllm.inference import generate, stream_text
from minllm.tokenizer import Tokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", default="out/base/ckpt.pt")
parser.add_argument("--tokenizer", default="out/tokenizer.json")
parser.add_argument("--prompt", default="Once upon a time")
parser.add_argument("--n", type=int, default=3, help="number of stories")
parser.add_argument("--max-tokens", type=int, default=200)
parser.add_argument("--temperature", type=float, default=0.8, help="0 always picks the likeliest token")
parser.add_argument("--top-k", type=int, default=50, help="sample only among the k likeliest tokens")
parser.add_argument("--top-p", type=float, default=None, help="sample only among the likeliest tokens that sum to p")
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

device = pick_device()
torch.manual_seed(args.seed)
tok = Tokenizer.load(args.tokenizer)
model, ckpt = load_checkpoint(args.ckpt, device)
print(f"{args.ckpt}: step {ckpt['step']}, val loss {ckpt['val_loss']:.4f}\n")

# A story starts after <|endoftext|>, exactly as in the training data, and the
# model ends it by writing <|endoftext|> again.
eot = tok.special_tokens[EOT]
prompt = [eot] + tok.encode_ordinary(args.prompt)
sampling = dict(temperature=args.temperature, top_k=args.top_k, top_p=args.top_p, stop=[eot])

if args.n == 1:
    print(args.prompt, end="", flush=True)
    for piece in stream_text(model, tok, prompt, args.max_tokens, **sampling):
        print(piece, end="", flush=True)
    print()
else:
    for ids in generate(model, [prompt] * args.n, args.max_tokens, **sampling):
        print(args.prompt + tok.decode(ids), end="\n\n---\n\n")
