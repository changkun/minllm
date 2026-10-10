"""
Train the tokenizer on TinyStories and show what it learned.

    uv run scripts/train_tokenizer.py

See docs/01-tokenizer.md.
"""

import argparse
import os
import time

from minllm.data import SPECIAL_TOKENS, read_stories
from minllm.tokenizer import Tokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--data", default="data/tinystories")
parser.add_argument("--max-mb", type=int, default=64, help="megabytes of the training split to learn from")
parser.add_argument("--vocab-size", type=int, default=4096)
parser.add_argument("--out", default="out/tokenizer.json")
args = parser.parse_args()

start = time.time()
stories = read_stories(f"{args.data}/train.txt", max_bytes=args.max_mb << 20)
tok = Tokenizer.train(stories, args.vocab_size, SPECIAL_TOKENS, verbose=True)
print(f"learned {len(tok.merges)} merges in {time.time() - start:.1f}s")

os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
tok.save(args.out)
print(f"saved {args.out}, vocabulary size {tok.vocab_size}")

# How much shorter is text in tokens than in bytes? Measured on stories the
# tokenizer has not seen.
n_bytes = n_tokens = 0
for story in read_stories(f"{args.data}/valid.txt", max_bytes=4 << 20):
    n_bytes += len(story.encode("utf-8"))
    n_tokens += len(tok.encode_ordinary(story))
print(f"compression on held-out stories: {n_bytes / n_tokens:.2f} bytes per token")

for text in ["Once upon a time, there was a little girl named Lily.", "Tokenization of unfamiliar words: xylophone, 2026."]:
    print(" | ".join(tok.decode([i]) for i in tok.encode(text)))
