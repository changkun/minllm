"""
Measure what a larger vocabulary buys and what it costs.

    uv run scripts/bench_tokenizer.py

Writes runs/bench/tokenizer.json, which docs/01-tokenizer.md quotes.
"""

import argparse
import json
import os
import re
import time
from collections import Counter

from minllm.data import SPECIAL_TOKENS, read_stories
from minllm.tokenizer import Tokenizer
from minllm.common import result_path

parser = argparse.ArgumentParser()
parser.add_argument("--data", default="data/tinystories")
parser.add_argument("--max-mb", type=int, default=64)
parser.add_argument("--out", default="runs/bench/tokenizer.json")
args = parser.parse_args()
args.out = result_path(args.out)  # a path under runs/ goes to the results folder

train = list(read_stories(f"{args.data}/train.txt", max_bytes=args.max_mb << 20))
held_out = list(read_stories(f"{args.data}/valid.txt", max_bytes=4 << 20))
held_out_bytes = sum(len(story.encode("utf-8")) for story in held_out)

# The corpus split into chunks, as Tokenizer.train sees it: many, and few of them distinct.
chunks = Counter(c for story in train for c in Tokenizer()._split.findall(story))
print(f"{sum(chunks.values()):,} chunks, {len(chunks):,} distinct")

results = []
print(f"{'vocab':>6} {'train s':>8} {'bytes/token':>12} {'rarest token seen':>18}")
for vocab_size in [512, 1024, 2048, 4096, 8192, 16384]:
    start = time.time()
    tok = Tokenizer.train(train, vocab_size, SPECIAL_TOKENS)
    seconds = time.time() - start
    held_out_tokens = sum(len(tok.encode_ordinary(story)) for story in held_out)
    # How often does the training text use the token that was learned last?
    counts = Counter()
    for story in train:
        counts.update(tok.encode_ordinary(story))
    last = max(tok.merges.values())
    row = {
        "vocab_size": tok.vocab_size,
        "train_seconds": seconds,
        "bytes_per_token": held_out_bytes / held_out_tokens,
        "last_merge": tok.vocab[last].decode("utf-8", errors="replace"),
        "last_merge_count": counts[last],
    }
    if vocab_size == 4096:  # the vocabulary of the later stages: how many of the held-out words are one token
        words = [c for story in held_out for c in tok._split.findall(story) if re.fullmatch(r" ?[A-Za-z]+", c)]
        row["words_single_token"] = sum(len(tok.encode_ordinary(w)) == 1 for w in words) / len(words)
    results.append(row)
    print(f"{row['vocab_size']:>6} {seconds:>8.1f} {row['bytes_per_token']:>12.2f} {row['last_merge_count']:>18,}")

os.makedirs(os.path.dirname(args.out), exist_ok=True)
with open(args.out, "w") as f:
    json.dump({"train_megabytes": args.max_mb, "chunks": sum(chunks.values()), "distinct_chunks": len(chunks), "sizes": results}, f, indent=1)
print(f"wrote {args.out}")
