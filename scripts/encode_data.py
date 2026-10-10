"""
Encode TinyStories with the trained tokenizer. Each split becomes one flat
array of token ids on disk, the form the training loop reads.

    uv run scripts/encode_data.py
"""

import argparse
import time
from array import array

from minllm.data import EOT, read_stories
from minllm.tokenizer import Tokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--data", default="data/tinystories")
parser.add_argument("--tokenizer", default="out/tokenizer.json")
args = parser.parse_args()

tok = Tokenizer.load(args.tokenizer)
assert tok.vocab_size <= 1 << 16, "token ids are stored as 16-bit integers"
eot = tok.special_tokens[EOT]

for split in ["valid", "train"]:
    start = time.time()
    ids = array("H")  # unsigned 16-bit
    n_stories = 0
    for story in read_stories(f"{args.data}/{split}.txt"):
        # Every story opens with <|endoftext|>. The model learns that this
        # token means "a new story starts here", and that it also ends one.
        ids.append(eot)
        ids.extend(tok.encode_ordinary(story))
        n_stories += 1
    with open(f"{args.data}/{split}.bin", "wb") as f:
        ids.tofile(f)
    print(f"{split}: {n_stories:,} stories, {len(ids):,} tokens, {time.time() - start:.0f}s")
