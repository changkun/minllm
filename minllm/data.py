"""
The training corpus: TinyStories (Eldan & Li, 2023, arXiv:2305.07759).

Short stories written by GPT-4 in the vocabulary of a small child. The
language is simple enough that a model with a few million parameters learns to
write fluent English from it, which makes it the right corpus for a laptop.
"""

import numpy as np
import torch

from minllm.common import fetch

URL = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-{split}.txt"
EOT = "<|endoftext|>"  # separates the stories in the raw files
# The chat markers are reserved from the start, so that the vocabulary, and with
# it the shape of the model, stays the same once the chat stages need them.
SPECIAL_TOKENS = [EOT, "<|im_start|>", "<|im_end|>"]


def download(split, path, max_bytes=None):
    """
    Fetch one split ("train" or "valid") to `path`. With `max_bytes`, fetch
    only the head of the file and drop the story that the cut falls into.
    """
    if not fetch(URL.format(split=split), path, max_bytes):
        print(f"{path} exists, skipping download")
    elif max_bytes:
        with open(path, "rb+") as f:
            data = f.read()
            f.truncate(data.rindex(EOT.encode()) + len(EOT))


def read_stories(path, max_bytes=None):
    """Yield the stories of a raw file one at a time, without the separators."""
    rest, seen = "", 0
    with open(path, encoding="utf-8") as f:
        while block := f.read(1 << 20):
            *stories, rest = (rest + block).split(EOT)
            for story in stories:
                if story := story.strip():
                    yield story
            seen += len(block)
            if max_bytes and seen >= max_bytes:
                return
    if rest.strip():
        yield rest.strip()


def load_tokens(path):
    """An encoded split: one long array of token ids, read from disk on demand."""
    return np.memmap(path, dtype=np.uint16, mode="r")


def get_batch(tokens, batch_size, seq_len, device, generator=None):
    """
    Cut random windows out of the token stream. The model reads x and has to
    predict y, which is x shifted left by one token: at every position, the
    target is the token that comes next.
    """
    starts = torch.randint(len(tokens) - seq_len - 1, (batch_size,), generator=generator)
    windows = np.stack([tokens[s : s + seq_len + 1] for s in starts.tolist()])
    windows = torch.from_numpy(windows.astype(np.int64)).to(device)
    return windows[:, :-1], windows[:, 1:]
