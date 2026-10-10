"""
The training corpus: TinyStories (Eldan & Li, 2023, arXiv:2305.07759).

Short stories written by GPT-4 in the vocabulary of a small child. The
language is simple enough that a model with a few million parameters learns to
write fluent English from it, which makes it the right corpus for a laptop.
"""

import os
import urllib.request

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
    if os.path.exists(path):
        print(f"{path} exists, skipping download")
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    headers = {"Range": f"bytes=0-{max_bytes - 1}"} if max_bytes else {}
    request = urllib.request.Request(URL.format(split=split), headers=headers)
    tmp = path + ".part"
    with urllib.request.urlopen(request) as response, open(tmp, "wb") as f:
        total = int(response.headers["Content-Length"])
        done = 0
        while block := response.read(1 << 20):
            f.write(block)
            done += len(block)
            print(f"\r{path}: {done >> 20}/{total >> 20} MB", end="", flush=True)
    print()
    if max_bytes:
        with open(tmp, "rb+") as f:
            data = f.read()
            f.truncate(data.rindex(EOT.encode()) + len(EOT))
    os.replace(tmp, path)


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
