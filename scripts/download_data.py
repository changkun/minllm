"""
Download TinyStories.

    uv run scripts/download_data.py                # validation split, first 512 MB of the training split
    uv run scripts/download_data.py --train-mb 0   # the whole training split, 2.2 GB
"""

import argparse

from minllm.data import download

parser = argparse.ArgumentParser()
parser.add_argument("--dir", default="data/tinystories")
parser.add_argument("--train-mb", type=int, default=512, help="megabytes of the training split to fetch, 0 for all")
args = parser.parse_args()

download("valid", f"{args.dir}/valid.txt")
download("train", f"{args.dir}/train.txt", max_bytes=(args.train_mb << 20) or None)
