"""
Export a model of this repository to Hugging Face format, which transformers,
vLLM, MLX and llama.cpp's converter read.

    uv run scripts/export.py --model out/sft-words
    uv run scripts/export.py --model out/grpo-gsm8k

Writes out/export/<name>/. See docs/10-stack.md.
"""

import argparse
import os

from minllm import checkpoint, export

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True, help="qwen, or the directory of a trained model")
parser.add_argument("--out", default=None)
args = parser.parse_args()

out = args.out or f"out/export/{os.path.basename(args.model.rstrip('/'))}"
model, tok, meta = checkpoint.load(args.model, "cpu")
export.export(model, tok, meta, out)
size = sum(os.path.getsize(f"{out}/{name}") for name in os.listdir(out)) / 2**20
print(f"exported {args.model} to {out}: {', '.join(sorted(os.listdir(out)))} ({size:.0f} MB)")
