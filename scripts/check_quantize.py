"""
Check an exported quantized model against MLX, which computes with the packed
words directly: the same logits, and the same greedy continuation.

    uv run scripts/export.py --model out/qwen-q4
    uv run --with mlx-lm scripts/check_quantize.py --model out/qwen-q4

mlx-lm is installed for this command only. See docs/10a-quantization.md.
"""

import argparse
import os

import mlx.core as mx
import numpy as np
import torch
from mlx_lm import load, stream_generate

from minllm import chat, checkpoint, quantize
from minllm.inference import generate

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True, help="the quantized model that was exported")
parser.add_argument("--export", default=None, help="where it was exported to")
parser.add_argument("--tokens", type=int, default=48)
args = parser.parse_args()
exported = args.export or f"out/export/{os.path.basename(args.model.rstrip('/'))}"

ours, tok, meta = checkpoint.load(args.model, "cpu")
ours = quantize.expand(ours)  # the rounded weights at full size, in 32-bit floats
theirs, their_tok = load(exported)

messages = [{"role": "user", "content": "Write a short story using these words: garden, kitten."}]
if meta["family"] == "qwen":
    messages = [{"role": "system", "content": "You are a helpful assistant."}] + messages
ids = chat.encode(tok, messages)
assert their_tok.encode(chat.render(messages), add_special_tokens=False) == ids, "token ids differ"

with torch.no_grad():
    a = ours(torch.tensor([ids]))[0][0].numpy()
b = np.array(theirs(mx.array([ids]))[0].astype(mx.float32))
print(f"logits: largest difference {np.abs(a - b).max():.3f} where the largest logit is {np.abs(a).max():.1f}, over {a.size:,} values")

our_greedy = generate(ours, [ids], args.tokens, temperature=0)[0]
their_greedy = [r.token for r in stream_generate(theirs, their_tok, ids, max_tokens=args.tokens)][: len(our_greedy)]
same = next((i for i, (x, y) in enumerate(zip(our_greedy, their_greedy)) if x != y), len(our_greedy))
print(f"generation: identical for {same} of {len(our_greedy)} tokens: {tok.decode(our_greedy[:same])!r}")
