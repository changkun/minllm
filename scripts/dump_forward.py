"""
Put the 4-bit story model where the website serves it, and save what it
computes for two prompts. The website runs the model in the browser with an
implementation of its own (site/src/labs/model.js), and its test compares
that with what is saved here: a few weights, the logits after each prompt,
and the 20 tokens the model then finds likeliest.

    uv run scripts/quantize.py --model out/base --bits 4     # writes out/base-q4, 8 MB
    uv run scripts/dump_forward.py                           # writes site/public/models/base-q4 and site/test/forward.json
"""

import argparse
import json
import os
import shutil

import torch

from minllm import checkpoint, inference, quantize
from minllm.data import EOT

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="out/base-q4")
parser.add_argument("--prompts", nargs="+", default=["Once upon a time", "The tiny robot was sad because"])
parser.add_argument("--site", default="site/public/models/base-q4")
parser.add_argument("--out", default="site/test/forward.json")
args = parser.parse_args()

os.makedirs(args.site, exist_ok=True)
for name in ("model.safetensors", "meta.json"):
    shutil.copyfile(f"{args.model}/{name}", f"{args.site}/{name}")

# On the CPU and at full size, so that the numbers do not depend on the machine.
model, tok, meta = checkpoint.load(args.model, "cpu")
model = quantize.expand(model)
state = model.state_dict()
weights = {name: state[f"{name}.weight"][row, :8].tolist() for name, row in [("embed", 1000), ("layers.0.attn.q_proj", 0), ("layers.7.mlp.down_proj", 383)]}

prompts = []
for prompt in args.prompts:
    ids = [tok.special_tokens[EOT]] + tok.encode_ordinary(prompt)
    with torch.no_grad():
        logits = model(torch.tensor([ids]))[0][0, -1]
    greedy = inference.generate(model, [ids], 20, temperature=0)[0]
    prompts.append({"prompt": prompt, "ids": ids, "logits": [round(v, 4) for v in logits.tolist()], "greedy": greedy})
    print(f"{prompt!r} -> {tok.decode(greedy)!r}")

with open(args.out, "w") as f:
    json.dump({"model": args.model, "quant": meta["quant"], "weights": weights, "prompts": prompts}, f)
print(f"wrote {args.out}, {os.path.getsize(args.out) / 1e3:.0f} kB, and {args.site}, {os.path.getsize(f'{args.site}/model.safetensors') / 1e6:.1f} MB")
