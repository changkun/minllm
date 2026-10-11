"""
Save one row of a weight matrix, for the figure of docs/10a-quantization.md
that shows what rounding to 16 values does to it.

    uv run scripts/dump_weights.py            # Qwen, writes runs/quant/row.json

The row is a typical one: of all rows of the matrix, the one whose range,
compared to the range of its groups of 64 weights, is the median.
"""

import argparse
import json
import os

from minllm import checkpoint

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="qwen")
parser.add_argument("--matrix", default="layers.0.attn.q_proj.weight")
parser.add_argument("--group", type=int, default=64)
parser.add_argument("--out", default="runs/quant/row.json")
args = parser.parse_args()

model, _, _ = checkpoint.load(args.model, "cpu")
w = model.state_dict()[args.matrix].float()
groups = w.reshape(w.shape[0], -1, args.group)
row_range = w.amax(1) - w.amin(1)
group_range = (groups.amax(-1) - groups.amin(-1)).mean(1)
ratio = row_range / group_range  # how much finer a group's steps are than the row's
row = ratio.argsort()[len(ratio) // 2].item()
print(f"{args.matrix} of {args.model}: {tuple(w.shape)}, row {row}, whose range is {ratio[row]:.2f} times that of its average group")

os.makedirs(os.path.dirname(args.out), exist_ok=True)
with open(args.out, "w") as f:
    json.dump({"model": args.model, "matrix": args.matrix, "row": row, "group": args.group, "ratio": ratio[row].item(), "weights": [round(v, 5) for v in w[row].tolist()]}, f)
print(f"wrote {args.out}")
