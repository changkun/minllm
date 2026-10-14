"""
Look at what the routers of a trained mixture of experts do with held-out
stories: how the work is spread over the experts, how much of the routing a
token's identity decides, and which tokens each expert gets.

    uv run scripts/analyze_moe.py                            # out/moe
    uv run scripts/analyze_moe.py --model out/moe-no-balance --out runs/moe/routing-no-balance.json

See docs/02a-moe.md.
"""

import argparse
import json
import os

import torch

from minllm.common import load_checkpoint, pick_device, result_path
from minllm.data import get_batch, load_tokens
from minllm.tokenizer import Tokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="out/moe")
parser.add_argument("--out", default="runs/moe/routing.json")
parser.add_argument("--data", default="data/tinystories/valid.bin")
parser.add_argument("--tokenizer", default="out/tokenizer.json")
parser.add_argument("--windows", type=int, default=400, help="windows of held-out text, half to fit on and half to score on")
parser.add_argument("--min-count", type=int, default=100, help="a token is listed for an expert only if it occurs this often")
args = parser.parse_args()
args.out = result_path(args.out)  # a path under runs/ goes to the results folder

device = pick_device()
tok = Tokenizer.load(args.tokenizer)
model, ckpt = load_checkpoint(f"{args.model}/ckpt.pt", device)
model.eval()
cfg, mixtures = model.cfg, model.mixtures()
E = cfg.n_experts

# Run the stories through the model and keep what goes into every mixture.
inputs = [[] for _ in mixtures]
for i, m in enumerate(mixtures):
    m.register_forward_pre_hook(lambda module, args, i=i: inputs[i].append(args[0].float().cpu()))
x, _ = get_batch(load_tokens(args.data), args.windows, cfg.max_seq_len, device, torch.Generator().manual_seed(0))
with torch.no_grad():
    for rows in x.split(50):
        model.hidden(rows)
tokens = x.cpu()  # (windows, seq)
half = args.windows // 2
fit = torch.arange(args.windows)[:, None].expand_as(tokens) < half  # the windows the per-token rule is fitted on

blocks = []
for i, m in enumerate(mixtures):
    with torch.no_grad():
        weights = m.cpu().route(torch.cat(inputs[i]).view(-1, cfg.dim)).view(*tokens.shape, E)
    first = weights.argmax(-1)  # the expert with the largest weight, per token
    firsts = torch.bincount(first.flatten(), minlength=E) / first.numel()

    # How much does the token alone decide? Give every token id the expert it
    # most often has first, on one half of the text, and count how often that
    # guess is right on the other half. Guessing the most common expert for
    # every token is the baseline.
    table = torch.zeros(cfg.vocab_size, E).index_put_((tokens[fit], first[fit]), torch.ones(()), accumulate=True)
    by_token = (table.argmax(-1)[tokens[~fit]] == first[~fit]).float().mean().item()
    # And how much does the neighbor say? Consecutive tokens with the same first expert.
    by_previous = (first[:, 1:] == first[:, :-1]).float().mean().item()

    # The tokens that an expert gets most reliably, among the frequent ones.
    table = torch.zeros(cfg.vocab_size, E).index_put_((tokens.flatten(), first.flatten()), torch.ones(()), accumulate=True)
    counts = table.sum(-1)
    rate = (table / counts.clamp(min=1)[:, None]).masked_fill(counts[:, None] < args.min_count, -1)
    favorites = []
    for e in range(E):
        best = rate[:, e].topk(12)
        favorites.append([{"token": tok.decode([t]), "rate": round(r, 3), "count": int(counts[t])} for r, t in zip(best.values.tolist(), best.indices.tolist()) if r > 0])

    blocks.append({
        "share": [round(v, 4) for v in m.share.tolist()],  # of all routed work, first and second choices alike
        "first": [round(v, 4) for v in firsts.tolist()],
        "top_weight": weights.max(-1).values.mean().item(),
        "by_token": by_token,
        "by_token_baseline": firsts.max().item(),
        "by_previous": by_previous,
        "by_previous_baseline": firsts.pow(2).sum().item(),
        "favorites": favorites,
    })
    print(f"block {i + 1}: busiest expert {max(blocks[-1]['share']):.1%}, idlest {min(blocks[-1]['share']):.1%} | first weight {blocks[-1]['top_weight']:.2f} | ", end="")
    print(f"first expert from the token alone {by_token:.1%} (baseline {firsts.max():.1%}) | same as the previous token {by_previous:.1%} (chance {firsts.pow(2).sum():.1%})")

os.makedirs(os.path.dirname(args.out), exist_ok=True)
with open(args.out, "w") as f:
    summary = {"model": args.model, "step": ckpt["step"], "tokens": tokens.numel(), "n_experts": E, "n_active": cfg.n_active}
    json.dump(summary | {"blocks": blocks}, f, indent=1, ensure_ascii=False)
print(f"wrote {args.out}")
