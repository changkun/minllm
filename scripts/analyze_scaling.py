"""
Fit the scaling laws to the runs of scripts/scaling.py.

    uv run scripts/analyze_scaling.py

Per compute budget, the valley of final loss against model size and its
floor; through the floors, how the best size and the best number of tokens
grow with compute; and one formula, L(N, D) = E + A / N^alpha + B / D^beta,
for every run. The formula is also fitted on the smaller budgets alone and
asked for the largest one, which it has not seen. Writes
runs/scaling/summary.json. See docs/03c-scaling.md.
"""

import glob
import json
import os
from collections import defaultdict

from minllm import scaling
from minllm.data import load_tokens
from minllm.common import result_path


def runs(pattern):
    out = []
    for path in sorted(glob.glob(pattern)):
        with open(path) as f:
            records = [json.loads(line) for line in f]
        meta, final = records[0]["run"], [r for r in records if "val_loss" in r][-1]
        out.append(meta | {"name": os.path.basename(path)[:-6], "loss": final["val_loss"], "minutes": final["elapsed"] / 60})
    return out


data_tokens = len(load_tokens("data/tinystories/train.bin"))
grid = runs(result_path("runs/scaling/c*.jsonl"))
for r in grid:
    r["passes"] = r["tokens"] / data_tokens  # over the training text: above 1, some of it is read again
    r["budget"] = float(r["name"].split("-")[0][1:])
    r["flops"] = scaling.flops(r["params"], r["tokens"])
sweep = runs(result_path("runs/scaling/lr-*.jsonl"))

budgets = defaultdict(list)
for r in grid:
    budgets[r["budget"]].append(r)
valleys = {}
for budget, rs in sorted(budgets.items()):
    rs.sort(key=lambda r: r["params"])
    best = min(rs, key=lambda r: r["loss"])
    # The floor is fitted on the lowest run and its neighbors; it is bracketed when runs lie on both sides of it.
    i = rs.index(best)
    near = rs[max(0, i - 1) : i + 2] if 0 < i < len(rs) - 1 else rs[:3] if i == 0 else rs[-3:]
    n_opt, l_opt = scaling.valley([r["params"] for r in near], [r["loss"] for r in near])
    valleys[budget] = {"runs": [{k: r[k] for k in ("name", "params", "tokens", "passes", "lr", "loss", "minutes")} for r in rs],
                       "params": n_opt, "tokens": budget / (6 * n_opt), "loss": l_opt, "bracketed": 0 < i < len(rs) - 1}
    print(f"{budget:.0e}: best size {n_opt / 1e6:.2f}M on {budget / (6 * n_opt) / 1e6:.0f}M tokens ({budget / 6 / n_opt / n_opt:.0f} per parameter), loss {l_opt:.3f}"
          + ("" if valleys[budget]["bracketed"] else "  (the lowest run is at an end: not bracketed)"))

bracketed = [b for b, v in sorted(valleys.items()) if v["bracketed"]]
laws = {}
if len(bracketed) >= 2:
    k_n, a = scaling.power_law(bracketed, [valleys[b]["params"] for b in bracketed])
    k_d, b_exp = scaling.power_law(bracketed, [valleys[b]["tokens"] for b in bracketed])
    laws = {"params": {"k": k_n, "a": a}, "tokens": {"k": k_d, "b": b_exp}, "budgets": bracketed}
    print(f"best size grows as C^{a:.2f}, best tokens as C^{b_exp:.2f}")

fit = scaling.loss_law([r["params"] for r in grid], [r["tokens"] for r in grid], [r["loss"] for r in grid])
print("L(N, D) = {E:.3f} + {A:.1f} / N^{alpha:.3f} + {B:.1f} / D^{beta:.3f}".format(**fit))
misses = {r["name"]: r["loss"] - scaling.law(fit, r["params"], r["tokens"]) for r in grid}
worst = max(misses, key=lambda name: abs(misses[name]))
print(f"largest miss of the fit on its own runs: {misses[worst]:+.3f}, {worst}")

# Fitted without the largest budget, then asked for it.
top = max(budgets)
seen = [r for r in grid if r["budget"] < top]
held = [r for r in grid if r["budget"] == top]
partial = scaling.loss_law([r["params"] for r in seen], [r["tokens"] for r in seen], [r["loss"] for r in seen])
predictions = [{"name": r["name"], "params": r["params"], "tokens": r["tokens"], "loss": r["loss"], "predicted": scaling.law(partial, r["params"], r["tokens"])} for r in held]
for p in predictions:
    print(f"{p['name']}: predicted {p['predicted']:.3f} from the smaller budgets, measured {p['loss']:.3f}")
split = {b: scaling.optimal_split(fit, b) for b in sorted(budgets)}

# Stage 3's run, width 384 for 6,000 steps of 64 windows of 256 tokens, against the split the law prefers at its compute.
base = {"params": next(r["params"] for r in grid if r["dim"] == 384), "tokens": 6000 * 64 * 256}
with open("out/base/log.jsonl") as f:  # stage 3's run, as train.py logged it
    base["loss"] = [json.loads(line) for line in f if "val_loss" in line][-1]["val_loss"]
base["flops"] = scaling.flops(base["params"], base["tokens"])
base["predicted"] = scaling.law(fit, base["params"], base["tokens"])
n_best, d_best = scaling.optimal_split(fit, base["flops"])
base["best"] = {"params": n_best, "tokens": d_best, "predicted": scaling.law(fit, n_best, d_best)}
print(f"stage 3: {base['flops']:.1e} FLOPs, predicted {base['predicted']:.3f}, measured {base['loss']:.3f};"
      f" the law prefers {n_best / 1e6:.1f}M on {d_best / 1e6:.0f}M tokens, predicted {base['best']['predicted']:.3f}")

summary = {"valleys": {f"{b:.0e}": v for b, v in valleys.items()}, "laws": laws, "fit": fit, "misses": misses, "fit_without_largest": partial, "predictions": predictions,
           "fit_split": {f"{b:.0e}": {"params": n, "tokens": d} for b, (n, d) in split.items()},
           "fit_split_exponent": fit["beta"] / (fit["alpha"] + fit["beta"]),  # the law's best size grows as C to this power
           "stage3": base, "hours": {"grid": sum(r["minutes"] for r in grid) / 60, "sweep": sum(r["minutes"] for r in sweep) / 60},
           "sweep": [{k: r[k] for k in ("name", "dim", "params", "lr", "loss")} for r in sweep]}
with open(result_path("runs/scaling/summary.json"), "w") as f:
    json.dump(summary, f, indent=1)
print("wrote", result_path("runs/scaling/summary.json"))
