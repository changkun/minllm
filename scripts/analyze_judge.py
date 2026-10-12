"""
What the judges' probabilities say beyond their verdicts: how well they
rank right solutions above wrong ones, what a threshold other than one half
makes of them, and how the story judges behave when the stories are swapped.

    uv run scripts/analyze_judge.py      # reads runs/judge/*.json, writes runs/judge/summary.json

The threshold is chosen on half of the problems and measured on the other
half, so that it is not fitted to what it is measured on. See docs/07b-judge.md.
"""

import bisect
import glob
import json
import os


def auc(scores, truth):
    """The chance that a right solution gets a higher score than a wrong one."""
    negatives = sorted(s for s, t in zip(scores, truth) if not t)
    positives = [s for s, t in zip(scores, truth) if t]
    ranked = sum(bisect.bisect_left(negatives, s) + 0.5 * (bisect.bisect_right(negatives, s) - bisect.bisect_left(negatives, s)) for s in positives)
    return ranked / (len(positives) * len(negatives))


def accuracy(scores, truth, cut):
    return sum((s >= cut) == bool(t) for s, t in zip(scores, truth)) / len(scores)


summary = {}
for path in sorted(glob.glob("runs/judge/*.json")):
    name = os.path.basename(path)[:-5]
    if name == "summary":
        continue
    with open(path) as f:
        r = json.load(f)
    if r["task"] == "gsm8k":
        p, t = r["p_correct"], r["truth"]
        fit, test = (p[0::2], t[0::2]), (p[1::2], t[1::2])
        cut = max(sorted(set(fit[0])), key=lambda c: accuracy(*fit, c))
        summary[name] = {"auc": auc(p, t), "threshold": cut, "accuracy_at_half": accuracy(*test, 0.5), "accuracy_at_threshold": accuracy(*test, cut), "measured_on": len(test[0])}
    else:
        a, b = r["p_first_as_a"], r["p_first_as_b"]
        consistent = [(x >= 0.5) == (y >= 0.5) for x, y in zip(a, b)]
        agree = [(x >= 0.5) == bool(y) for x, y, c in zip(a, r["annotator"], consistent) if c]
        summary[name] = {"consistent_pairs": sum(consistent), "agrees_where_consistent": sum(agree) / len(agree) if agree else None, "mean_p_of_a": (sum(a) + sum(1 - y for y in b)) / (2 * len(a))}
    print(name, {k: round(v, 3) if isinstance(v, float) else v for k, v in summary[name].items()})
with open("runs/judge/summary.json", "w") as f:
    json.dump(summary, f, indent=1)
print("wrote runs/judge/summary.json")
