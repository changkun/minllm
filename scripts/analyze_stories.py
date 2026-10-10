"""
Look past the score at what the policies of stage 8 write: do the stories
still use the requested words, still vary, and still read like stories?

    uv run scripts/analyze_stories.py

Reads the replies that scripts/evaluate.py saved in out/eval/ and writes
runs/rlhf/stories.json. See docs/08-rlhf.md.
"""

import argparse
import json
import os
from collections import Counter

import torch
import torch.nn.functional as F

from minllm import checkpoint, preferences, tasks
from minllm.common import pick_device

parser = argparse.ArgumentParser()
parser.add_argument("--models", default="sft-words,ppo,ppo-no-kl,dpo")
parser.add_argument("--judge", default="out/base", help="a language model that never saw any of this training")
parser.add_argument("--out", default="runs/rlhf/stories.json")
args = parser.parse_args()

device = pick_device()
judge, tok, _ = checkpoint.load(args.judge, device)
examples = tasks.test_examples("taste", 1000)
eot = tok.special_tokens["<|endoftext|>"]


@torch.no_grad()
def strangeness(stories):
    """
    The pretrained story model's loss on the stories, in nats per token: how
    surprising it finds them. It was trained on plain stories only, so text
    that has drifted away from reading like a story gets a higher loss.
    """
    total = count = 0
    for story in stories:
        ids = ([eot] + tok.encode_ordinary(story))[: judge.cfg.max_seq_len]
        if len(ids) > 1:
            x = torch.tensor([ids], device=device)
            logits = judge(x[:, :-1])[0]
            total += F.cross_entropy(logits[0], x[0, 1:], reduction="sum").item()
            count += len(ids) - 1
    return total / count


results = {}
print(f"{'model':12s} {'taste':>7s} {'words used':>11s} {'tokens':>7s} {'same opening':>13s} {'strangeness':>12s}")
for name in args.models.split(","):
    with open(f"out/eval/taste-{name}.jsonl", encoding="utf-8") as f:
        stories = [json.loads(line)["reply"] for line in f]
    openings = Counter(" ".join(story.split()[:6]) for story in stories)
    results[name] = {
        "taste": sum(preferences.taste(s) for s in stories) / len(stories),
        "words_used": sum(tasks.words_score(e, s) for e, s in zip(examples, stories)) / len(stories),
        "tokens": sum(len(tok.encode_ordinary(s)) for s in stories) / len(stories),
        # the share of stories that begin with the single most common six words
        "same_opening": openings.most_common(1)[0][1] / len(stories),
        "most_common_opening": openings.most_common(1)[0][0],
        "strangeness": strangeness(stories),
    }
    r = results[name]
    print(f"{name:12s} {r['taste']:+7.3f} {r['words_used']:11.1%} {r['tokens']:7.0f} {r['same_opening']:13.1%} {r['strangeness']:12.3f}")

os.makedirs(os.path.dirname(args.out), exist_ok=True)
with open(args.out, "w") as f:
    json.dump(results, f, indent=1)
print(f"wrote {args.out}")
