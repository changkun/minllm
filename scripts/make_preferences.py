"""
Collect preference data: two replies per prompt, and which one is preferred.

    uv run scripts/make_preferences.py                  # stories, compared by the simulated annotator
    uv run scripts/make_preferences.py --task gsm8k     # math replies, compared by the verifier

For stories the annotator is the simulated one of minllm/preferences.py,
and the comparisons go to data/preferences/train.jsonl and held_out.jsonl.
To give your own preferences instead, see scripts/label_preferences.py.

For GSM8K, the 0.5B model fine-tuned in stage 6 answers training problems
several times, and a right reply is preferred to a wrong one of the same
problem: data/preferences/gsm8k.jsonl. See docs/08b-dpo.md.
"""

import argparse
import json
import os
import random
import time

import torch

from minllm import chat, checkpoint, dpo, preferences, rl, tasks
from minllm.common import free_memory, pick_device, result_path
from minllm.pretrained import SYSTEM_PROMPT

parser = argparse.ArgumentParser()
parser.add_argument("--task", choices=["taste", "gsm8k"], default="taste")
parser.add_argument("--model", default=None, help="the model whose replies are compared")
parser.add_argument("--n", type=int, default=None, help="stories: comparisons; gsm8k: training problems")
parser.add_argument("--held-out", type=int, default=500, help="stories: how many comparisons to keep out of training")
parser.add_argument("--samples", type=int, default=8, help="gsm8k: replies per problem")
parser.add_argument("--per-prompt", type=int, default=2, help="gsm8k: pairs per problem, at most")
parser.add_argument("--max-new-tokens", type=int, default=256, help="gsm8k: the limit on a reply's length, stage 9's")
parser.add_argument("--out", default="data/preferences")
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

device = pick_device()
torch.manual_seed(args.seed)
rng = random.Random(args.seed)
args.model = args.model or ("out/sft-words" if args.task == "taste" else "out/sft-gsm8k-lora")
model, tok, _ = checkpoint.load(args.model, device)
os.makedirs(args.out, exist_ok=True)

if args.task == "gsm8k":
    # The training problems that stage 9 drew from: the last 200 stay held out.
    examples = rng.sample(tasks.gsm8k_examples("train", SYSTEM_PROMPT)[:-200], args.n or 480)
    end = tok.special_tokens[chat.IM_END]
    autocast = torch.autocast(device_type=torch.device(device).type, dtype=torch.bfloat16, enabled=device != "cpu")
    start, replies, texts = time.time(), [], []
    for i in range(0, len(examples), 8):
        with autocast:
            _, r, t = rl.rollout(model, tok, [e["messages"] for e in examples[i : i + 8]], args.samples, args.max_new_tokens)
        replies, texts = replies + r, texts + t
        free_memory(device)
        print(f"\r{len(texts)}/{len(examples) * args.samples} replies", end="", flush=True)
    # The verifier as the annotator: a reply is right if the model ended it and its answer is the answer.
    finished = [r[-1] == end for r in replies]
    right = [f and tasks.SCORE["gsm8k"](examples[i // args.samples], t) == 1 for i, (f, t) in enumerate(zip(finished, texts))]
    chosen = dpo.pairs(texts, right, args.samples, args.per_prompt)
    pairs = [{"messages": examples[a // args.samples]["messages"], "replies": [texts[a], texts[b]], "preferred": 0, "finished": [finished[a], finished[b]]} for a, b in chosen]
    preferences.save(f"{args.out}/gsm8k.jsonl", pairs)

    groups = [right[i : i + args.samples] for i in range(0, len(right), args.samples)]
    summary = {
        "model": args.model,
        "problems": len(examples),
        "replies": len(texts),
        "right": sum(right) / len(right),
        "finished": sum(finished) / len(finished),
        "problems_all_right": sum(all(g) for g in groups),
        "problems_none_right": sum(not any(g) for g in groups),
        "problems_with_both": sum(any(g) and not all(g) for g in groups),
        "pairs": len(pairs),
        "tokens_preferred": sum(len(replies[a]) for a, _ in chosen) / len(chosen),
        "tokens_other": sum(len(replies[b]) for _, b in chosen) / len(chosen),
        "other_finished": sum(finished[b] for _, b in chosen) / len(chosen),
        "minutes": (time.time() - start) / 60,
        "args": vars(args),
    }
    print(f"\r{len(texts)} replies to {len(examples)} problems, {summary['right']:.1%} right; {summary['problems_with_both']} problems with right and wrong replies give {len(pairs)} pairs, written to {args.out}/gsm8k.jsonl")
    with open(result_path("runs/dpo/pairs.json"), "w") as f:
        json.dump(summary, f, indent=1)
    raise SystemExit

# Prompts from training stories; the test prompts of the later evaluations stay unseen.
examples = list(tasks.words_examples("data/tinystories/train.txt", tasks.content_words(), args.n or 4000, seed=args.seed + 100))
pairs = []
for start in range(0, len(examples), 256):
    chunk = examples[start : start + 256]
    # Two samples per prompt, at temperature 1: the model's own distribution.
    _, _, texts = rl.rollout(model, tok, [e["messages"] for e in chunk], samples=2, max_new_tokens=220)
    for i, e in enumerate(chunk):
        a, b = texts[2 * i], texts[2 * i + 1]
        pairs.append({"messages": e["messages"], "replies": [a, b], "preferred": preferences.prefer(a, b, rng)})
    print(f"\r{len(pairs)}/{len(examples)}", end="", flush=True)

preferences.save(f"{args.out}/train.jsonl", pairs[: -args.held_out])
preferences.save(f"{args.out}/held_out.jsonl", pairs[-args.held_out :])

# How good is the annotator's signal? Its taste is known, so this can be said.
tastes = [[preferences.taste(r) for r in p["replies"]] for p in pairs]
consistent = sum((t[p["preferred"]] >= t[1 - p["preferred"]]) for t, p in zip(tastes, pairs)) / len(pairs)
mean = sum(sum(t) for t in tastes) / (2 * len(tastes))
print(f"\r{len(pairs)} comparisons written to {args.out}")
print(f"mean taste of a story: {mean:+.3f}; the annotator picked the story of higher taste in {consistent:.1%} of comparisons")
with open(result_path("runs/rlhf/preferences.json"), "w") as f:
    json.dump({"model": args.model, "comparisons": len(pairs), "held_out": args.held_out, "mean_taste": mean, "annotator_consistency": consistent}, f, indent=1)
