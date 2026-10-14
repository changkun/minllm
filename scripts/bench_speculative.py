"""
Measure speculative decoding: one reply at a time, by the model alone and
with a draft, at several draft lengths k.

    uv run scripts/bench_speculative.py --target data/qwen2.5-1.5b-instruct --draft qwen   # GSM8K problems
    uv run scripts/bench_speculative.py --target out/wide --draft out/base                 # the story models of 3a
    uv run scripts/bench_speculative.py --target out/wide --draft out/base --dtype float32 --repeats 1 --out runs/bench/speculative-wide-mps-float32.json

Every case runs in a fresh process, three times, and the fastest is kept:
the laptop is in use, and one measurement can be off by a factor of two.
Writes runs/bench/speculative-<target>.json with the speed of every case,
how many draft tokens were kept, and whether every reply is the one the
model writes alone. --dtype float32, or MINLLM_DEVICE=cpu, measures the
same in other arithmetic. See docs/04b-speculative.md.
"""

import argparse
import json
import os
import subprocess
import sys
import time

import torch

from minllm import chat, checkpoint, inference, pretrained, speculative, tasks
from minllm.common import pick_device
from minllm.data import EOT

parser = argparse.ArgumentParser()
parser.add_argument("--target", required=True)
parser.add_argument("--draft", required=True, help="the smaller model; prompt lookup is measured too")
parser.add_argument("--n", type=int, default=10, help="prompts, each answered once")
parser.add_argument("--max-new-tokens", type=int, default=256)
parser.add_argument("--ks", default="1,2,3,4,6,8")
parser.add_argument("--repeats", type=int, default=3)
parser.add_argument("--dtype", default=None, choices=["bfloat16", "float32"], help="of the weights and the arithmetic; bfloat16 on a GPU and float32 on the CPU unless given")
parser.add_argument("--one", default=None, help="run a single case, given as JSON, and print its result (used internally)")
parser.add_argument("--out", default=None)
args = parser.parse_args()


def dtype(device):
    return getattr(torch, args.dtype) if args.dtype else torch.bfloat16 if device != "cpu" else torch.float32


def load(spec, device):
    if os.path.exists(f"{spec}/config.json"):  # a published model, as downloaded
        return pretrained.load_model(spec, device, dtype(device)).eval(), pretrained.load_tokenizer(spec)
    model, tok, _ = checkpoint.load(spec, device, dtype(device))
    return model.eval(), tok


def prompts(tok, family):
    """GSM8K's first test problems in the chat format, or the openings of stories for a story model."""
    if family == "qwen":
        return [chat.encode(tok, e["messages"]) for e in tasks.gsm8k_examples("test", pretrained.SYSTEM_PROMPT)[: args.n]]
    stories = open("data/tinystories/valid.txt", encoding="utf-8").read().split(EOT)[1 : args.n + 1]
    return [tok.encode(EOT + s.strip().split(".")[0] + ".") for s in stories]


def one(case):
    """Generate every reply for one case and return the time and the tokens."""
    device = pick_device()
    model, tok = load(args.target, device)
    family = "qwen" if os.path.exists(f"{args.target}/config.json") or args.target == "qwen" else "small"
    stop = [tok.special_tokens[chat.IM_END]] if family == "qwen" else [tok.special_tokens[EOT]]
    texts = prompts(tok, family)
    small = load(args.draft, device)[0] if case["draft"] == "model" else None
    sync = torch.mps.synchronize if device == "mps" else (lambda: None)
    torch.manual_seed(0)

    def reply(prompt):
        if case["draft"] == "none":
            return inference.generate(model, [prompt], args.max_new_tokens, temperature=case["temperature"], stop=stop)[0], None
        max_len = len(prompt) + args.max_new_tokens + case["k"] + 1
        draft = speculative.ModelDraft(small, max_len) if case["draft"] == "model" else speculative.LookupDraft(model.cfg.vocab_size)
        return speculative.generate(model, draft, prompt, args.max_new_tokens, k=case["k"], temperature=case["temperature"], stop=stop)

    with torch.autocast(device_type=torch.device(device).type, dtype=torch.bfloat16, enabled=device != "cpu" and dtype(device) == torch.bfloat16):
        reply(texts[0][:32])  # warm the device up
        sync()
        start, outputs, totals = time.time(), [], {"passes": 0, "proposed": 0, "kept": 0}
        for prompt in texts:
            new, stats = reply(prompt)
            outputs.append(new)
            for key in totals:
                totals[key] += stats[key] if stats else 0
        sync()
    return {"seconds": time.time() - start, "tokens": sum(len(o) for o in outputs), "outputs": outputs} | totals


if args.one:
    print("RESULT " + json.dumps(one(json.loads(args.one))))
    sys.exit()

ks = [int(k) for k in args.ks.split(",")]
cases = [{"draft": "none", "k": 0, "temperature": 0.0}]
cases += [{"draft": "model", "k": k, "temperature": 0.0} for k in ks]
cases += [{"draft": "lookup", "k": k, "temperature": 0.0} for k in ks if k in (2, 4, 8)]
cases += [{"draft": d, "k": 4, "temperature": 1.0} for d in ("none", "model", "lookup")]
results = []
for case in cases:
    runs = []
    for _ in range(args.repeats):
        command = [sys.executable, __file__, "--target", args.target, "--draft", args.draft, "--n", str(args.n), "--max-new-tokens", str(args.max_new_tokens), "--one", json.dumps(case)]
        command += ["--dtype", args.dtype] if args.dtype else []
        out = subprocess.run(command, check=True, capture_output=True, text=True).stdout
        runs.append(json.loads(out.split("RESULT ", 1)[1]))
    best = min(runs, key=lambda r: r["seconds"])
    result = case | {k: best[k] for k in ("seconds", "tokens", "passes", "proposed", "kept")} | {"all_seconds": [r["seconds"] for r in runs]}
    result["tokens_per_second"] = best["tokens"] / best["seconds"]
    result["outputs"] = best["outputs"]
    results.append(result)
    kept = f", {best['kept'] / max(1, best['proposed']):.0%} of drafts kept, {best['tokens'] / max(1, best['passes']):.2f} tokens a pass" if best["passes"] else ""
    print(f"{case['draft']:6s} k={case['k']} T={case['temperature']:g}: {result['tokens_per_second']:6.1f} tokens/s{kept}", flush=True)

# Greedy speculation should write what the model writes alone. On a GPU a pass over several tokens need not
# give a pass over one token's numbers to the last bit, and that can part them at a near tie (docs/04b-speculative.md).
alone = results[0]["outputs"]
for r in results:
    if r["temperature"] == 0:
        r["same_as_alone"] = sum(a == b for a, b in zip(alone, r["outputs"]))
for r in results:
    r["outputs"] = None if r["temperature"] else r["outputs"]
name = os.path.basename(args.target.rstrip("/"))
out = args.out or f"runs/bench/speculative-{name}-{pick_device()}.json"
with open(out, "w") as f:
    json.dump({"target": args.target, "draft": args.draft, "n": args.n, "max_new_tokens": args.max_new_tokens, "device": pick_device(), "dtype": str(dtype(pick_device())).removeprefix("torch."), "cases": results}, f, indent=1)
print(f"wrote {out}")
