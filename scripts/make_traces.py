"""
Have a reasoning model think through GSM8K's problems: the teacher,
DeepSeek-R1-Distill-Qwen-1.5B, samples one reply to each, thinking first.

    uv run scripts/make_traces.py --n 960               # training problems: the thinking that ended and was right
    uv run scripts/make_traces.py --split test --n 192  # the teacher's own score on the first test problems

On the training problems, a reply is kept if it closed its thinking,
ended within the limit and has the right answer, as DeepSeek-R1's
authors kept their teacher's: data/reasoning/traces.jsonl, for
`scripts/finetune.py --task gsm8k-traces`. Both write every reply to
data/reasoning/replies-<split>.jsonl as it is made, from which a stopped run
continues, and what the teacher did to runs/reasoning/teacher-<split>.json.
See docs/09c-reasoning.md.

The teacher is downloaded with `uv run python -c "from minllm import
pretrained; pretrained.download('data/r1-distill-qwen-1.5b',
'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B', files=['config.json',
'tokenizer.json', 'tokenizer_config.json', 'model.safetensors'])"`, 3.5 GB.
"""

import argparse
import json
import os
import statistics
import time

import torch

from minllm import inference, pretrained, reasoning, tasks
from minllm.common import pick_device, result_path

parser = argparse.ArgumentParser()
parser.add_argument("--teacher", default="data/r1-distill-qwen-1.5b")
parser.add_argument("--split", choices=["train", "test"], default="train")
parser.add_argument("--n", type=int, default=960)
parser.add_argument("--max-new-tokens", type=int, default=2048)
parser.add_argument("--temperature", type=float, default=0.6, help="as the teacher's authors advise, with top-p 0.95")
parser.add_argument("--batch-size", type=int, default=64)
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

device = pick_device()
torch.manual_seed(args.seed)
model = pretrained.load_model(args.teacher, device, torch.bfloat16)
tok = pretrained.load_tokenizer(args.teacher)
eos = tok.special_tokens[reasoning.R1_EOS]

problems = tasks.gsm8k_examples(args.split, "")
if args.split == "train":
    problems = problems[:-200]  # the problems that fine-tuning holds out stay unseen
indices = list(range(len(problems)))[: args.n]
question = lambda e: e["messages"][1]["content"].removesuffix(f"\n\n{tasks.GSM8K_INSTRUCTION}")
prompts = {i: reasoning.r1_prompt(tok, question(problems[i])) for i in indices}

# Every reply is written as its batch ends, and a run started again skips the problems replied to:
# a stopped run continues where it stopped. Delete the file to start over.
os.makedirs("data/reasoning", exist_ok=True)
replies = f"data/reasoning/replies-{args.split}.jsonl"
results = {}
if os.path.exists(replies):
    with open(replies, encoding="utf-8") as f:
        results = {r["index"]: r for r in map(json.loads, f) if r["index"] in prompts}
resumed = len(results)
start = time.time()
order = sorted((i for i in indices if i not in results), key=lambda i: len(prompts[i]))
with open(replies, "a", encoding="utf-8") as out:
    for b in range(0, len(order), args.batch_size):
        chunk = order[b : b + args.batch_size]
        steps = torch.stack(list(inference.stream(model, [prompts[i] for i in chunk], args.max_new_tokens, temperature=args.temperature, top_p=0.95, stop=[eos])), dim=1).tolist()
        for i, row in zip(chunk, steps):
            ended = -1 in row
            ids = row[: row.index(-1)] if ended else row
            text = tok.decode(ids)
            thinking, after = reasoning.split_thinking(text)
            right = reasoning.answer(after or text) == float(problems[i]["number"])
            results[i] = {"index": i, "tokens": len(ids), "ended": ended, "closed": reasoning.END_THINK in text, "right": right, "text": text}
            out.write(json.dumps(results[i], ensure_ascii=False) + "\n")
        out.flush()
        done = len(results)
        print(f"{done}/{len(indices)} | {sum(r['right'] for r in results.values()) / done:.1%} right | {(time.time() - start) / 60:.1f} minutes", flush=True)

minutes = (time.time() - start) / 60
rows = [results[i] for i in indices]
kept = [i for i in indices if results[i]["right"] and results[i]["ended"] and results[i]["closed"]]
summary = {
    "teacher": args.teacher, "split": args.split, "n": len(indices), "max_new_tokens": args.max_new_tokens, "temperature": args.temperature,
    "minutes": minutes, "resumed": resumed,  # the minutes of the last start only, if replies were made before it
    "right": sum(r["right"] for r in rows) / len(rows),
    "ended": sum(r["ended"] for r in rows) / len(rows),
    "closed": sum(r["closed"] for r in rows) / len(rows),
    "tokens_mean": statistics.mean(r["tokens"] for r in rows), "tokens_median": statistics.median(r["tokens"] for r in rows),
    "kept": len(kept),
    "tokens": [r["tokens"] for r in rows], "scores": [int(r["right"]) for r in rows],
    "samples": [{"number": problems[i]["number"]} | results[i] for i in indices[:: max(1, len(indices) // 12)]],
}
print(f"teacher on {len(rows)} {args.split} problems: {summary['right']:.1%} right, {summary['ended']:.1%} ended within {args.max_new_tokens} tokens, {summary['tokens_mean']:.0f} tokens on average; {len(kept)} kept; {minutes:.0f} minutes")
os.makedirs(result_path("runs/reasoning"), exist_ok=True)
with open(result_path(f"runs/reasoning/teacher-{args.split}.json"), "w") as f:
    json.dump(summary, f, indent=1, ensure_ascii=False)
if args.split == "train":
    with open("data/reasoning/traces.jsonl", "w", encoding="utf-8") as f:
        for i in kept:
            f.write(json.dumps({"index": i, "number": problems[i]["number"], "trace": results[i]["text"], "tokens": results[i]["tokens"]}, ensure_ascii=False) + "\n")
    print(f"wrote {len(kept)} traces to data/reasoning/traces.jsonl")
