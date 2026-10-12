"""
Evaluate a model on the multiple-choice questions of MMLU, by the
probabilities it gives or, to compare, by what it writes.

    uv run scripts/evaluate_choice.py --model qwen                       # the letter after "Answer:", five solved examples first
    uv run scripts/evaluate_choice.py --model qwen --shots 0 --format chat
    uv run scripts/evaluate_choice.py --model qwen --shots 0 --format chat --rotate 1   # the options moved round by one place
    uv run scripts/evaluate_choice.py --model qwen --method cloze        # no options shown: each option's text as the continuation
    uv run scripts/evaluate_choice.py --model qwen --method generate --shots 0 --per-subject 20    # let it write, and read the letter off its reply

Writes runs/eval/mmlu-<model>-<how>.json with every question's verdict, which
scripts/evaluate.py --compare can set against another. See
docs/07a-multiple-choice.md.
"""

import argparse
import json
import os
import time
from collections import Counter

import torch

from minllm import checkpoint, choice, evaluate, quantize, tasks
from minllm.common import pick_device
from minllm.pretrained import SYSTEM_PROMPT

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True, help="qwen, or the directory of a trained model")
parser.add_argument("--method", choices=["letter", "cloze", "generate"], default="letter")
parser.add_argument("--format", choices=["plain", "chat"], default="plain", help="the prompt as text to continue, or as a user's turn")
parser.add_argument("--shots", type=int, default=5, help="solved examples of the same subject shown first")
parser.add_argument("--rotate", type=int, default=0, help="move every question's options round by this many places")
parser.add_argument("--per-subject", type=int, default=None, help="only the first questions of every subject; all by default")
parser.add_argument("--max-tokens", type=int, default=4096, help="a prompt longer than this is shown fewer solved examples")
parser.add_argument("--max-new-tokens", type=int, default=96, help="the room for a written reply")
parser.add_argument("--budget", type=int, default=24_000, help="tokens in a batch, padding included")
parser.add_argument("--bits", type=int, default=0, help="round the weights to this many bits first (docs/10a-quantization.md)")
parser.add_argument("--group", type=int, default=64)
parser.add_argument("--out", default=None)
args = parser.parse_args()

device = pick_device()
model, tok, meta = checkpoint.load(args.model, device, torch.float32 if device == "cpu" else torch.bfloat16)
if meta.get("quant"):
    model = quantize.expand(model)
if args.bits:
    model = quantize.rounded(model.float(), args.bits, args.group)
system = SYSTEM_PROMPT if meta["family"] == "qwen" else ""
limit = min(args.max_tokens, model.cfg.max_seq_len)

examples = tasks.mmlu_examples("test")
subjects = sorted({e["subject"] for e in examples})
if args.per_subject:
    examples = [e for s in subjects for e in [x for x in examples if x["subject"] == s][: args.per_subject]]
examples = [choice.rotated(e, args.rotate) for e in examples]
solved = {s: [e for e in tasks.mmlu_examples("dev") if e["subject"] == s] for s in subjects}
answers = [e["answer"] for e in examples]
progress = lambda done, total: print(f"\r{done}/{total}", end="", flush=True)
result = {"model": args.model, "task": "mmlu", "method": args.method, "n": len(examples), "rotate": args.rotate}
start = time.time()


def prompt(example, shots):
    """The tokens of a question's prompt, with as many of the solved examples as fit."""
    for k in range(shots, -1, -1):
        ids = choice.encode(tok, choice.lettered(example, solved[example["subject"]][:k]), args.format, system)
        if len(ids) <= limit:
            return ids, k
    return ids[-limit:], 0  # a model with a short context reads the end of the question


if args.method == "letter":
    prompts, shown = zip(*(prompt(e, args.shots) for e in examples))
    logp, likeliest = choice.next_token_logprobs(model, list(prompts), choice.letter_ids(tok, args.format), args.budget, progress)
    chosen = logp.argmax(-1).tolist()
    # what the model would say first if it were let to: the likeliest token of all, by how many questions it leads
    first = [[tok.decode([token]), count] for token, count in Counter(likeliest.tolist()).most_common(8)]
    result |= {"format": args.format, "shots": args.shots, "fewer_shots": sum(k < args.shots for k in shown), "tokens": sum(map(len, prompts)), "first_tokens": first, "logprobs": [[round(v, 3) for v in row] for row in logp.tolist()]}
    how = f"letter-{args.format}-{args.shots}shot"
elif args.method == "cloze":
    # Four continuations per question, and each option once more after "Answer:" alone,
    # which says how likely its text is whatever the question.
    contexts = [choice.encode(tok, choice.cloze(e), "plain")[-(limit - 64) :] for e in examples]
    options = [[tok.encode_ordinary(f" {o}".rstrip())[:64] for o in e["options"]] for e in examples]
    flat = [(c, o) for c, four in zip(contexts, options) for o in four]
    bare = tok.encode_ordinary("Answer:")
    logp = choice.continuation_logprobs(model, [c for c, _ in flat], [o for _, o in flat], args.budget, progress).view(-1, 4)
    alone = choice.continuation_logprobs(model, [bare] * len(flat), [o for _, o in flat], args.budget).view(-1, 4)
    lengths = torch.tensor([[len(o) for o in four] for four in options])
    rules = {"sum": logp, "per_token": logp / lengths, "minus_unconditional": logp - alone}
    truth = torch.tensor(answers)
    result |= {"format": "plain", "shots": 0, "tokens": sum(len(c) + len(o) for c, o in flat)}
    result["rules"] = {}
    for name, scores in rules.items():
        mean, error = evaluate.mean_and_error((scores.argmax(-1) == truth).float().tolist())
        result["rules"][name] = {"mean": mean, "error": error, "choice": scores.argmax(-1).tolist()}
        print(f"\r  by {name}: {mean:.1%} ± {error:.1%}")
    chosen = rules["sum"].argmax(-1).tolist()
    result |= {"logprobs": [[round(v, 2) for v in row] for row in logp.tolist()], "unconditional": [[round(v, 2) for v in row] for row in alone.tolist()], "lengths": lengths.tolist()}
    how = "cloze"
else:
    # What stage 7 does: the model writes a reply, and a rule reads the answer off it.
    messages = [{"messages": ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": choice.lettered(e, solved[e["subject"]][: args.shots])}]} for e in examples]
    replies = evaluate.replies(model, tok, messages, max_new_tokens=args.max_new_tokens, batch_size=16, progress=progress)
    os.makedirs("out/eval", exist_ok=True)
    with open("out/eval/mmlu-replies.jsonl", "w", encoding="utf-8") as f:  # every reply, for a closer look
        for e, r in zip(examples, replies):
            f.write(json.dumps({"subject": e["subject"], "answer": choice.LETTERS[e["answer"]], "reply": r}, ensure_ascii=False) + "\n")
    read = [choice.read_letter(r) for r in replies]
    chosen = [-1 if r is None else r for r in read]
    result |= {"format": "chat", "shots": args.shots, "max_new_tokens": args.max_new_tokens, "unreadable": sum(r is None for r in read), "samples": [{"reply": r, "read": c} for r, c in list(zip(replies, chosen))[:: max(1, len(replies) // 40)]]}
    how = f"generate-chat-{args.shots}shot"

seconds = time.time() - start
scores = [float(c == a) for c, a in zip(chosen, answers)]
mean, error = evaluate.mean_and_error(scores)
print(f"\r{args.model} on MMLU by {how}: {mean:.1%} ± {error:.1%} over {len(scores)} questions, {seconds:.0f}s")

name = os.path.basename(args.model.rstrip("/")) + (f"-{args.bits}bit" if args.bits else "")
out = args.out or f"runs/eval/mmlu-{name}-{how}{f'-rot{args.rotate}' if args.rotate else ''}.json"
os.makedirs(os.path.dirname(out), exist_ok=True)
if args.bits:
    result |= {"bits": args.bits, "group": args.group}
result |= {"mean": mean, "error": error, "seconds": seconds, "subjects": subjects, "subject": [subjects.index(e["subject"]) for e in examples], "answer": answers, "choice": chosen, "scores": scores}
# a few questions in full, one from every fifth subject, to be looked at beside their numbers
result["shown"] = {str(i): {"question": e["question"], "options": e["options"]} for i, e in enumerate(examples) if (i == 0 or examples[i - 1]["subject"] != e["subject"]) and subjects.index(e["subject"]) % 5 == 0}
with open(out, "w") as f:
    json.dump(result, f, separators=(",", ":"))
print(f"wrote {out}, {os.path.getsize(out) / 1e3:.0f} kB")
