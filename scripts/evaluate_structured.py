"""
Ask the 0.5B model for its answer to GSM8K's test problems as a JSON
object, and measure what comes back: is it JSON, does it have the asked
form, and is the answer right?

    uv run scripts/evaluate_structured.py --form thinking                  # ask in the prompt, and hope
    uv run scripts/evaluate_structured.py --form thinking --constrain      # allow only tokens that keep the reply in the form
    uv run scripts/evaluate_structured.py --form answer --constrain        # no room to think: {"answer": 18}
    uv run scripts/evaluate_structured.py --form answer-first --constrain  # the answer before the reasoning
    uv run scripts/evaluate_structured.py --form reason-then-answer --constrain   # free text, then the answer as JSON on its last line

Writes runs/structured/<form>-<asked|constrained>.json. See docs/04a-structured-output.md.
"""

import argparse
import json
import os
import time

import torch

from minllm import checkpoint, evaluate, schema as schemas, tasks
from minllm.common import pick_device
from minllm.pretrained import SYSTEM_PROMPT

# The forms a reply can be asked to have: a JSON schema, and the same as a line to show the model.
FORMS = {
    "thinking": ({"type": "object", "properties": {"reasoning": {"type": "string"}, "answer": {"type": "number"}}}, '{"reasoning": "<your work>", "answer": <number>}'),
    "answer": ({"type": "object", "properties": {"answer": {"type": "number"}}}, '{"answer": <number>}'),
    "answer-first": ({"type": "object", "properties": {"answer": {"type": "number"}, "reasoning": {"type": "string"}}}, '{"answer": <number>, "reasoning": "<your work>"}'),
    "reason-then-answer": ({"type": "object", "properties": {"answer": {"type": "number"}}}, '{"answer": <number>}'),
}
# A form in which the JSON is only the last line, after free text. It is not a schema, so it is written as an expression.
PATTERNS = {"reason-then-answer": schemas.TEXT + r"\n" + schemas.regex(FORMS["reason-then-answer"][0])}

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="qwen")
parser.add_argument("--form", choices=list(FORMS), default="thinking")
parser.add_argument("--constrain", action="store_true", help="mask every token that would leave the form")
parser.add_argument("--n", type=int, default=None, help="number of test problems; all 1,319 by default")
parser.add_argument("--max-new-tokens", type=int, default=512)
parser.add_argument("--batch-size", type=int, default=64)
parser.add_argument("--out", default=None)
args = parser.parse_args()

schema, shown = FORMS[args.form]
device = pick_device()
model, tok, meta = checkpoint.load(args.model, device, torch.float32 if device == "cpu" else torch.bfloat16)
problems = tasks.gsm8k_examples("test", SYSTEM_PROMPT)[: args.n]
examples = []
for p in problems:
    question = p["messages"][-1]["content"].split("\n\n" + tasks.GSM8K_INSTRUCTION)[0]
    if args.form in PATTERNS:
        ask = f"{question}\n\nWork it out step by step. Then give the answer on the last line, as a JSON object in this form:\n{shown}"
    else:
        ask = f"{question}\n\nAnswer with a JSON object and nothing else, in this form:\n{shown}"
    examples.append({"messages": [p["messages"][0], {"role": "user", "content": ask}], "number": p["number"]})


def fits(value):
    """Whether a parsed reply has the asked form: exactly its keys, in any order, each of its type."""
    wanted = schema["properties"]
    if not isinstance(value, dict) or set(value) != set(wanted):
        return False
    types = {"string": str, "number": (int, float)}
    return all(isinstance(value[key], types[rule["type"]]) and not isinstance(value[key], bool) for key, rule in wanted.items())


def read(reply):
    """The reply as JSON: (value, how). how is "json", "fenced" if it only parses without a ``` fence around it, or None."""
    text = reply.strip()
    if args.form in PATTERNS:  # only the last line is JSON
        text = text.splitlines()[-1] if text else text
    for how, candidate in (("json", text), ("fenced", text.removeprefix("```json").removeprefix("```").removesuffix("```").strip())):
        try:
            return json.loads(candidate), how
        except ValueError:
            continue
    return None, None


start = time.time()
progress = lambda done, total: print(f"\r{done}/{total}", end="", flush=True)
timing = {}
if args.constrain:
    from minllm import constrain

    replies = constrain.replies(model, tok, examples, schema, args.max_new_tokens, args.batch_size, progress=progress, timing=timing, pattern=PATTERNS.get(args.form))
else:
    replies = evaluate.replies(model, tok, examples, args.max_new_tokens, args.batch_size, progress=progress)
seconds = time.time() - start

rows = []
for example, reply in zip(examples, replies):
    value, how = read(reply)
    in_form = how is not None and fits(value)
    right = in_form and float(value["answer"]) == float(example["number"])
    rows.append({"json": how == "json", "fenced": how == "fenced", "in_form": in_form, "right": right, "tokens": len(tok.encode_ordinary(reply))})
n = len(rows)
share = lambda key: sum(r[key] for r in rows) / n
mean, error = evaluate.mean_and_error([float(r["right"]) for r in rows])
name = f"{args.form}-{'constrained' if args.constrain else 'asked'}"
print(f"\r{name}: JSON {share('json'):.1%}, JSON in a fence {share('fenced'):.1%}, in the asked form {share('in_form'):.1%}, right {mean:.1%} ± {error:.1%}, {n} problems, {seconds:.0f}s")

out = args.out or f"runs/structured/{name}.json"
os.makedirs(os.path.dirname(out), exist_ok=True)
result = {"model": args.model, "form": args.form, "constrained": args.constrain, "schema": schema, "n": n, "max_new_tokens": args.max_new_tokens, "seconds": seconds, "timing": timing, "reply_tokens": sum(r["tokens"] for r in rows)}
result |= {"json": share("json"), "fenced": share("fenced"), "in_form": share("in_form"), "mean": mean, "error": error, "cut_off": sum(r["tokens"] >= args.max_new_tokens - 2 for r in rows)}
result |= {"scores": [float(r["right"]) for r in rows], "rows": [[int(r["json"]), int(r["fenced"]), int(r["in_form"]), r["tokens"]] for r in rows], "samples": [{"reply": reply, "number": e["number"]} for e, reply in list(zip(examples, replies))[:: max(1, n // 30)]]}
with open(out, "w") as f:
    json.dump(result, f, indent=1)
print(f"wrote {out}")
