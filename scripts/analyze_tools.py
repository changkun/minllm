"""
Compare the runs of docs/06a-tool-calling.md and docs/09b-tool-rl.md
problem by problem: the same model with its calls run and with them
failing, the model fine-tuned with the tool against the one fine-tuned on
the same problems without it, the published model with and without the
tool in its prompt, its tool instructions with the call markers spelled
out and as the single tokens Qwen's tokenizer makes of them, and GRPO
with the tool against its start and against stage 9's GRPO without it.

    uv run scripts/analyze_tools.py      # reads runs/eval/ and out/eval/, writes runs/eval/tools-summary.json

It also counts, from all conversations that scripts/evaluate_tools.py
saved in out/eval/, how often the model's turn after a result begins with
that result: whether it reads what the tool says. And of the problems that
each GRPO run turned from wrong to right and from right to wrong, how many
of the replies before stage 9's GRPO had a calculation wrong: whether what
stage 9 gained was arithmetic, which a calculator would have done.

See docs/06a-tool-calling.md and docs/09b-tool-rl.md.
"""

import json

from minllm import evaluate, pretrained, tools
from minllm.pretrained import SYSTEM_PROMPT


def load(name):
    with open(f"runs/eval/{name}.json") as f:
        return json.load(f)


pairs = {
    "with the tool, against without it (stage 6)": ("gsm8k-sft-gsm8k-lora", "gsm8k-tools-sft-gsm8k-tools-lora"),
    "calls failing, against calls run": ("gsm8k-tools-sft-gsm8k-tools-lora", "gsm8k-tools-sft-gsm8k-tools-lora-no-execute"),
    "published, told about the tool, against not": ("gsm8k-qwen", "gsm8k-tools-qwen"),
    "published, told about the tool with the markers as Qwen's tokenizer writes them, against not": ("gsm8k-qwen", "gsm8k-tools-qwen-system-markers"),
    "published, the markers as Qwen's tokenizer writes them, against spelled out": ("gsm8k-tools-qwen", "gsm8k-tools-qwen-system-markers"),
    "GRPO without the tool, against its start (stage 6)": ("gsm8k-sft-gsm8k-lora", "gsm8k-grpo-gsm8k"),
    "GRPO with the tool, against its start (6a)": ("gsm8k-tools-sft-gsm8k-tools-lora", "gsm8k-tools-grpo-gsm8k-tools"),
    "GRPO with the tool, against GRPO without it (stage 9)": ("gsm8k-grpo-gsm8k", "gsm8k-tools-grpo-gsm8k-tools"),
    "after GRPO, calls failing, against calls run": ("gsm8k-tools-grpo-gsm8k-tools", "gsm8k-tools-grpo-gsm8k-tools-no-execute"),
}
TOOLS = ["gsm8k-tools-sft-gsm8k-tools-lora", "gsm8k-tools-sft-gsm8k-tools-lora-no-execute", "gsm8k-tools-grpo-gsm8k-tools", "gsm8k-tools-grpo-gsm8k-tools-no-execute"]
summary = {"comparisons": {}, "runs": {}}
for label, (a, b) in pairs.items():
    difference, low, high = evaluate.paired_difference(load(a)["scores"], load(b)["scores"])
    summary["comparisons"][label] = {"from": a, "to": b, "difference": difference, "low": low, "high": high}
    print(f"{label}: {difference:+.1%}, 95% interval {low:+.1%} to {high:+.1%}")
for name in TOOLS + ["gsm8k-tools-qwen", "gsm8k-tools-qwen-system-markers"]:
    r = load(name)
    n = r["n"]
    written = sum(w[1] for w in r["written"])
    summary["runs"][name] = {
        "mean": r["mean"], "error": r["error"], "n": n,
        "with_a_call": sum(c > 0 for c in r["calls"]) / n, "calls_per_problem": sum(r["calls"]) / n, "malformed": sum(r["bad_calls"]),
        "written_calculations": written, "wrong_calculations": sum(w[1] - w[0] for w in r["written"]), "seconds": r["seconds"],
    }
    if "system_tokens" in r:
        summary["runs"][name]["system_tokens"] = r["system_tokens"]
    print(name, {k: round(v, 3) if isinstance(v, float) else v for k, v in summary["runs"][name].items()})
# Does the model go on from the tool's result?
for name in TOOLS:
    follows = results = 0
    with open(f"out/eval/{name}.jsonl", encoding="utf-8") as f:
        for line in f:
            conversation = json.loads(line)["conversation"]
            for k, m in enumerate(conversation[:-1]):
                if m["role"] == "user" and m["content"].startswith("<tool_response>"):
                    results += 1
                    follows += conversation[k + 1]["content"].strip().startswith(m["content"].split("\n")[1])
    summary["runs"][name] |= {"results": results, "next_turn_begins_with_it": follows}
    print(f"{name}: the next turn begins with the result after {follows} of {results}")
# What each GRPO run changed, problem by problem, and whether stage 9's gains were arithmetic.
for label, (a, b) in [("stage 9", ("gsm8k-sft-gsm8k-lora", "gsm8k-grpo-gsm8k")), ("with the tool", ("gsm8k-tools-sft-gsm8k-tools-lora", "gsm8k-tools-grpo-gsm8k-tools"))]:
    before, after = load(a)["scores"], load(b)["scores"]
    gained = [i for i, (x, y) in enumerate(zip(before, after)) if not x and y]
    lost = [i for i, (x, y) in enumerate(zip(before, after)) if x and not y]
    summary["grpo"] = summary.get("grpo", {}) | {label: {"gained": len(gained), "lost": len(lost)}}
    if label == "stage 9":
        with open(f"out/eval/{a}.jsonl", encoding="utf-8") as f:
            slipped = [any(not right for *_, right in tools.written_calculations(json.loads(line)["reply"])) for line in f]
        summary["grpo"][label] |= {"gained_with_a_wrong_calculation_before": sum(slipped[i] for i in gained), "wrong_calculation_before": sum(slipped)}
    print(f"GRPO {label}: {len(gained)} problems turned right, {len(lost)} turned wrong", summary["grpo"][label])
# What the tool's description adds to every prompt, in tokens: the system prompt without it, and with it, the markers spelled out or as Qwen's tokenizer writes them.
tok = pretrained.load_tokenizer()
turn = lambda content, marked: len(tools.encode(tok, [{"role": "system", "content": content}], add_generation_prompt=False, system_markers=marked)[0])
prompt = lambda content, marked: turn(content, marked) - turn("", marked)  # without the chat format around it
listed = tools.system_prompt(SYSTEM_PROMPT, [tools.CALCULATE])
summary["system_tokens"] = {"plain": prompt(SYSTEM_PROMPT, False), "with_the_tool": prompt(listed, False), "with_the_tool_markers_as_tokens": prompt(listed, True)}
print("tokens of the system's turn:", summary["system_tokens"])
with open("runs/eval/tools-summary.json", "w") as f:
    json.dump(summary, f, indent=1)
print("wrote runs/eval/tools-summary.json")
