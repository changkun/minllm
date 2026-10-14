"""
Tool calling: a model that hands part of its work to a program.

A model calls a tool by writing the call as text, in a form it was trained
on. Whoever runs the model reads the call, runs the program, and writes its
result into the conversation, and the model goes on from there. Nothing in
the model changes. What makes it work is training data in which calls and
their results appear where they help.

The form here is Qwen2.5's, so that the published model can be compared
with the fine-tuned one. The tools are listed in the system prompt, a call
is a JSON object between the control tokens <tool_call> and </tool_call>,
and a result comes back in a user's turn between <tool_response> and
</tool_response>:

    <|im_start|>assistant
    Natalia sold 48/2 =
    <tool_call>
    {"name": "calculate", "arguments": {"expression": "48/2"}}
    </tool_call><|im_end|>
    <|im_start|>user
    <tool_response>
    24
    </tool_response><|im_end|>
    <|im_start|>assistant
    24 clips in May. ...

The tool here is a calculator, and the task GSM8K, whose solutions mark
every calculation for one: "48/2 = <<48/2=24>>24". See docs/06a-tool-calling.md.
"""

import ast
import json
import operator
import re

import torch

from minllm import chat, inference, tasks
from minllm.common import fetch, free_memory

CALL_START, CALL_END = "<tool_call>", "</tool_call>"

CALCULATE = {
    "type": "function",
    "function": {
        "name": "calculate",
        "description": "Evaluate an arithmetic expression and return its value.",
        "parameters": {
            "type": "object",
            "properties": {"expression": {"type": "string", "description": "An expression of numbers, + - * / and parentheses, such as (48 + 24) / 2."}},
            "required": ["expression"],
        },
    },
}

# ----------------------------------------------------------------- the tool

OPERATORS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod, ast.FloorDiv: operator.floordiv}


def calculate(expression):
    """
    The value of an arithmetic expression, as text. Only numbers, the
    operators and parentheses are read. Anything else is an error, returned
    as text as well, since the model is to read it and go on.
    """

    def value(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return node.value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            return -value(node.operand) if isinstance(node.op, ast.USub) else value(node.operand)
        if isinstance(node, ast.BinOp) and type(node.op) in OPERATORS:
            left, right = value(node.left), value(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 64:
                raise ValueError("exponent too large")
            return OPERATORS[type(node.op)](left, right)
        raise ValueError("only numbers, + - * / ** % and parentheses")

    text = expression.replace(",", "").replace("x", "*").replace("×", "*").replace("÷", "/").replace("$", "").strip()
    try:
        result = value(ast.parse(text, mode="eval").body)
    except (SyntaxError, ValueError, ZeroDivisionError, OverflowError, RecursionError) as error:
        return f"error: {error}"
    if isinstance(result, float):
        result = round(result, 6)
        if result.is_integer():
            result = int(result)
    return str(result)


# --------------------------------------------------------------- the format


def system_prompt(system, tools):
    """The system's turn with the tools listed, word for word as Qwen2.5's chat template writes it."""
    listed = "".join("\n" + json.dumps(tool, ensure_ascii=False) for tool in tools)
    return (
        f"{system}\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
        f"You are provided with function signatures within <tools></tools> XML tags:\n<tools>{listed}\n</tools>\n\n"
        "For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n"
        '<tool_call>\n{"name": <function-name>, "arguments": <args-json-object>}\n</tool_call>'
    )


def call_text(name, arguments):
    return f'{CALL_START}\n{json.dumps({"name": name, "arguments": arguments}, ensure_ascii=False)}\n{CALL_END}'


def response_text(result):
    return f"<tool_response>\n{result}\n</tool_response>"


def encode(tok, messages, add_generation_prompt=True, system_markers=False):
    """
    Like chat.encode, but in an assistant's turn <tool_call> and </tool_call>
    become the control tokens they are in the published vocabulary. In
    anyone else's turn they stay characters, unless system_markers is set:
    then they are control tokens in the system's turn too, where the tool
    instructions show them, as Qwen's own tokenizer writes them there.
    """
    start, end = tok.special_tokens[chat.IM_START], tok.special_tokens[chat.IM_END]
    marked = {"assistant", "system"} if system_markers else {"assistant"}
    ids, labels = [], []
    for m in messages:
        header = [start] + tok.encode_ordinary(f"{m['role']}\n")
        content = (_with_calls(tok, m["content"]) if m["role"] in marked else tok.encode_ordinary(m["content"])) + [end]
        newline = tok.encode_ordinary("\n")
        ids += header + content + newline
        labels += [-100] * len(header) + (content if m["role"] == "assistant" else [-100] * len(content)) + [-100] * len(newline)
    if add_generation_prompt:
        ids += [start] + tok.encode_ordinary("assistant\n")
    return ids, labels


def _with_calls(tok, text):
    ids = []
    for part in re.split(f"({re.escape(CALL_START)}|{re.escape(CALL_END)})", text):
        ids += [tok.special_tokens[part]] if part in (CALL_START, CALL_END) else tok.encode_ordinary(part)
    return ids


def training_example(tok, messages):
    """(ids, labels) as chat.training_example makes them: only the assistant's turns are learned, its calls included."""
    return encode(tok, messages, add_generation_prompt=False)


def calls(text):
    """The calls in a reply, as (name, arguments), or (None, the text) for one that is not a JSON call."""
    found = []
    for body in re.findall(f"{re.escape(CALL_START)}(.*?){re.escape(CALL_END)}", text, re.S):
        try:
            call = json.loads(body)
            found.append((call["name"], call["arguments"]))
        except (ValueError, KeyError, TypeError):
            found.append((None, body))
    return found


# ------------------------------------------------------------------ the data


def gsm8k_conversations(split, system, dir="data/gsm8k"):
    """
    GSM8K's problems, each with its solution as a conversation that calls
    the calculator wherever the solution marks a calculation. A mark reads
    "<<48/2=24>>24": the text up to it is the assistant's, the expression
    goes into a call, and the result comes back from the tool. The solution
    goes on after it in the assistant's next turn.
    """
    path = f"{dir}/{split}.jsonl"
    fetch(tasks.GSM8K_URL.format(split=split), path)
    examples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            # the problem as stage 6 asks it, so that the two fine-tunings differ in the tool alone
            question = f"{row['question']}\n\n{tasks.GSM8K_INSTRUCTION}"
            messages = [{"role": "system", "content": system_prompt(system, [CALCULATE])}, {"role": "user", "content": question}]
            parts = re.split(r"<<([^=<>]*)=([^<>]*)>>", row["answer"])  # text, expression, result, text, ...
            text, disagree = parts[0], 0
            for i in range(1, len(parts), 3):
                expression, result, after = parts[i], parts[i + 1], parts[i + 2]
                messages.append({"role": "assistant", "content": f"{text.rstrip()}\n{call_text('calculate', {'expression': expression})}".lstrip()})
                value = calculate(expression)
                messages.append({"role": "user", "content": response_text(value)})
                disagree += not _same(value, result)
                text = after if after.startswith(result) else result + after  # the solution writes the result after the mark
            messages.append({"role": "assistant", "content": text.strip()})
            number = row["answer"].split("####")[-1].strip().replace(",", "")
            examples.append({"messages": messages, "number": number, "marks": (len(parts) - 1) // 3, "disagree": disagree})
    return examples


def _same(value, marked):
    """Whether the calculator's value is the result the solution marked, to the marked precision."""
    try:
        number = marked.replace(",", "").strip()
        decimals = len(number.split(".")[1]) if "." in number else 0
        return abs(float(value) - float(number)) <= 0.5 * 10**-decimals + 1e-9
    except ValueError:
        return False


# ------------------------------------------------------------------- the loop


@torch.no_grad()
def run(model, tok, conversations, max_new_tokens=256, max_calls=16, batch_size=32, execute=True, progress=None, system_markers=False):
    """
    Let the model answer every conversation, running its calls. A round
    generates the next assistant turn of every conversation that is still
    open. A turn with calls gets the results in a user's turn, and the
    conversation goes on to the next round. A turn without calls is the
    answer. With execute=False no call is run, and the tool answers each
    with an error, which measures what the calls are worth. system_markers
    is passed to encode.

    Returns the conversations, each with what was added, and per
    conversation the number of calls and of calls that were not JSON.
    """
    stop = [tok.special_tokens[chat.IM_END]]
    open_ = [[dict(m) for m in c] for c in conversations]
    counts = [{"calls": 0, "bad_calls": 0, "turns": 0} for _ in conversations]
    active = list(range(len(conversations)))
    device_type = next(model.parameters()).device.type
    while active:
        order = sorted(active, key=lambda i: len(encode(tok, open_[i], system_markers=system_markers)[0]))
        still = []
        for start in range(0, len(order), batch_size):
            chunk = order[start : start + batch_size]
            prompts = [encode(tok, open_[i], system_markers=system_markers)[0] for i in chunk]
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=device_type != "cpu"):
                outputs = inference.generate(model, prompts, max_new_tokens, temperature=0.0, stop=stop)
            for i, ids in zip(chunk, outputs):
                text = tok.decode(ids)
                open_[i].append({"role": "assistant", "content": text})
                counts[i]["turns"] += 1
                made = calls(text)
                if not made or counts[i]["calls"] >= max_calls:
                    continue
                open_[i].append({"role": "user", "content": _results(made, counts[i], execute)})
                still.append(i)
        if progress:
            progress(len(conversations) - len(still), len(conversations))
        active = still
    return open_, counts


def _results(made, counts, execute):
    """The tool's turn for a list of calls: each result, or the error the model gets instead."""
    results = []
    for name, arguments in made:
        counts["calls"] += 1
        if name is None:
            counts["bad_calls"] += 1
            results.append("error: a call is a JSON object with a name and arguments")
        elif name != "calculate" or not isinstance(arguments, dict) or not isinstance(arguments.get("expression"), str):
            counts["bad_calls"] += 1
            results.append("error: the only tool is calculate, with an expression")
        else:
            results.append(calculate(arguments["expression"]) if execute else "error: the tool is not available")
    return "\n".join(response_text(r) for r in results)


@torch.no_grad()
def rollout(model, tok, conversations, samples=1, max_new_tokens=256, max_calls=16, temperature=1.0, execute=True, batch_size=32):
    """
    Like run, but sampled, several times for every conversation, and kept
    as the token ids that the model chose, for reinforcement learning
    (docs/09b-tool-rl.md). run writes each turn into the conversation as
    text and encodes the conversation again, and text encoded again need
    not give back the ids that were sampled; the policy gradient has to be
    taken on those.

    Returns, per episode: the prompt's ids; the ids that follow it, in
    which the model's turns alternate with the tool's; for each of them 1
    if the model wrote it and 0 if the tool or the chat format did; the
    conversation as messages; and its counts, as run gives them. Only the
    tokens marked 1 are trained on. Each round generates batch_size
    conversations at a time, of similar length: a conversation with calls
    grows to over a thousand tokens.
    """
    start, end = tok.special_tokens[chat.IM_START], tok.special_tokens[chat.IM_END]
    newline = tok.encode_ordinary("\n")
    prompts = [encode(tok, c)[0] for c in conversations for _ in range(samples)]
    messages = [[dict(m) for m in c] for c in conversations for _ in range(samples)]
    added, written = [[] for _ in prompts], [[] for _ in prompts]
    counts = [{"calls": 0, "bad_calls": 0, "turns": 0} for _ in prompts]
    active = list(range(len(prompts)))
    device = next(model.parameters()).device if model is not None else "cpu"
    while active:
        order = sorted(active, key=lambda i: len(prompts[i]) + len(added[i]))
        steps = []
        for first in range(0, len(order), batch_size):
            rows = [prompts[i] + added[i] for i in order[first : first + batch_size]]
            steps += torch.stack(list(inference.stream(model, rows, max_new_tokens, temperature=temperature, stop=[end])), dim=1).tolist()
            free_memory(device)
        still = []
        for i, row in zip(order, steps):
            stopped = -1 in row  # the model ended its turn, and did not run into the limit
            ids = row[: row.index(-1)] if stopped else row
            text = tok.decode(ids)
            added[i] += ids + [end] * stopped
            written[i] += [1] * (len(ids) + stopped)
            messages[i].append({"role": "assistant", "content": text})
            counts[i]["turns"] += 1
            made = calls(text) if stopped else []
            if not made or counts[i]["calls"] >= max_calls:
                continue
            content = _results(made, counts[i], execute)
            messages[i].append({"role": "user", "content": content})
            # The tool's turn, and the opening of the model's next one, as encode writes them.
            turn = newline + [start] + tok.encode_ordinary("user\n") + tok.encode_ordinary(content) + [end] + newline + [start] + tok.encode_ordinary("assistant\n")
            added[i] += turn
            written[i] += [0] * len(turn)
            still.append(i)
        active = still
    return prompts, added, written, messages, counts


def answer_text(conversation, start):
    """What the assistant wrote from message `start` on, its calls left out: the text that is scored."""
    turns = [m["content"] for m in conversation[start:] if m["role"] == "assistant"]
    return "\n".join(re.sub(f"{re.escape(CALL_START)}.*?{re.escape(CALL_END)}", "", t, flags=re.S).strip() for t in turns)


# A written calculation, as GSM8K's solutions and the models' replies write them: 16 - 3 - 4 = 9.
NUMBER = r"\d(?:[\d,]*\d)?(?:\.\d+)?"
# It has to stand alone: no digit or point just before it, and after the result no
# operator or digit, as in "1 - 2/3 = 1/3", whose "= 1" is not a result.
CALCULATION = re.compile(rf"(?<![\d.,])((?:{NUMBER}|\(|\))(?:\s*[-+*/x×÷]\s*(?:{NUMBER}|\(|\)))+)\s*=\s*\$?(-?{NUMBER})(?![\d/*x×÷^%]|\.\d|\s*[-+*/]\s*\d)")


def written_calculations(text):
    """
    Every calculation written out in a text, as (expression, what was
    written, whether that is its value). A model that does its arithmetic
    in its head makes mistakes, and this finds them. A calculation counts
    only if it is the whole left side: in "3x + 9 - 1 = 87" the "9 - 1" is
    part of an equation. A result written to fewer decimals counts as right
    if it is the value rounded.
    """
    found = []
    for match in CALCULATION.finditer(text):
        expression, written = match.groups()
        before = text[: match.start()].rstrip()
        if before and before[-1] in "+-*/×÷^(=":
            continue
        value = calculate(expression)
        if value.startswith("error"):
            continue
        number = written.replace(",", "")
        decimals = len(number.split(".")[1]) if "." in number else 0
        right = abs(float(value) - float(number)) <= 0.5 * 10**-decimals + 1e-9 * abs(float(value))
        found.append((expression.strip(), written, right))
    return found
