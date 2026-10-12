"""
Multiple choice: scoring a model by the probabilities it gives, without
letting it write.

A question with four options has four possible continuations, and a language
model gives every continuation a probability. Which of the four it finds
likeliest takes one forward pass to see. Nothing is sampled, no reply has to
be parsed, and no limit on a reply's length can cut the answer off. This is
how evaluation harnesses score MMLU and most other multiple-choice
benchmarks. The log-probability of a continuation is the number that stage 9
computed to train on; here it is only read. See docs/07a-multiple-choice.md.
"""

import re

import torch
import torch.nn.functional as F

from minllm import chat, rl

LETTERS = "ABCD"
HEADER = "The following are multiple choice questions (with answers) about {subject}.\n\n"

# ---------------------------------------------------------------- prompts


def ask(example, answered=False):
    """
    A question with its lettered options, ending in "Answer:", as the
    authors of MMLU wrote the prompt. An answered one goes on with the right
    letter and a blank line, to be shown to the model as an example.
    """
    lines = [example["question"].strip()] + [f"{letter}. {option}" for letter, option in zip(LETTERS, example["options"])]
    return "\n".join(lines) + "\nAnswer:" + (f" {LETTERS[example['answer']]}\n\n" if answered else "")


def lettered(example, shots=()):
    """The whole prompt of a question: what the subject is, solved examples of that subject, and the question."""
    return HEADER.format(subject=example["subject"].replace("_", " ")) + "".join(ask(shot, answered=True) for shot in shots) + ask(example)


def cloze(example):
    """The question alone, with no options to choose from. The model is then scored on each option's text as the continuation."""
    return f"Question: {example['question'].strip()}\nAnswer:"


def encode(tok, text, format, system=""):
    """
    The tokens of a prompt. "plain" is the text itself, to be continued, and
    the answer then starts with a space. "chat" puts the text into a user
    turn and opens the assistant's, and the answer is its first token.
    """
    if format == "plain":
        return tok.encode_ordinary(text)
    messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": text}]
    return chat.encode(tok, messages)


def letter_ids(tok, format):
    """The token of each letter as an answer: " A" after "Answer:", and "A" at the start of an assistant's turn."""
    ids = [tok.encode_ordinary(letter if format == "chat" else f" {letter}") for letter in LETTERS]
    assert all(len(i) == 1 for i in ids), "every answer letter has to be a single token"
    return [i[0] for i in ids]


def rotated(example, by):
    """The same question with its options moved round by `by` places: what was option A is then under another letter."""
    n = len(example["options"])
    return example | {"options": [example["options"][(i - by) % n] for i in range(n)], "answer": (example["answer"] + by) % n}


ANSWERS = [
    re.compile(r"answer is:?\s*\**\(?([ABCD])\b"),  # "the correct answer is B", "The answer is: **(B)"
    re.compile(r"Answer:\s*\**\(?([ABCD])\b"),
    re.compile(r"^\s*\**\(?([ABCD])\)?\**(?:[.:)]|\s*$)", re.MULTILINE),  # a line that opens with the letter: "B", "B.", "(B) 4"
]


def read_letter(reply):
    """
    The letter a written reply gives as its answer, as an index, or None.
    A reply is prose, and "A" is also a word, so a letter counts only where
    a reply names its answer: after "answer is" or "Answer:", or at the
    start of a line, followed by a period, a colon or a bracket. The first
    of these that occurs decides. A reply that lists all four options
    before it chooses is therefore read as A.
    """
    for pattern in ANSWERS:
        if found := pattern.search(reply):
            return LETTERS.index(found.group(1))
    return None


# ---------------------------------------------------------------- scoring


def batches(lengths, budget):
    """
    Indices in batches of similar length, the longest first, so that a
    batch that does not fit fails at once. A batch holds at most `budget`
    tokens, padding included.
    """
    order = sorted(range(len(lengths)), key=lambda i: -lengths[i])
    start = 0
    while start < len(order):
        rows = max(1, budget // lengths[order[start]])
        yield order[start : start + rows]
        start += rows


def _hidden(model, x):
    device_type = x.device.type
    with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=device_type != "cpu"):
        return model.hidden(x)


@torch.no_grad()
def next_token_logprobs(model, contexts, ids, budget=24_000, progress=None):
    """
    log p of each token in `ids` as the next token after every context,
    (contexts, ids), and the token that the model finds likeliest of all
    there, (contexts,). One forward pass per context, and only its last
    position is used. The output layer is applied in 32-bit floats: in 16
    bits, two logits of size 20 closer than an eighth cannot be told apart.
    """
    device = next(model.parameters()).device
    head = model.lm_head.weight.float()
    out = torch.empty(len(contexts), len(ids))
    likeliest = torch.empty(len(contexts), dtype=torch.long)
    done = 0
    for rows in batches([len(c) for c in contexts], budget):
        x = torch.zeros(len(rows), len(contexts[rows[0]]), dtype=torch.long)
        for r, i in enumerate(rows):
            x[r, : len(contexts[i])] = torch.tensor(contexts[i])  # padded on the right: a causal model's earlier positions do not see it
        last = torch.tensor([len(contexts[i]) - 1 for i in rows])
        h = _hidden(model, x.to(device))[torch.arange(len(rows)), last.to(device)]
        logp = F.log_softmax(F.linear(h.float(), head), dim=-1)
        out[rows] = logp[:, ids].cpu()
        likeliest[rows] = logp.argmax(-1).cpu()
        done += len(rows)
        if progress:
            progress(done, len(contexts))
    return out, likeliest


@torch.no_grad()
def continuation_logprobs(model, contexts, continuations, budget=24_000, progress=None):
    """
    The sum of log p over the tokens of each continuation after its
    context, one number per pair: the log of the probability that the model
    writes exactly this text there.
    """
    device = next(model.parameters()).device
    head = model.lm_head.weight.float()
    out = torch.empty(len(contexts))
    done = 0
    for rows in batches([len(c) + len(t) for c, t in zip(contexts, continuations)], budget):
        x, y, which = rl.batch([contexts[i] for i in rows], [continuations[i] for i in rows], device)
        h, targets = _hidden(model, x)[y != -100].float(), y[y != -100]
        # A thousand positions at a time: the logits of one are as many numbers as the vocabulary has tokens.
        logp = torch.cat([rl.token_logprobs(F.linear(part, head), part_targets) for part, part_targets in zip(h.split(1024), targets.split(1024))])
        sums = torch.zeros(len(rows), device=device).index_add(0, which, logp)
        out[rows] = sums.cpu()
        done += len(rows)
        if progress:
            progress(done, len(contexts))
    return out
