"""
Evaluation by a judge: a language model that scores what another wrote.

Some replies can be checked by a program: a number is right or wrong.
Most cannot. Whether a story is good, a summary faithful or an answer
helpful is a judgment, and asking people for it is slow and expensive. A
model can be asked instead. It reads the task, the reply, and what to look
for, and gives a verdict.

The verdict is read the way docs/07a-multiple-choice.md reads an answer:
the prompt ends where the verdict begins, and the model's probabilities
for the possible verdicts are compared. That gives a number between 0 and
1 for every verdict, and needs no reply to be parsed. The judge is a model
like any other, and this file measures it against verdicts that are known:
the right answers of GSM8K, and the hidden taste of the annotator of
docs/08-rlhf.md. See docs/07b-judge.md.
"""

import torch

from minllm import chat, choice

# --------------------------------------------------------------- the prompts


def correctness(problem, reply, reference=None):
    """A conversation that asks whether a solution's final answer is right, with the right answer given if `reference` is."""
    given = f"\n\nThe correct final answer is {reference}." if reference is not None else ""
    content = (
        "Here is a math problem and a proposed solution.\n\n"
        f"Problem:\n{problem}\n\nSolution:\n{reply}{given}\n\n"
        'Is the final answer of the solution correct? Reply with a JSON object, {"correct": true} or {"correct": false}.'
    )
    return [{"role": "user", "content": content}]


CORRECT = ('{"correct":', [" true", " false"])

# What the annotator of docs/08-rlhf.md likes, in words. minllm/preferences.py
# has it as a function; here a judge is told it.
RUBRIC = (
    "A good story ends happily, has someone speak, is short (about 40 to 110 words), "
    "is not gloomy, and does not stop in the middle of a sentence."
)


def comparison(instruction, a, b, rubric=None):
    """A conversation that asks which of two replies to the same instruction is better."""
    criteria = f"\n\n{rubric}" if rubric else ""
    content = (
        f"Two stories were written for the same request.\n\nRequest: {instruction}\n\n"
        f"Story A:\n{a}\n\nStory B:\n{b}{criteria}\n\n"
        'Which story is better? Reply with a JSON object, {"better": "A"} or {"better": "B"}.'
    )
    return [{"role": "user", "content": content}]


BETTER = ('{"better": "', ["A", "B"])


# ---------------------------------------------------------------- the verdict


@torch.no_grad()
def verdicts(model, tok, conversations, verdict, system="", budget=24_000, progress=None):
    """
    The judge's probability of each possible verdict, (conversations,
    verdicts), among those verdicts. verdict is (the text that opens the
    assistant's reply, [the verdicts, each one token]): the reply is
    started for the judge, and only its next token is read.
    """
    opening, options = verdict
    ids = [tok.encode_ordinary(option) for option in options]
    assert all(len(i) == 1 for i in ids), "every verdict has to be a single token"
    prefix = tok.encode_ordinary(opening)
    with_system = lambda c: ([{"role": "system", "content": system}] if system else []) + c
    prompts = [chat.encode(tok, with_system(c)) + prefix for c in conversations]
    logp, _ = choice.next_token_logprobs(model, prompts, [i[0] for i in ids], budget, progress)
    return logp.softmax(-1)
