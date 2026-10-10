"""
Evaluation: how often does a model do what a task asks?

A task supplies examples and a scoring function. Evaluating a model is then
three steps: generate a reply for every example, score each reply, and say how
sure the average is. See docs/07-evaluation.md.
"""

import math
import random

import torch

from minllm import chat
from minllm.data import EOT
from minllm.inference import generate


@torch.no_grad()
def replies(model, tok, examples, max_new_tokens, batch_size=64, temperature=0.0, top_k=None, top_p=None, progress=None):
    """The model's reply to every example, as text, in the order given."""
    stop = [tok.special_tokens[chat.IM_END], tok.special_tokens[EOT]]
    prompts = [chat.encode(tok, e["messages"]) for e in examples]
    # Batches of similar length waste the least on padding.
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    out = [None] * len(prompts)
    device_type = next(model.parameters()).device.type
    for start in range(0, len(order), batch_size):
        chunk = order[start : start + batch_size]
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=device_type != "cpu"):
            ids = generate(model, [prompts[i] for i in chunk], max_new_tokens, temperature=temperature, top_k=top_k, top_p=top_p, stop=stop)
        for i, row in zip(chunk, ids):
            out[i] = tok.decode(row)
        if progress:
            progress(min(start + batch_size, len(order)), len(order))
    return out


def mean_and_error(scores):
    """
    The mean score and the half-width of its 95% confidence interval. The
    test examples are a sample of all possible ones, so the mean over them is
    an estimate. With n examples its standard error is the standard deviation
    of the scores divided by sqrt(n), and 1.96 standard errors to either side
    cover the true value 95% of the time.
    """
    n = len(scores)
    mean = sum(scores) / n
    variance = sum((s - mean) ** 2 for s in scores) / max(1, n - 1)
    return mean, 1.96 * math.sqrt(variance / n)


def paired_difference(a, b, resamples=2000, seed=0):
    """
    How much better is model b than model a, when both were scored on the
    same examples? Returns the mean difference and its 95% interval, by the
    bootstrap: resample the examples with replacement many times and see how
    the difference varies. Comparing per example removes the noise of which
    examples happened to be drawn, so this interval is much tighter than
    comparing the two models' separate intervals.
    """
    assert len(a) == len(b)
    rng = random.Random(seed)
    diffs = [y - x for x, y in zip(a, b)]
    n = len(diffs)
    means = sorted(sum(rng.choices(diffs, k=n)) / n for _ in range(resamples))
    return sum(diffs) / n, means[int(0.025 * resamples)], means[int(0.975 * resamples)]
