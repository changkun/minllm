"""
Reasoning: spending more computation on a problem at the time it is asked.

A reasoning model writes a long chain of thought before it answers, and is
right more often the longer it is allowed to think. Three ways to such a
model are measured in docs/09c-reasoning.md: reinforcement learning with a
verifier from a model that already writes its steps (as DeepSeek-R1 was
trained), fine-tuning on the thinking of a reasoning model (as the small
DeepSeek-R1 models were made), and, without training, sampling several
answers and taking the most common (self-consistency). This file holds
what they share: the teacher's format, reading an answer, the vote, and
cutting a reply's thinking at a budget.
"""

import math
import re
from collections import Counter

import torch

from minllm import inference, tasks

# ------------------------------------------------------------- the teacher

# DeepSeek-R1-Distill-Qwen-1.5B: Qwen2.5-Math-1.5B fine-tuned on the
# thinking of DeepSeek-R1. It reads Qwen's vocabulary with markers of its
# own, and its chat template opens the reply with <think> so that it starts
# by thinking. Its authors ask for no system prompt, the instruction below
# in the user's turn, and sampling at temperature 0.6 with top-p 0.95.
R1_BOS, R1_EOS = "<｜begin▁of▁sentence｜>", "<｜end▁of▁sentence｜>"
R1_USER, R1_ASSISTANT = "<｜User｜>", "<｜Assistant｜>"
THINK, END_THINK = "<think>", "</think>"
R1_INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."


def r1_prompt(tok, question):
    """The teacher's prompt, as its chat template writes it with the generation prompt: the reply opens with <think> and a newline."""
    s = tok.special_tokens
    return [s[R1_BOS], s[R1_USER]] + tok.encode_ordinary(f"{question}\n{R1_INSTRUCTION}") + [s[R1_ASSISTANT], s[THINK]] + tok.encode_ordinary("\n")


def boxed(text):
    """The contents of the last \\boxed{...} in a text, braces inside it included, or None."""
    start = text.rfind("\\boxed{")
    if start < 0:
        return None
    depth, i = 1, start + len("\\boxed{")
    for j in range(i, len(text)):
        depth += {"{": 1, "}": -1}.get(text[j], 0)
        if depth == 0:
            return text[i:j]
    return None


def answer(reply):
    """
    The number a reply gives as its answer: the one in its last \\boxed{}
    if it has one, as the teacher writes it, or else as stage 7 reads it
    (tasks.gsm8k_number).
    """
    inside = boxed(reply)
    if inside is not None:
        numbers = re.findall(r"-?\d[\d,]*\.?\d*", inside.replace("\\!", ""))
        if numbers:
            try:
                return float(numbers[-1].replace(",", "").rstrip("."))
            except ValueError:
                pass
    return tasks.gsm8k_number(reply)


def split_thinking(reply):
    """The thinking, before </think>, and the answer after it. A reply that never closed its thinking is all thinking."""
    if END_THINK not in reply:
        return reply, ""
    thinking, after = reply.split(END_THINK, 1)
    return thinking.removeprefix(THINK).strip(), after.strip()


# --------------------------------------------------------------- the vote


def majority(numbers):
    """The most common answer among several, ties going to the one given first; None if none was read."""
    counted = Counter(n for n in numbers if n is not None)
    if not counted:
        return None
    top = max(counted.values())
    return next(n for n in numbers if counted.get(n) == top)


def vote_curves(samples, truths, ks):
    """
    From n sampled answers per problem, at every k: majority@k, how often
    the most common of k answers is right, averaged over the n / k disjoint
    groups of k; and pass@k, how often at least one of k is, estimated from
    all n as Chen et al. (2021) do: 1 - C(n - c, k) / C(n, k) for c right.
    pass@k is what a perfect judge of the k answers would reach; the vote
    is what a judge that only counts reaches.
    """
    curves = {"majority": [], "pass": []}
    for k in ks:
        vote, passed = 0.0, 0.0
        for answers, truth in zip(samples, truths):
            groups = [answers[i : i + k] for i in range(0, len(answers) - k + 1, k)]
            vote += sum(majority(g) == truth for g in groups) / len(groups)
            n, c = len(answers), sum(a == truth for a in answers)
            passed += 1.0 - (math.comb(n - c, k) / math.comb(n, k) if n - c >= k else 0.0)
        curves["majority"].append(vote / len(samples))
        curves["pass"].append(passed / len(samples))
    return curves


# ------------------------------------------------------------- the budget


@torch.no_grad()
def forced_answers(model, prompts, replies, budget, cue, stop, max_new_tokens=24, batch_size=64):
    """
    Each reply cut after `budget` of its tokens and made to answer there:
    the cue (token ids) is appended, such as the end of the thinking and
    the opening of the answer, and the model writes on greedily. A reply
    that ended within the budget keeps its own ending. Returns the token
    ids of every reply as answered, which is how a budget for thinking is
    enforced (Muennighoff et al., 2025).
    """
    out = [None] * len(replies)
    cut = [i for i, r in enumerate(replies) if len(r) > budget]
    for i, r in enumerate(replies):
        if len(r) <= budget:
            out[i] = r
    for start in range(0, len(cut), batch_size):
        chunk = cut[start : start + batch_size]
        rows = [prompts[i] + replies[i][:budget] + cue for i in chunk]
        written = inference.generate(model, rows, max_new_tokens, temperature=0.0, stop=stop)
        for i, w in zip(chunk, written):
            out[i] = replies[i][:budget] + cue + w
    return out
