"""
Human preferences, and a stand-in for the human.

RLHF starts from comparisons: an annotator reads two replies to the same
prompt and says which is better. Nobody writes down what "better" means. It
exists only in the annotator's head, and the comparisons are noisy glimpses of
it.

To study the method, this file plays the annotator. It has a hidden taste, a
function that scores a story, and it compares two stories by their scores
with some noise, as a person would. Because the taste is known here, every
later step can be checked against it: whether the reward model recovered it,
and whether a policy trained on the reward model got better by the
annotator's real standard or only by the model's. With real annotators that
check is not available, which is the central difficulty of RLHF.
scripts/label_preferences.py collects comparisons from you instead, in the
same format. See docs/08-rlhf.md.
"""

import json
import math
import random
import re

HAPPY = {"happy", "happily", "smiled", "smile", "laughed", "fun", "friends", "love", "loved", "hug", "hugged", "glad", "joy"}
SAD = {"sad", "cried", "cry", "scared", "angry", "hurt", "mad", "afraid", "lost", "broke", "broken", "sorry"}


def taste(story):
    """
    The simulated annotator's taste, higher is better. They like a story that
    ends happily, has someone speak, stays short, and is not gloomy. They
    dislike one that stops in the middle of a sentence.
    """
    words = re.findall(r"[a-z']+", story.lower())
    if not words:
        return -2.0
    sentences = [s for s in re.split(r"(?<=[.!?\"”])\s+", story.strip()) if s]
    last = set(re.findall(r"[a-z']+", sentences[-1].lower())) if sentences else set()
    score = 1.0 * bool(last & HAPPY)  # a happy ending
    score += 0.5 * any(quote in story for quote in '"“”')  # dialogue
    score -= 0.25 * min(4, sum(word in SAD for word in words))  # gloom
    score -= 1.0 * max(0, len(words) - 110) / 60  # long-winded
    score -= 1.0 * max(0, 40 - len(words)) / 40  # or too thin to be a story
    score -= 1.0 * (story.strip()[-1] not in '.!?"”')  # unfinished
    return score


def prefer(a, b, rng, noise=0.4):
    """
    Which of two stories the annotator picks: 0 for a, 1 for b. The larger
    the difference in taste, the surer the choice, and close calls go either
    way. This is the Bradley-Terry model of a comparison.
    """
    p_a = 1 / (1 + math.exp(-(taste(a) - taste(b)) / noise))
    return 0 if rng.random() < p_a else 1


def save(path, pairs):
    with open(path, "w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")


def load(path):
    """Comparisons, each {"messages": the prompt, "replies": [a, b], "preferred": 0 or 1}."""
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]
