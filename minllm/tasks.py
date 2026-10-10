"""
Tasks: what a model is fine-tuned to do, and later checked and rewarded on.

A task turns its data into conversations. An example is a dict with
"messages", the conversation up to the point where the model answers, and
"answer", a reference reply to learn from. See docs/06-finetuning.md.
"""

import json
import random
import re
from collections import Counter

from minllm import preferences
from minllm.common import fetch
from minllm.data import read_stories

# ---------------------------------------------------------------- words
#
# "Write a short story using these words: garden, kitten." The examples are
# cut from TinyStories itself: take a story, pick words that occur in it, and
# ask for a story with those words. The story is then a correct answer.

LOWERCASE_WORD = re.compile(r"\b[a-z]+\b")


def content_words(path="data/tinystories/train.txt", skip=300, keep=1500, max_bytes=32 << 20):
    """
    The words worth asking for: at least four letters, not among the `skip`
    most frequent words of the corpus (the, was, said, happy), and frequent
    enough to be learned, the `keep` that come next. Names never qualify,
    because only words written in lowercase are counted.
    """
    counts = Counter()
    for story in read_stories(path, max_bytes=max_bytes):
        counts.update(LOWERCASE_WORD.findall(story))
    ranked = [word for word, _ in counts.most_common()]
    return {word for word in ranked[skip : skip + keep] if len(word) >= 4}


def words_examples(path, vocabulary, n, seed=0, max_words=3, max_story_chars=800):
    """Yield up to n examples, each built from one story of the file at path."""
    rng = random.Random(seed)
    made = 0
    for story in read_stories(path):
        candidates = sorted(set(LOWERCASE_WORD.findall(story)) & vocabulary)
        if len(story) > max_story_chars or not candidates:
            continue  # the story has to fit the small model's context
        words = rng.sample(candidates, min(rng.randint(1, max_words), len(candidates)))
        yield {"messages": words_prompt(words), "answer": story, "words": words}
        made += 1
        if made == n:
            return


def words_prompt(words):
    return [{"role": "user", "content": f"Write a short story using these words: {', '.join(words)}."}]


def words_score(example, reply):
    """The fraction of the requested words that the reply uses, as whole words."""
    used = set(re.findall(r"[a-z]+", reply.lower()))
    return sum(word in used for word in example["words"]) / len(example["words"])


# ---------------------------------------------------------------- gsm8k
#
# Grade-school math word problems (Cobbe et al., 2021, arXiv:2110.14168),
# 7,473 for training and 1,319 for testing. Each has a worked solution that
# ends in a line "#### <number>".

GSM8K_URL = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/{split}.jsonl"
GSM8K_INSTRUCTION = "Show your work, then give the final answer on the last line as '#### <number>'."


def gsm8k_examples(split, system_prompt, dir="data/gsm8k"):
    """The examples of one split, "train" or "test", downloaded on first use."""
    path = f"{dir}/{split}.jsonl"
    fetch(GSM8K_URL.format(split=split), path)
    examples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            # The solutions carry calculator annotations, "<<16-3-4=9>>9". The text reads fine without them.
            solution = re.sub(r"<<.*?>>", "", row["answer"])
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"{row['question']}\n\n{GSM8K_INSTRUCTION}"},
            ]
            number = solution.split("####")[-1].strip().replace(",", "")
            examples.append({"messages": messages, "answer": solution, "number": number})
    return examples


def gsm8k_number(reply):
    """The number a reply gives as its answer: the one after the last ####, or else the last number in it."""
    text = reply.split("####")[-1] if "####" in reply else reply
    numbers = re.findall(r"-?\d[\d,]*\.?\d*", text)
    if not numbers:
        return None
    number = (numbers[0] if "####" in reply else numbers[-1]).replace(",", "").rstrip(".")
    try:
        return float(number)
    except ValueError:
        return None


def gsm8k_score(example, reply):
    """1 if the reply's answer is the right number, else 0."""
    return float(gsm8k_number(reply) == float(example["number"]))


# ------------------------------------------------------------- evaluation

# "taste" scores the stories of the words task by the simulated annotator's
# hidden standard (minllm/preferences.py), which stage 8 tries to learn.
SCORE = {"words": words_score, "gsm8k": gsm8k_score, "taste": lambda example, reply: preferences.taste(reply)}
MAX_NEW_TOKENS = {"words": 220, "taste": 220, "gsm8k": 320}  # room for a whole reply


def test_examples(task, n=None, system_prompt=""):
    """Examples the models were never trained on: held-out stories, and GSM8K's test split."""
    if task in ("words", "taste"):
        return list(words_examples("data/tinystories/valid.txt", content_words(), n or 1000, seed=7))
    return gsm8k_examples("test", system_prompt)[:n]
