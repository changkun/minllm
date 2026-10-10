import random

from minllm import preferences

GOOD = 'Tom found a ball in the park. "Do you want to play?" he asked Mia. ' + "They played all day in the sun. " * 6 + "Tom and Mia were happy."
SAD = "Tom was sad. He cried because he lost his ball. " + "He looked and looked but it was lost. " * 6 + "Tom was angry and went home."


def test_taste():
    assert preferences.taste(GOOD) == 1.5  # a happy ending and dialogue, nothing held against it
    assert preferences.taste(SAD) < 0 < preferences.taste(GOOD)
    assert preferences.taste(GOOD + " Then the") < preferences.taste(GOOD) - 1.9  # cut off, and the happy ending is gone
    assert preferences.taste(GOOD + " They played more." * 40 + " They were happy.") < preferences.taste(GOOD)  # long-winded
    assert preferences.taste("Happy.") < preferences.taste(GOOD)  # too thin
    assert preferences.taste("") == -2.0
    assert preferences.taste(GOOD.replace('"Do you want to play?"', "“Do you want to play?”")) == 1.5  # curly quotes count too


def test_annotator_is_noisy_but_right_on_average():
    rng = random.Random(0)
    clear = sum(preferences.prefer(GOOD, SAD, rng) == 0 for _ in range(200))
    assert clear == 200  # no contest
    close = GOOD.replace('"Do you want to play?" he asked Mia.', "He asked Mia to play with him and she said yes.")  # only dialogue differs
    picks = sum(preferences.prefer(GOOD, close, rng) == 0 for _ in range(2000))
    assert 1400 < picks < 1700  # preferred about 78% of the time, not always


def test_save_load(tmp_path):
    pairs = [{"messages": [{"role": "user", "content": "hi"}], "replies": ["a", "b é"], "preferred": 1}]
    preferences.save(tmp_path / "p.jsonl", pairs)
    assert preferences.load(tmp_path / "p.jsonl") == pairs
