from minllm import selftrain


def test_keep():
    # three prompts with four replies each
    texts = ["a", "b", "c", "d"] + ["e", "e", "e", "f"] + ["g", "h", "i", "j"]
    scores = [1, 1, 0.5, 1] + [1, 1, 1, 0] + [0, 0, 0, 0]
    finished = [True, False, True, True] + [True] * 4 + [True] * 4
    # the second reply ran into the token limit, and the third used half of its words
    assert selftrain.keep(texts, scores, finished, samples=4, per_prompt=2) == [0, 3, 4]
    # a prompt contributes at most per_prompt replies, and the same text only once
    assert selftrain.keep(texts, scores, finished, samples=4, per_prompt=1) == [0, 4]
    assert selftrain.keep(texts, scores, finished, samples=4, per_prompt=4) == [0, 3, 4]
    # a prompt the model never got right teaches nothing
    assert not [i for i in selftrain.keep(texts, scores, finished, samples=4) if i >= 8]
