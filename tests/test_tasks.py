import json

from minllm import tasks
from minllm.data import EOT


def test_words_examples(tmp_path):
    stories = [
        "Lily went to the garden. The kitten was purple and very happy.",
        "Tom had a big ball. He liked the ball.",
        "x" * 2000 + " garden",  # too long for the context
        "The dragon sat in the garden with a spoon.",
    ]
    path = tmp_path / "stories.txt"
    path.write_text("".join(f"\n{story}\n{EOT}" for story in stories), encoding="utf-8")
    vocabulary = {"garden", "kitten", "purple", "dragon", "spoon"}

    examples = list(tasks.words_examples(path, vocabulary, n=10, seed=0))
    assert [e["answer"] for e in examples] == [stories[0], stories[3]]  # one has no such word, one is too long
    for e in examples:
        assert 1 <= len(e["words"]) <= 3 and len(set(e["words"])) == len(e["words"])
        assert all(word in vocabulary and word in e["answer"] for word in e["words"])
        assert e["messages"] == [{"role": "user", "content": f"Write a short story using these words: {', '.join(e['words'])}."}]
    assert "Lily" not in vocabulary  # names never qualify: only lowercase words are counted
    assert list(tasks.words_examples(path, vocabulary, n=1)) == examples[:1]


def test_content_words(tmp_path):
    path = tmp_path / "stories.txt"
    path.write_text(f"the the the the cat cat cat garden garden Garden kitten\n{EOT}", encoding="utf-8")
    assert tasks.content_words(path, skip=2, keep=10) == {"garden", "kitten"}  # "the" and "cat" are too frequent


def test_gsm8k_examples(tmp_path):
    row = {"question": "Ann has 3 apples and buys 4. How many?", "answer": "She has 3 + 4 = <<3+4=7>>7 apples.\n#### 7"}
    big = {"question": "How much?", "answer": "It is 1,000 + 250 = <<1000+250=1250>>1,250.\n#### 1,250"}
    (tmp_path / "test.jsonl").write_text(json.dumps(row) + "\n" + json.dumps(big) + "\n", encoding="utf-8")
    first, second = tasks.gsm8k_examples("test", "Be brief.", dir=tmp_path)
    assert first["messages"][0] == {"role": "system", "content": "Be brief."}
    assert first["messages"][1]["content"].startswith(row["question"]) and "####" in first["messages"][1]["content"]
    assert first["answer"] == "She has 3 + 4 = 7 apples.\n#### 7" and first["number"] == "7"
    assert second["number"] == "1250"


def test_words_score():
    example = {"words": ["garden", "kitten", "purple"]}
    assert tasks.words_score(example, "The Kitten sat in the garden.") == 2 / 3
    assert tasks.words_score(example, "A purple kitten. A garden!") == 1.0
    assert tasks.words_score(example, "The gardener had kittens.") == 0.0  # whole words only


def test_gsm8k_score():
    example = {"number": "1250"}
    assert tasks.gsm8k_score(example, "It is 1,000 + 250 = 1,250.\n#### 1,250") == 1.0
    assert tasks.gsm8k_score(example, "First 3 apples, then 1250 in total.\n#### $1250.00") == 1.0
    assert tasks.gsm8k_score(example, "#### 1251") == 0.0
    assert tasks.gsm8k_score(example, "The answer is 1250 dollars.") == 1.0  # no ####: the last number counts
    assert tasks.gsm8k_score(example, "There are 1250 of them, in 3 boxes.") == 0.0
    assert tasks.gsm8k_score(example, "I do not know.") == 0.0
    assert tasks.gsm8k_number("#### -4.5") == -4.5
