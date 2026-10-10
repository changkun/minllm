from collections import Counter

import pytest
import regex

from minllm.tokenizer import SPLIT_PATTERN, Tokenizer, merge

TEXT = """Once upon a time, there was a little girl named Lily. She loved to play
outside in the sunshine. One day, she saw a big, red ball in the park.
"Can I play with your ball?" Lily asked the boy. The boy smiled and said,
"Yes, you can!" They played together all day, and they were very happy.
Lily's mom said it was time to go home at 5 o'clock. Thé café cost 1234 ¥. 你好，世界 🙂
"""


def train_textbook(text, n_merges):
    """
    BPE training as it is usually explained: count every pair, merge the most
    frequent one everywhere, repeat. Tokenizer.train must learn the same merges.
    """
    chunks = Counter(regex.findall(SPLIT_PATTERN, text))
    seqs = {chunk: list(chunk.encode("utf-8")) for chunk in chunks}
    merges = {}
    for new_id in range(256, 256 + n_merges):
        pairs = Counter()
        for chunk, seq in seqs.items():
            for pair in zip(seq, seq[1:]):
                pairs[pair] += chunks[chunk]
        if not pairs:
            break
        best = max(pairs, key=lambda p: (pairs[p], p))
        seqs = {chunk: merge(seq, best, new_id) for chunk, seq in seqs.items()}
        merges[best] = new_id
    return merges


def test_merge():
    assert merge([1, 2, 3, 1, 2], (1, 2), 9) == [9, 3, 9]
    assert merge([1, 1, 1], (1, 1), 9) == [9, 1]
    assert merge([], (1, 2), 9) == []


def test_untrained_tokenizer_is_utf8():
    tok = Tokenizer()
    assert tok.vocab_size == 256
    assert tok.encode("héllo") == list("héllo".encode("utf-8"))


def test_train_matches_textbook():
    tok = Tokenizer.train(TEXT * 3, vocab_size=256 + 150)
    assert tok.merges == train_textbook(TEXT * 3, 150)


def test_roundtrip():
    tok = Tokenizer.train(TEXT, vocab_size=400)
    for text in [TEXT, "", " ", "unseen wörds: xylophone 98765\n\n\ttab", "日本語のテキスト 🙂🙃"]:
        assert tok.decode(tok.encode(text)) == text


def test_training_compresses():
    tok = Tokenizer.train(TEXT, vocab_size=400)
    assert len(tok.encode(TEXT)) < len(TEXT.encode("utf-8")) / 2


def test_training_stops_when_nothing_is_left_to_merge():
    tok = Tokenizer.train("aaaa aaaa", vocab_size=1000)
    assert tok.vocab_size < 1000
    assert tok.decode(tok.encode("aaaa aaaa")) == "aaaa aaaa"


def test_special_tokens():
    eot = "<|endoftext|>"
    text = f"one story{eot}another story{eot}"
    tok = Tokenizer.train(text, vocab_size=300, special_tokens=[eot])
    assert tok.vocab_size <= 300
    eot_id = tok.special_tokens[eot]
    assert eot_id == tok.vocab_size - 1

    ids = tok.encode(text)
    assert ids.count(eot_id) == 2 and ids[-1] == eot_id
    assert tok.decode(ids) == text
    # No merge was learned from the characters of the special token.
    assert not any(b"<|" in tok.vocab[i] for i in tok.merges.values())
    # encode_ordinary spells the special token out instead.
    assert eot_id not in tok.encode_ordinary(text)
    assert tok.decode(tok.encode_ordinary(text)) == text


def test_longer_special_token_wins():
    tok = Tokenizer(special_tokens=["<|a|>", "<|a|><|b|>"])
    assert tok.encode("<|a|><|b|>") == [tok.special_tokens["<|a|><|b|>"]]


def test_vocab_size_too_small():
    with pytest.raises(AssertionError):
        Tokenizer.train(TEXT, vocab_size=256, special_tokens=["<|endoftext|>"])


def test_save_load(tmp_path):
    tok = Tokenizer.train(TEXT, vocab_size=400, special_tokens=["<|endoftext|>"])
    tok.save(tmp_path / "tok.json")
    loaded = Tokenizer.load(tmp_path / "tok.json")
    assert loaded.merges == tok.merges and loaded.special_tokens == tok.special_tokens
    text = TEXT + "<|endoftext|>"
    assert loaded.encode(text) == tok.encode(text)
