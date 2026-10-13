import math
import os

import pytest
import torch

from minllm import inference, pretrained, reasoning
from minllm.model import Config, Transformer

TEACHER = "data/r1-distill-qwen-1.5b"
QWEN = "data/qwen2.5-0.5b-instruct"


def test_answers_are_read_from_boxes_first():
    assert reasoning.answer("so 3 + 4 = 7.\n\\boxed{7}") == 7.0
    assert reasoning.answer("\\boxed{1,200} and then 5 more words") == 1200.0
    assert reasoning.answer("\\boxed{\\frac{3}{4}} hmm") == 4.0  # the last number inside; GSM8K's answers are whole numbers
    assert reasoning.answer("\\boxed{\\$18}") == 18.0
    assert reasoning.answer("no box, 16 - 3 = 13\n#### 13") == 13.0  # as stage 7 reads it
    assert reasoning.boxed("\\boxed{a{b}c} x") == "a{b}c" and reasoning.boxed("nothing") is None


def test_thinking_is_split_off():
    assert reasoning.split_thinking("<think>\nfirst 2+2\n</think>\n\nIt is \\boxed{4}.") == ("first 2+2", "It is \\boxed{4}.")
    assert reasoning.split_thinking("still thinking") == ("still thinking", "")


def test_majority():
    assert reasoning.majority([3.0, 5.0, 5.0, None, 3.0]) == 3.0  # a tie goes to the answer given first
    assert reasoning.majority([None, 2.0, 7.0, 7.0]) == 7.0
    assert reasoning.majority([None, None]) is None


def test_vote_curves():
    samples = [[1.0, 1.0, 2.0, 1.0], [2.0, 3.0, 3.0, 3.0], [9.0, 9.0, 9.0, 9.0]]
    curves = reasoning.vote_curves(samples, [1.0, 2.0, 5.0], [1, 2, 4])
    assert curves["majority"][0] == pytest.approx((3 / 4 + 1 / 4 + 0) / 3)  # k = 1: the share of right samples
    assert curves["majority"][2] == pytest.approx((1 + 0 + 0) / 3)  # all four: the vote is 1, then 3, then 9
    assert curves["pass"][0] == pytest.approx(curves["majority"][0])
    assert curves["pass"][1] == pytest.approx((1 + (1 - math.comb(3, 2) / math.comb(4, 2)) + 0) / 3)
    assert curves["pass"][2] == pytest.approx(2 / 3)


def test_forced_answers_cut_at_the_budget():
    torch.manual_seed(0)
    model = Transformer(Config(vocab_size=32, max_seq_len=64, dim=32, n_layers=1, n_heads=2, n_kv_heads=1, mlp_dim=64)).eval()
    prompts, replies, cue = [[1, 2], [3]], [[5, 6, 7, 8, 9], [4, 4]], [30, 31]
    out = reasoning.forced_answers(model, prompts, replies, 3, cue, stop=[0], max_new_tokens=4)
    assert out[1] == [4, 4]  # within the budget: as written
    assert out[0][:5] == [5, 6, 7, 30, 31]
    assert out[0][5:] == inference.generate(model, [[1, 2, 5, 6, 7, 30, 31]], 4, temperature=0.0, stop=[0])[0]


@pytest.mark.skipif(not os.path.exists(f"{TEACHER}/tokenizer.json"), reason="the teacher of docs/09c-reasoning.md is not downloaded")
def test_teacher_prompt_is_its_template():
    tok = pretrained.load_tokenizer(TEACHER)
    ids = reasoning.r1_prompt(tok, "What is 2+2?")
    assert tok.decode(ids) == "<｜begin▁of▁sentence｜><｜User｜>What is 2+2?\n" + reasoning.R1_INSTRUCTION + "<｜Assistant｜><think>\n"
    assert ids[-2] == tok.special_tokens["<think>"]
    if os.path.exists(f"{QWEN}/vocab.json"):  # the vocabulary is Qwen's, with markers of its own
        text = "Natalia sold 48/2 = 24 clips.\n\\boxed{72}"
        assert tok.encode_ordinary(text) == pretrained.load_tokenizer(QWEN).encode_ordinary(text)
