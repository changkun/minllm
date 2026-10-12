import math

import torch

from minllm import choice
from minllm.model import Config, Transformer
from minllm.tokenizer import Tokenizer

EXAMPLE = {"subject": "high_school_physics", "question": "What falls faster in a vacuum? ", "options": ["a feather", "a stone", "neither", "both"], "answer": 2}


def model_and_tokenizer():
    torch.manual_seed(0)
    tok = Tokenizer.train("the question is answered by a letter. A B C D " * 20, 300, special_tokens=["<|endoftext|>", "<|im_start|>", "<|im_end|>"])
    model = Transformer(Config(vocab_size=len(tok.vocab), max_seq_len=128, dim=32, n_layers=2, n_heads=4, n_kv_heads=2, mlp_dim=64)).eval()
    return model, tok


def test_prompt_is_the_authors():
    shot = EXAMPLE | {"question": "Is ice cold?", "options": ["yes", "no", "maybe", "unknown"], "answer": 0}
    text = choice.lettered(EXAMPLE, [shot])
    assert text.startswith("The following are multiple choice questions (with answers) about high school physics.\n\n")
    assert "Is ice cold?\nA. yes\nB. no\nC. maybe\nD. unknown\nAnswer: A\n\n" in text
    assert text.endswith("What falls faster in a vacuum?\nA. a feather\nB. a stone\nC. neither\nD. both\nAnswer:")
    assert choice.cloze(EXAMPLE) == "Question: What falls faster in a vacuum?\nAnswer:"


def test_rotation_moves_the_answer_with_its_option():
    for by in range(4):
        turned = choice.rotated(EXAMPLE, by)
        assert turned["options"][turned["answer"]] == "neither"
        assert sorted(turned["options"]) == sorted(EXAMPLE["options"])
    assert choice.rotated(EXAMPLE, 1)["options"] == ["both", "a feather", "a stone", "neither"]


def test_read_letter():
    assert choice.read_letter("B") == 1
    assert choice.read_letter("B. a stone") == 1
    assert choice.read_letter("The answer is C.") == 2
    assert choice.read_letter("To see which falls faster, consider the vacuum. The correct answer is: **(D)** both.") == 3
    assert choice.read_letter("Answer: D. both") == 3
    assert choice.read_letter("What falls faster in a vacuum is:\n\nC. neither\n\nBoth fall alike.") == 2  # the question again, then the option
    assert choice.read_letter("Both of them fall.") is None  # the B of a word is not an answer
    assert choice.read_letter("A feather and a stone fall alike in a vacuum.") is None  # nor is the article
    assert choice.read_letter("To answer this, let us look at each option in turn.") is None


def test_batches_cover_everything_within_the_budget():
    lengths = [5, 90, 17, 40, 40, 3, 64]
    seen = []
    for rows in choice.batches(lengths, 100):
        assert len(rows) == 1 or len(rows) * max(lengths[i] for i in rows) <= 100
        seen += rows
    assert sorted(seen) == list(range(len(lengths)))


def test_next_token_logprobs_match_the_model():
    model, tok = model_and_tokenizer()
    contexts = [tok.encode_ordinary(text) for text in ["the question", "a letter. A B", "the question is answered by a letter. A B C"]]
    ids = choice.letter_ids(tok, "plain")
    got, likeliest = choice.next_token_logprobs(model, contexts, ids, budget=20)  # a budget that forces several batches
    for context, row, top in zip(contexts, got, likeliest):
        logits = model(torch.tensor([context]))[0][0, -1]
        assert torch.allclose(row, logits.log_softmax(-1)[ids], atol=1e-5)
        assert top == logits.argmax()


def test_continuation_logprobs_are_the_log_of_a_product():
    model, tok = model_and_tokenizer()
    contexts = [tok.encode_ordinary("the question is"), tok.encode_ordinary("a letter.")]
    continuations = [tok.encode_ordinary(" answered by a letter"), tok.encode_ordinary(" A")]
    got = choice.continuation_logprobs(model, contexts, continuations, budget=16)
    for context, continuation, total in zip(contexts, continuations, got):
        logp = model(torch.tensor([context + continuation]))[0][0].log_softmax(-1)
        expected = sum(logp[len(context) - 1 + i, t].item() for i, t in enumerate(continuation))
        assert math.isclose(total.item(), expected, abs_tol=1e-4)
    # one token as a continuation is the next token
    ids = choice.letter_ids(tok, "plain")
    one = choice.continuation_logprobs(model, [contexts[1]] * 4, [[i] for i in ids])
    assert torch.allclose(one, choice.next_token_logprobs(model, [contexts[1]], ids)[0][0], atol=1e-5)
