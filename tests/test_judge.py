import torch

from minllm import judge
from minllm.model import Config, Transformer
from minllm.tokenizer import Tokenizer


def test_prompts_carry_what_they_are_given():
    c = judge.correctness("What is 2 + 2?", "2 + 2 = 4\n#### 4", reference="4")
    assert "Problem:\nWhat is 2 + 2?" in c[0]["content"] and "The correct final answer is 4." in c[0]["content"]
    assert "The correct final answer" not in judge.correctness("q", "r")[0]["content"]
    pair = judge.comparison("Write a story.", "first", "second", rubric=judge.RUBRIC)
    text = pair[0]["content"]
    assert text.index("Story A:\nfirst") < text.index("Story B:\nsecond") < text.index("A good story ends happily")


def test_verdicts_are_the_next_token_among_the_options():
    tok = Tokenizer.train('{"better": "A"} {"better": "B"} Story A B ' * 40, 300, special_tokens=["<|endoftext|>", "<|im_start|>", "<|im_end|>"])
    torch.manual_seed(0)
    model = Transformer(Config(vocab_size=tok.vocab_size, max_seq_len=256, dim=32, n_layers=2, n_heads=4, n_kv_heads=2, mlp_dim=64)).eval()
    conversations = [judge.comparison("Write.", "a", "b"), judge.comparison("Write.", "b", "a")]
    p = judge.verdicts(model, tok, conversations, judge.BETTER)
    assert p.shape == (2, 2) and torch.allclose(p.sum(-1), torch.ones(2))
    # by hand: the probabilities of "A" and "B" after the opening, renormalized
    from minllm import chat

    prompt = chat.encode(tok, conversations[0]) + tok.encode_ordinary(judge.BETTER[0])
    logits = model(torch.tensor([prompt]))[0][0, -1]
    ids = [tok.encode_ordinary(o)[0] for o in judge.BETTER[1]]
    assert torch.allclose(p[0], logits[ids].softmax(-1), atol=1e-5)
