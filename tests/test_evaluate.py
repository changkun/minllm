import torch

from minllm import chat, evaluate
from minllm.model import Config, Transformer
from minllm.tokenizer import Tokenizer


def test_replies_come_back_in_order():
    torch.manual_seed(0)
    tok = Tokenizer(special_tokens=["<|endoftext|>", "<|im_start|>", "<|im_end|>"])
    model = Transformer(Config(vocab_size=tok.vocab_size, max_seq_len=96, dim=32, n_layers=1, n_heads=2, n_kv_heads=1, mlp_dim=64)).eval()
    examples = [{"messages": [{"role": "user", "content": "x" * n}]} for n in (9, 2, 14, 5, 2)]
    together = evaluate.replies(model, tok, examples, max_new_tokens=6, batch_size=2)
    alone = [evaluate.replies(model, tok, [e], max_new_tokens=6)[0] for e in examples]
    assert together == alone and together[1] == together[4]  # equal prompts, equal greedy replies
    assert all(chat.IM_END not in reply for reply in together)


def test_mean_and_error():
    mean, error = evaluate.mean_and_error([1.0] * 50 + [0.0] * 50)
    assert mean == 0.5 and abs(error - 1.96 * 0.5025 / 10) < 1e-3  # about ten points either way at n = 100
    assert evaluate.mean_and_error([1.0] * 10) == (1.0, 0.0)
    # Four times the examples halve the error.
    assert abs(evaluate.mean_and_error([1.0, 0.0] * 200)[1] / evaluate.mean_and_error([1.0, 0.0] * 50)[1] - 0.5) < 0.01


def test_paired_difference():
    a = [1.0, 0.0] * 100
    b = [1.0, 1.0] + a[2:]  # the same answers, but for one example it now gets right
    diff, low, high = evaluate.paired_difference(a, b)
    assert abs(diff - 1 / 200) < 1e-9 and low >= 0.0 and high <= 0.02
    # Scored separately, the two models' intervals overlap almost entirely.
    assert evaluate.mean_and_error(a)[1] > 5 * (high - low) / 2
