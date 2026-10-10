import random

import torch

from minllm import reward
from minllm.model import Config, Transformer
from minllm.tokenizer import Tokenizer


def test_reward_model_learns_from_comparisons():
    torch.manual_seed(0)
    tok = Tokenizer(special_tokens=["<|endoftext|>", "<|im_start|>", "<|im_end|>"])
    body = Transformer(Config(vocab_size=tok.vocab_size, max_seq_len=64, dim=32, n_layers=1, n_heads=2, n_kv_heads=1, mlp_dim=64))
    model = reward.RewardModel(body)
    prompt = [{"role": "user", "content": "q"}]
    # A hidden rule: replies with more a's are better.
    rng = random.Random(0)
    replies = ["".join(rng.choice("ab") for _ in range(rng.randint(4, 12))) for _ in range(64)]
    assert reward.score(model, tok, [prompt] * 4, replies[:4]) == [0.0] * 4  # the head starts at zero

    # Padding on the right does not change a row's reward.
    x, lengths = reward.encode(tok, [prompt] * 2, [replies[0], replies[0] + "bbbbbbbb"], "cpu")
    alone, _ = reward.encode(tok, [prompt], [replies[0]], "cpu")
    with torch.no_grad():
        model.head.weight.normal_()
        assert torch.allclose(model(x, lengths)[0], model(alone, lengths[:1])[0], atol=1e-5)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    for _ in range(150):
        i, j = rng.sample(range(48), 2)
        better, worse = (i, j) if replies[i].count("a") / len(replies[i]) >= replies[j].count("a") / len(replies[j]) else (j, i)
        x, lengths = reward.encode(tok, [prompt] * 2, [replies[better], replies[worse]], "cpu")
        r = model(x, lengths)
        loss = reward.comparison_loss(r[0], r[1])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    scores = reward.score(model, tok, [prompt] * 16, replies[48:])  # replies it never saw
    truth = [r.count("a") / len(r) for r in replies[48:]]
    agree = sum((scores[i] > scores[j]) == (truth[i] > truth[j]) for i in range(16) for j in range(i) if truth[i] != truth[j])
    total = sum(truth[i] != truth[j] for i in range(16) for j in range(i))
    assert agree / total > 0.8
