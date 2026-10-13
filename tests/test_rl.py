import torch

from minllm import chat, rl
from minllm.model import Config, Transformer
from minllm.tokenizer import Tokenizer


def setup():
    torch.manual_seed(0)
    tok = Tokenizer(special_tokens=["<|endoftext|>", "<|im_start|>", "<|im_end|>"])
    model = Transformer(Config(vocab_size=tok.vocab_size, max_seq_len=96, dim=32, n_layers=2, n_heads=2, n_kv_heads=1, mlp_dim=64))
    conversations = [[{"role": "user", "content": text}] for text in ("hi", "a longer question", "ok")]
    return model, tok, conversations


def test_rollout_and_batch():
    model, tok, conversations = setup()
    prompts, replies, texts = rl.rollout(model, tok, conversations, samples=2, max_new_tokens=12)
    assert len(prompts) == len(replies) == len(texts) == 6
    assert prompts[0] == prompts[1] == chat.encode(tok, conversations[0]) and prompts[2] != prompts[0]
    end = tok.special_tokens["<|im_end|>"]
    for reply, text in zip(replies, texts):
        assert 1 <= len(reply) <= 12 and end not in reply[:-1]
        assert tok.decode(reply) == text + ("<|im_end|>" if reply[-1] == end else "")

    x, y, rows = rl.batch(prompts, replies, "cpu")
    assert x.shape == y.shape and x.shape[1] % 32 == 0
    for i, (p, r) in enumerate(zip(prompts, replies)):
        assert y[i][y[i] != -100].tolist() == r  # the targets of a row are its reply
        assert x[i, : len(p)].tolist() == p  # and the model reads the prompt first
        assert x[i, len(p) : len(p) + len(r) - 1].tolist() == r[:-1]
    assert rows.tolist() == [i for i, r in enumerate(replies) for _ in r]

    # The log-probability of a reply, token by token, is what a plain forward pass gives.
    logits = rl.target_logits(model, x, y)
    logp = rl.token_logprobs(logits, y)
    full = torch.log_softmax(model(x)[0], dim=-1)
    p, r = prompts[2], replies[2]
    expected = [full[2, len(p) - 1 + t, token].item() for t, token in enumerate(r)]
    assert torch.allclose(logp[rows == 2], torch.tensor(expected), atol=1e-5)
    assert torch.allclose(rl.per_reply(logp, rows, 6)[2], torch.tensor(expected).sum(), atol=1e-4)


def test_masked_tokens_are_not_targets():
    # A reply in which a tool answered: the tool's tokens are read, not learned.
    prompts, replies, masks = [[1, 2, 3], [4, 5]], [[6, 7, 8, 9, 10], [11, 12]], [[1, 1, 0, 0, 1], [1, 1]]
    x, y, rows = rl.batch(prompts, replies, "cpu", masks)
    assert y[0][y[0] != -100].tolist() == [6, 7, 10] and y[1][y[1] != -100].tolist() == [11, 12]
    assert x[0, :7].tolist() == [1, 2, 3, 6, 7, 8, 9]  # but the model still reads them
    assert rows.tolist() == [0, 0, 0, 1, 1]
    model, _, _ = setup()
    logp = rl.reply_logprobs(model, prompts, replies, "cpu", masks=masks)
    assert [len(l) for l in logp] == [3, 2]
    full = rl.reply_logprobs(model, prompts, replies, "cpu")
    assert torch.allclose(logp[0], full[0][[0, 1, 4]], atol=1e-6)


def test_policy_gradient_makes_rewarded_replies_likelier():
    model, tok, conversations = setup()
    prompts, replies, _ = rl.rollout(model, tok, conversations, samples=4, max_new_tokens=8)
    x, y, rows = rl.batch(prompts, replies, "cpu")
    rewards = torch.tensor([1.0, 0.0, 0.0, 1.0] * 3)
    advantages = rl.group_advantages(rewards, group_size=4)
    assert torch.allclose(advantages.view(3, 4).sum(dim=1), torch.zeros(3), atol=1e-4)

    before = rl.per_reply(rl.token_logprobs(rl.target_logits(model, x, y), y), rows, 12).detach()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    for _ in range(5):
        loss = rl.policy_loss(rl.token_logprobs(rl.target_logits(model, x, y), y), advantages, rows)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    after = rl.per_reply(rl.token_logprobs(rl.target_logits(model, x, y), y), rows, 12).detach()
    change = after - before
    assert change[rewards == 1].mean() > 0 > change[rewards == 0].mean()


def test_group_advantages():
    rewards = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.4, 0.6, 0.8])
    a = rl.group_advantages(rewards, group_size=4).view(3, 4)
    assert a[0, 0] > 0 and (a[0, 1:] < 0).all()
    assert torch.equal(a[1], torch.zeros(4))  # everyone got it right: nothing to prefer
    assert torch.allclose(a[2], -a[2].flip(0), atol=1e-5) and a[2].std() > 0.9


def test_ppo_clipping_and_kl():
    logp_old = torch.log(torch.tensor([0.5, 0.5, 0.5, 0.5]))
    rows, advantages = torch.tensor([0, 0, 1, 1]), torch.tensor([1.0, -1.0])
    # With no movement yet, the clipped objective has the gradient of REINFORCE.
    logp = logp_old.clone().requires_grad_()
    rl.policy_loss(logp, advantages, rows, logp_old).backward()
    reinforce = logp_old.clone().requires_grad_()
    rl.policy_loss(reinforce, advantages, rows).backward()
    assert torch.allclose(logp.grad, reinforce.grad)

    # Once a token has moved past the clip in the favored direction, pushing it further earns nothing.
    moved = torch.log(torch.tensor([0.9, 0.5, 0.1, 0.5])).requires_grad_()
    rl.policy_loss(moved, advantages, rows, logp_old, clip=0.2).backward()
    assert moved.grad[0] == 0 and moved.grad[2] == 0 and moved.grad[1] != 0

    logits = torch.randn(5, 11)
    assert torch.allclose(rl.kl(logits, logits), torch.zeros(5), atol=1e-6)
    assert (rl.kl(logits, torch.randn(5, 11)) > 0).all()


def test_update_matches_one_big_batch_and_respects_the_kl_penalty():
    import copy

    model, tok, conversations = setup()
    reference_model = copy.deepcopy(model)
    reference = lambda x, y: rl.target_logits(reference_model, x, y)
    prompts, replies, _ = rl.rollout(model, tok, conversations, samples=4, max_new_tokens=8)
    advantages = rl.group_advantages(torch.tensor([1.0, 0.0, 0.0, 1.0] * 3), group_size=4)
    old = rl.reply_logprobs(model, prompts, replies, "cpu")
    assert [len(o) for o in old] == [len(r) for r in replies]

    # Chunked or not, the step is the same.
    a, b = copy.deepcopy(model), copy.deepcopy(model)
    for m, chunk in ((a, 12), (b, 5)):
        rl.update(m, torch.optim.SGD(m.parameters(), lr=0.1), prompts, replies, advantages, "cpu", old_logp=old, chunk=chunk)
    assert all(torch.allclose(p, q, atol=1e-6) for p, q in zip(a.parameters(), b.parameters()))

    # A KL penalty holds the policy closer to where it started.
    def moved(kl_coef):
        m = copy.deepcopy(model)
        optimizer = torch.optim.SGD(m.parameters(), lr=0.3)
        for _ in range(8):
            stats = rl.update(m, optimizer, prompts, replies, advantages, "cpu", kl_coef=kl_coef, reference=reference)
        return stats["kl"]

    assert 0 < moved(kl_coef=5.0) < moved(kl_coef=0.0)
