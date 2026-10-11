"""
Reinforcement learning for a language model.

The model is a policy. The text so far is the state, the next token is the
action, and a finished reply is an episode that earns one reward at its end.
Training makes the tokens of rewarded replies more likely:

    loss = -mean(advantage * log p(token))

where the advantage of a reply is its reward minus what was expected. The
gradient of this loss is the policy gradient. Where the reward comes from is
the only difference between the two stages that use this file: a model
trained on human preferences in stage 8, a program that checks the answer in
stage 9. See docs/08-rlhf.md and docs/09-rlvr.md.
"""

import math

import torch
import torch.nn.functional as F

from minllm import chat
from minllm.common import free_memory
from minllm.inference import stream


@torch.no_grad()
def rollout(model, tok, conversations, samples=1, max_new_tokens=200, temperature=1.0):
    """
    Let the model reply to every conversation `samples` times. Returns, per
    reply, the prompt's token ids, the reply's token ids and its text. A reply
    that the model ended itself carries its <|im_end|>: stopping is an action
    like any other, and has to be learned like any other.
    """
    end = tok.special_tokens[chat.IM_END]
    prompts = [chat.encode(tok, messages) for messages in conversations for _ in range(samples)]
    steps = torch.stack(list(stream(model, prompts, max_new_tokens, temperature=temperature, stop=[end])), dim=1)
    replies = [row[: row.index(-1)] + [end] if -1 in row else row for row in steps.tolist()]
    texts = [tok.decode(r[:-1] if r[-1] == end else r) for r in replies]
    return prompts, replies, texts


def batch(prompts, replies, device):
    """
    Inputs x and targets y for running the model over prompts and their
    replies. Only reply tokens are targets. rows says, for every target in
    reading order, which reply it belongs to.
    """
    width = math.ceil(max(len(p) + len(r) for p, r in zip(prompts, replies)) / 32) * 32
    x = torch.zeros(len(prompts), width, dtype=torch.long)
    y = torch.full((len(prompts), width), -100, dtype=torch.long)
    for i, (p, r) in enumerate(zip(prompts, replies)):
        tokens = p + r
        x[i, : len(tokens) - 1] = torch.tensor(tokens[:-1])
        y[i, len(p) - 1 : len(tokens) - 1] = torch.tensor(r)
    x, y = x.to(device), y.to(device)
    return x, y, (y != -100).nonzero()[:, 0]


def target_logits(model, x, y, hidden=None):
    """
    The logits at every target position, (targets, vocab). hidden stands in
    for model.hidden when given: a compiled version, or one under autocast.
    """
    h = (hidden or model.hidden)(x)
    return model.lm_head(h[y != -100].to(model.lm_head.weight.dtype)).float()


def token_logprobs(logits, y):
    """log p of each target token under logits, one number per target."""
    return -F.cross_entropy(logits, y[y != -100], reduction="none")


def kl(logits, reference_logits):
    """KL(policy || reference) at every target position: how far each next-token distribution has moved."""
    logp, logq = F.log_softmax(logits, dim=-1), F.log_softmax(reference_logits, dim=-1)
    return (logp.exp() * (logp - logq)).sum(dim=-1)


def per_reply(values, rows, n):
    """Sum per-token values into one number per reply."""
    return torch.zeros(n, device=values.device, dtype=values.dtype).index_add(0, rows, values)


def policy_loss(logp, advantages, rows, old_logp=None, clip=0.2):
    """
    The policy-gradient loss over the target tokens of a batch. Every token of
    a reply shares the reply's advantage.

    Without old_logp this is REINFORCE: -advantage * log p. It is right for
    one gradient step on fresh samples. With old_logp, the log-probabilities
    the tokens had when they were sampled, it is PPO's clipped objective,
    which allows several steps on the same samples: the ratio p / p_old says
    how far the policy has moved since, and clipping it removes the incentive
    to move further than 1 +- clip in the direction the advantage favors.
    """
    advantage = advantages[rows]
    if old_logp is None:
        return -(advantage * logp).mean()
    ratio = (logp - old_logp).exp()
    return -torch.min(ratio * advantage, ratio.clamp(1 - clip, 1 + clip) * advantage).mean()


def group_advantages(rewards, group_size, eps=1e-6):
    """
    GRPO's advantage: a reply is compared with the other replies to the same
    prompt. Subtracting the group's mean removes how hard the prompt is, and
    dividing by its standard deviation puts all prompts on one scale. A group
    whose replies all earned the same reward teaches nothing, and gets zeros.
    """
    groups = rewards.view(-1, group_size)
    return ((groups - groups.mean(dim=1, keepdim=True)) / (groups.std(dim=1, keepdim=True) + eps)).view(-1)


@torch.no_grad()
def reply_logprobs(model, prompts, replies, device, chunk=64, hidden=None):
    """For every reply, the log-probability of each of its tokens under the model, as a list of 1-D tensors."""
    out = []
    for i in range(0, len(prompts), chunk):
        x, y, rows = batch(prompts[i : i + chunk], replies[i : i + chunk], device)
        logp = token_logprobs(target_logits(model, x, y, hidden), y)
        out += [logp[rows == j] for j in range(len(x))]
    return out


def update(model, optimizer, prompts, replies, advantages, device, old_logp=None, clip=0.2, kl_coef=0.0, reference=None, chunk=64, hidden=None):
    """
    One optimizer step on a set of sampled replies. The loss is the policy
    loss plus kl_coef times the KL divergence from a reference policy, both
    averaged over all reply tokens. reference(x, y) gives the reference
    policy's logits at the target positions.

    The replies are processed a chunk at a time and the gradients add up, so
    the step can cover more replies than fit in memory at once.
    """
    total = sum(len(r) for r in replies)
    optimizer.zero_grad(set_to_none=True)
    stats = {"loss": 0.0, "kl": 0.0}
    for i in range(0, len(prompts), chunk):
        x, y, rows = batch(prompts[i : i + chunk], replies[i : i + chunk], device)
        logits = target_logits(model, x, y, hidden)
        logp = token_logprobs(logits, y)
        old = torch.cat(old_logp[i : i + chunk]) if old_logp is not None else None
        share = len(logp) / total  # this chunk's part of the mean over all tokens
        loss = policy_loss(logp, advantages[i : i + chunk], rows, old, clip) * share
        if reference is not None:
            with torch.no_grad():
                reference_logits = reference(x, y)
            divergence = kl(logits, reference_logits).mean()
            loss = loss + kl_coef * divergence * share
            stats["kl"] += divergence.item() * share
        loss.backward()
        stats["loss"] += loss.item()
        del logits, logp, loss
        free_memory(device)  # every chunk has its own shapes, and the pool would keep them all
    stats["grad_norm"] = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0).item()
    optimizer.step()
    return stats
