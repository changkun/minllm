"""
The reward model: a network that scores a reply the way the annotators would.

It is the transformer of stage 2 with one change. Instead of predicting the
next token at every position, it reads the whole conversation and maps the
vector of its last token to a single number. It starts from the weights of the
language model, which already understands stories, and is trained on
comparisons. See docs/08-rlhf.md.
"""

import json
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from minllm import chat, checkpoint


class RewardModel(nn.Module):
    def __init__(self, body):
        super().__init__()
        self.body = body  # a Transformer, used up to its final norm
        self.head = nn.Linear(body.cfg.dim, 1)
        nn.init.zeros_(self.head.weight)  # every reply starts with reward 0
        nn.init.zeros_(self.head.bias)

    def forward(self, x, lengths):
        """x: (batch, seq) token ids, padded on the right. Returns one reward per row."""
        h = self.body.hidden(x)
        last = h[torch.arange(len(x), device=x.device), lengths - 1]  # the vector of each row's last real token
        return self.head(last.float()).squeeze(-1)


def encode(tok, conversations, replies, device, max_len=None):
    """Conversations with their replies as a padded batch (x, lengths)."""
    rows = [chat.encode(tok, m + [{"role": "assistant", "content": r}], add_generation_prompt=False)[:-1] for m, r in zip(conversations, replies)]
    if max_len:
        rows = [row[-max_len:] for row in rows]
    width = math.ceil(max(len(row) for row in rows) / 32) * 32
    x = torch.zeros(len(rows), width, dtype=torch.long)
    for i, row in enumerate(rows):
        x[i, : len(row)] = torch.tensor(row)
    return x.to(device), torch.tensor([len(row) for row in rows], device=device)


def comparison_loss(preferred, other):
    """
    Bradley-Terry: the probability that the annotator prefers a reply is the
    sigmoid of the difference of the two rewards. The loss is the negative
    log of the probability given to the choice that was actually made.
    """
    return -F.logsigmoid(preferred - other).mean()


@torch.no_grad()
def score(model, tok, conversations, replies, batch_size=128):
    """The reward of every reply, as a list of floats."""
    device = next(model.parameters()).device
    out = []
    for i in range(0, len(replies), batch_size):
        x, lengths = encode(tok, conversations[i : i + batch_size], replies[i : i + batch_size], device, model.body.cfg.max_seq_len)
        out += model(x, lengths).tolist()
    return out


def load(dir, device):
    """The reward model saved by scripts/train_reward.py, with its tokenizer."""
    with open(f"{dir}/meta.json") as f:
        meta = json.load(f)
    body, tok, _ = checkpoint.load(meta["base"], device)
    model = RewardModel(body).to(device)
    model.load_state_dict(torch.load(f"{dir}/reward.pt", map_location=device, weights_only=True)["model"])
    return model.eval(), tok
