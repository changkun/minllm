"""
A decoder-only transformer, in the style of Llama and Qwen.

    token ids -> embedding -> N x [attention, MLP] -> norm -> logits over the vocabulary

Compared to GPT-2 (and nanoGPT), the block uses the four changes that every
recent open model has adopted:

    RMSNorm      instead of LayerNorm
    RoPE         instead of learned position embeddings
    SwiGLU       instead of a GELU MLP
    GQA          grouped-query attention: query heads share key/value heads

Because the layout matches the published models, weights such as Qwen's can
later be loaded into this same code. See docs/02-model.md.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    vocab_size: int = 4096
    max_seq_len: int = 512
    dim: int = 384  # width of the residual stream
    n_layers: int = 8
    n_heads: int = 6  # query heads
    n_kv_heads: int = 2  # key/value heads, a divisor of n_heads
    mlp_dim: int = 1024  # hidden width of the MLP, about 8/3 * dim
    rope_theta: float = 10000.0
    norm_eps: float = 1e-6
    tie_embeddings: bool = True  # the input embedding and the output layer share one matrix


class RMSNorm(nn.Module):
    """Rescale each vector to unit root-mean-square, then apply a learned gain."""

    def __init__(self, dim, eps):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


def rope_table(head_dim, seq_len, theta):
    """
    Rotation angles for rotary position embeddings. Dimension pair i turns by
    theta^(-2i / head_dim) radians per token: the first pairs spin fast and
    resolve neighbouring positions, the last ones spin slowly and resolve
    distant ones.
    """
    freqs = theta ** (-torch.arange(0, head_dim, 2).float() / head_dim)
    angles = torch.outer(torch.arange(seq_len).float(), freqs)  # (seq_len, head_dim / 2)
    return angles.cos(), angles.sin()


def apply_rope(x, cos, sin):
    """
    Rotate queries or keys by their position. x is (batch, heads, seq, head_dim);
    its two halves hold the two coordinates of head_dim / 2 points in the plane.
    Rotating both q and k makes their dot product depend only on how far apart
    the two tokens are.
    """
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        assert cfg.dim % cfg.n_heads == 0 and cfg.n_heads % cfg.n_kv_heads == 0
        self.n_heads, self.n_kv_heads = cfg.n_heads, cfg.n_kv_heads
        self.head_dim = cfg.dim // cfg.n_heads
        self.q_proj = nn.Linear(cfg.dim, cfg.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.dim, cfg.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.dim, cfg.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.n_heads * self.head_dim, cfg.dim, bias=False)

    def forward(self, x, cos, sin):
        B, T, _ = x.shape
        # (B, T, dim) -> (B, heads, T, head_dim)
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        # Each key/value head serves a group of n_heads / n_kv_heads query heads.
        groups = self.n_heads // self.n_kv_heads
        k, v = k.repeat_interleave(groups, dim=1), v.repeat_interleave(groups, dim=1)
        # softmax(q k^T / sqrt(head_dim)) v, where a token sees only itself and the past
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.o_proj(y.transpose(1, 2).reshape(B, T, -1))


class MLP(nn.Module):
    """SwiGLU: one projection gates the other, elementwise."""

    def __init__(self, cfg):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.dim, cfg.mlp_dim, bias=False)
        self.up_proj = nn.Linear(cfg.dim, cfg.mlp_dim, bias=False)
        self.down_proj = nn.Linear(cfg.mlp_dim, cfg.dim, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.attn = Attention(cfg)
        self.mlp_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.mlp = MLP(cfg)

    def forward(self, x, cos, sin):
        # Each sublayer reads a normalized copy of the residual stream and adds
        # its result back, so the stream itself is never squashed.
        x = x + self.attn(self.attn_norm(x), cos, sin)
        x = x + self.mlp(self.mlp_norm(x))
        return x


class Transformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.layers = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.embed.weight
        cos, sin = rope_table(cfg.dim // cfg.n_heads, cfg.max_seq_len, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        # Small random weights, so that the untrained model predicts a nearly
        # uniform distribution. The layers that write into the residual stream
        # start smaller still, since 2 * n_layers of them add up.
        for name, p in self.named_parameters():
            if p.dim() >= 2:
                residual = name.endswith(("o_proj.weight", "down_proj.weight"))
                nn.init.normal_(p, std=0.02 / (2 * cfg.n_layers) ** 0.5 if residual else 0.02)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx, targets=None):
        """
        idx and targets are (batch, seq) token ids. Returns the logits
        (batch, seq, vocab) and, given targets, the mean cross-entropy loss.
        Positions whose target is -100 do not count towards the loss.
        """
        T = idx.shape[1]
        assert T <= self.cfg.max_seq_len, f"sequence of {T} tokens exceeds max_seq_len {self.cfg.max_seq_len}"
        x = self.embed(idx)
        cos, sin = self.rope_cos[:T], self.rope_sin[:T]
        for layer in self.layers:
            x = layer(x, cos, sin)
        logits = self.lm_head(self.norm(x))
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100)
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Append max_new_tokens sampled tokens to idx (batch, seq). This is the
        simplest correct sampler: it reruns the whole sequence for every new
        token. The inference stage replaces it with one that caches its work.
        """
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -self.cfg.max_seq_len :])
            logits = logits[:, -1]  # the prediction for the next token
            if temperature == 0:
                next_id = logits.argmax(-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k is not None:
                    kth = torch.topk(logits, min(top_k, logits.size(-1))).values[:, [-1]]
                    logits = logits.masked_fill(logits < kth, float("-inf"))
                next_id = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx
