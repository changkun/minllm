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
later be loaded into this same code. See docs/02-model.md. The latent
attention of DeepSeek's models can take the place of GQA: see
LatentAttention and docs/02c-latent-attention.md.
"""

import math
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
    qkv_bias: bool = False  # Qwen2.5 adds a bias to the query, key and value projections
    n_experts: int = 0  # above 0, the MLP of every block is a mixture of this many experts
    n_active: int = 2  # the experts that process each token
    # Positions beyond those trained on: {"type": "linear" | "ntk" | "yarn", "factor": s,
    # "original": the length trained on}. See rope_table and docs/02b-context.md.
    rope_scaling: dict | None = None
    kv_latent: int = 0  # above 0, attention is latent attention, and the cache keeps a latent of this width
    rope_dim: int = 32  # with kv_latent: the rotary part of every query and key, the key's shared by all heads


class RMSNorm(nn.Module):
    """Rescale each vector to unit root-mean-square, then apply a learned gain."""

    def __init__(self, dim, eps):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


def rope_table(head_dim, seq_len, theta, scaling=None):
    """
    Rotation angles for rotary position embeddings. Dimension pair i turns by
    theta^(-2i / head_dim) radians per token: the first pairs spin fast and
    resolve neighbouring positions, the last ones spin slowly and resolve
    distant ones.

    A model trained on windows of L tokens has seen every pair turn only as
    far as L tokens turn it. The slow pairs have not finished one turn by
    then, and further on they reach angles the model never saw. scaling
    stretches the table by a factor s so that s * L tokens fit:

      linear  every pair turns s times slower: position interpolation
              (Chen et al., 2023). Distant tokens look as they did, and
              neighbours look s times closer than they are.
      ntk     theta grows by s^(d / (d - 2)) (bloc97, 2023): the slowest
              pair turns s times slower, the fastest as before, and the
              ones between in proportion.
      yarn    each pair by how many turns it makes within L (Peng et al.,
              2023): those that turn many times are left as they are, those
              that turn less than once are interpolated, a ramp lies
              between, and attention is sharpened by 0.1 ln s + 1 to make
              up for the flatter scores over more tokens. As transformers
              computes it, so that an export loads there unchanged.
    """
    freqs = theta ** (-torch.arange(0, head_dim, 2).float() / head_dim)
    gain = 1.0
    if scaling:
        s, kind = scaling["factor"], scaling["type"]
        if kind == "linear":
            freqs = freqs / s
        elif kind == "ntk":
            freqs = (theta * s ** (head_dim / (head_dim - 2))) ** (-torch.arange(0, head_dim, 2).float() / head_dim)
        elif kind == "yarn":
            # the pair index at which a pair turns `turns` times within the original window
            index = lambda turns: head_dim * math.log(scaling["original"] / (turns * 2 * math.pi)) / (2 * math.log(theta))
            low, high = max(math.floor(index(32)), 0), min(math.ceil(index(1)), head_dim - 1)
            ramp = ((torch.arange(head_dim // 2).float() - low) / max(high - low, 1e-3)).clamp(0, 1)  # 0: as before, 1: interpolated
            freqs = freqs * (1 - ramp) + freqs / s * ramp
            gain = 0.1 * math.log(s) + 1.0
        else:
            raise ValueError(f"unknown rope scaling {kind}")
    angles = torch.outer(torch.arange(seq_len).float(), freqs)  # (seq_len, head_dim / 2)
    return angles.cos() * gain, angles.sin() * gain


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
        self.q_proj = nn.Linear(cfg.dim, cfg.n_heads * self.head_dim, bias=cfg.qkv_bias)
        self.k_proj = nn.Linear(cfg.dim, cfg.n_kv_heads * self.head_dim, bias=cfg.qkv_bias)
        self.v_proj = nn.Linear(cfg.dim, cfg.n_kv_heads * self.head_dim, bias=cfg.qkv_bias)
        self.o_proj = nn.Linear(cfg.n_heads * self.head_dim, cfg.dim, bias=False)

    def forward(self, x, cos, sin, mask=None, cache=None):
        B, T, _ = x.shape
        # (B, T, dim) -> (B, heads, T, head_dim)
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if cache is not None:
            # Generation: x holds only the new tokens. Store their keys and
            # values, and attend over those of every token so far.
            k, v = cache.append(k, v)
        # Each key/value head serves a group of n_heads / n_kv_heads query heads.
        groups = self.n_heads // self.n_kv_heads
        k, v = k.repeat_interleave(groups, dim=1), v.repeat_interleave(groups, dim=1)
        # softmax(q k^T / sqrt(head_dim)) v, where a token sees only itself and
        # the past. Without an explicit mask the queries and keys line up one to
        # one, and the library applies the causal mask itself.
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=mask is None and T > 1)
        return self.o_proj(y.transpose(1, 2).reshape(B, T, -1))


class LatentAttention(nn.Module):
    """
    Multi-head latent attention, as in DeepSeek-V2 and V3 (DeepSeek-AI, 2024,
    arXiv:2405.04434). Each token is compressed into a latent of kv_latent
    numbers, and every head's key and value are projected up from it. The
    cache keeps only the latent and one small rotary key that all heads
    share, instead of a key and a value per key/value head.

    A query and a key have two parts. head_dim numbers of content, whose key
    comes from the latent, and rope_dim numbers that carry the position, whose
    key comes from the token and is rotated. The position needs a part of
    its own: a key rotated after the up-projection could not be folded into
    the query, as forward does when it decodes. See docs/02c-latent-attention.md.
    """

    def __init__(self, cfg):
        super().__init__()
        self.n_heads, self.head_dim, self.rope_dim, self.latent = cfg.n_heads, cfg.dim // cfg.n_heads, cfg.rope_dim, cfg.kv_latent
        self.q_proj = nn.Linear(cfg.dim, cfg.n_heads * (self.head_dim + self.rope_dim), bias=False)
        self.kv_down = nn.Linear(cfg.dim, self.latent + self.rope_dim, bias=False)  # the latent and the rotary key
        self.kv_norm = RMSNorm(self.latent, cfg.norm_eps)
        self.k_up = nn.Linear(self.latent, cfg.n_heads * self.head_dim, bias=False)
        self.v_up = nn.Linear(self.latent, cfg.n_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.n_heads * self.head_dim, cfg.dim, bias=False)
        self.absorb = True  # decode against the latent itself; off, every step projects the cache up again

    def forward(self, x, cos, sin, mask=None, cache=None):
        B, T, _ = x.shape
        H, D, R = self.n_heads, self.head_dim, self.rope_dim
        q = self.q_proj(x).view(B, T, H, D + R).transpose(1, 2)
        q, q_rot = q.split([D, R], dim=-1)
        latent, k_rot = self.kv_down(x).split([self.latent, R], dim=-1)
        latent, k_rot = self.kv_norm(latent)[:, None], k_rot[:, None]  # (B, 1, T, ...): one "head", shared by all
        q_rot, k_rot = apply_rope(q_rot, cos, sin), apply_rope(k_rot, cos, sin)
        if cache is not None:
            latent, k_rot = cache.append(latent, k_rot)  # the cache holds these two, and nothing per head
        S = latent.shape[2]
        scale = 1 / math.sqrt(D + R)

        if cache is not None and T == 1 and self.absorb:
            # One new token. Its query meets each key as q . (W_k c) = (W_k^T q) . c,
            # so the query is projected down into the latent once, instead of
            # every cached latent being projected up into keys. Likewise the
            # values: the weights average the latents, and W_v is applied to
            # the average. Every head then attends over the same S vectors of
            # latent + rope_dim numbers, so the heads are rows of one matrix
            # product, and the cache is read once for all of them.
            w_k, w_v = self.k_up.weight.view(H, D, self.latent), self.v_up.weight.view(H, D, self.latent)
            q = torch.cat([torch.einsum("bhtd,hdl->bhtl", q, w_k), q_rot], dim=-1).reshape(B, H * T, -1)
            scores = q @ torch.cat([latent, k_rot], dim=-1)[:, 0].transpose(1, 2) * scale  # (B, heads, S)
            if mask is not None:
                scores = scores.masked_fill(~mask.expand(B, H, T, S).reshape(B, H * T, S), float("-inf"))
            y = scores.float().softmax(dim=-1).to(latent.dtype) @ latent[:, 0]  # (B, heads, latent)
            y = torch.einsum("bhtl,hdl->bhtd", y.view(B, H, T, -1), w_v)
        else:
            # Training and prefill: every head's keys and values, projected up from the latents.
            k = self.k_up(latent[:, 0]).view(B, S, H, D).transpose(1, 2)
            v = self.v_up(latent[:, 0]).view(B, S, H, D).transpose(1, 2)
            q, k = torch.cat([q, q_rot], dim=-1), torch.cat([k, k_rot.expand(B, H, S, R)], dim=-1)
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=mask is None and T > 1, scale=scale)
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


class MoE(nn.Module):
    """
    A mixture of experts in place of the MLP: n_experts MLPs, of which each
    token uses only the n_active that a router picks for it. The block holds
    n_experts MLPs of parameters and spends n_active MLPs of compute per
    token. See docs/02a-moe.md.
    """

    def __init__(self, cfg):
        super().__init__()
        assert 0 < cfg.n_active <= cfg.n_experts
        self.n_active = cfg.n_active
        self.router = nn.Linear(cfg.dim, cfg.n_experts, bias=False)
        self.experts = nn.ModuleList(MLP(cfg) for _ in range(cfg.n_experts))
        self.capacity = None  # see forward
        self.share = self.balance = self.dropped = None  # set by every forward pass, for the training loop

    def route(self, x):
        """
        The weight of every expert for every token, (tokens, n_experts): zero
        for all but the token's n_active likeliest experts, whose weights sum
        to one.
        """
        probs = F.softmax(self.router(x).float(), dim=-1)
        top, chosen = probs.topk(self.n_active, dim=-1)
        weights = torch.zeros_like(probs).scatter(-1, chosen, top / top.sum(-1, keepdim=True))
        # share is the part of the work that went to each expert. balance is
        # the load-balancing loss of the Switch Transformer: 1 when every
        # expert gets the same share, and larger the more the router favors
        # the experts that are busy already.
        self.share = (weights > 0).float().mean(0) / self.n_active
        self.balance = len(self.experts) * (self.share * probs.mean(0)).sum()
        return weights

    def forward(self, x):
        shape = x.shape
        x = x.reshape(-1, shape[-1])  # (tokens, dim): an expert treats every token on its own
        weights = self.route(x)
        if self.capacity is not None:
            # Every expert takes the same number of tokens: capacity times its
            # share of an even split. The shapes then do not depend on the
            # data, which a compiled training step needs. An expert that more
            # tokens were routed to than it has room for drops the ones that
            # weigh it least, and one with room to spare fills it with tokens
            # of weight zero. With capacity n_experts / n_active every expert
            # takes every token, and nothing is dropped.
            room = min(len(x), int(self.capacity * len(x) * self.n_active / len(self.experts)))
            routed = (weights > 0).sum(0)
            self.dropped = (routed - room).clamp(min=0).sum() / routed.sum()
        out = torch.zeros_like(x)
        for i, expert in enumerate(self.experts):
            if self.capacity is None:
                rows = weights[:, i].nonzero().squeeze(-1)  # exactly the tokens that were routed to expert i
                w = weights[rows, i]
            else:
                w, rows = weights[:, i].topk(room)
            out = out.index_add(0, rows, (w[:, None] * expert(x[rows])).to(out.dtype))
        return out.view(shape)


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.attn = LatentAttention(cfg) if cfg.kv_latent else Attention(cfg)
        self.mlp_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.mlp = MoE(cfg) if cfg.n_experts else MLP(cfg)

    def forward(self, x, cos, sin, mask=None, cache=None):
        # Each sublayer reads a normalized copy of the residual stream and adds
        # its result back, so the stream itself is never squashed.
        x = x + self.attn(self.attn_norm(x), cos, sin, mask, cache)
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
        rotated = cfg.rope_dim if cfg.kv_latent else cfg.dim // cfg.n_heads  # the dimensions that carry the position
        cos, sin = rope_table(rotated, cfg.max_seq_len, cfg.rope_theta, cfg.rope_scaling)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        # Small random weights, so that the untrained model predicts a nearly
        # uniform distribution. The layers that write into the residual stream
        # start smaller still, since 2 * n_layers of them add up.
        for name, p in self.named_parameters():
            if p.dim() >= 2:
                residual = name.endswith(("o_proj.weight", "down_proj.weight"))
                nn.init.normal_(p, std=0.02 / (2 * cfg.n_layers) ** 0.5 if residual else 0.02)
            elif name.endswith("bias"):
                nn.init.zeros_(p)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def mixtures(self):
        """The mixture-of-experts layers, one per block, or none in a dense model."""
        return [layer.mlp for layer in self.layers if isinstance(layer.mlp, MoE)]

    def balance_loss(self):
        """How unevenly the last forward pass spread its tokens over the experts, averaged over the blocks."""
        return sum(m.balance for m in self.mixtures()) / len(self.layers)

    def hidden(self, idx, cache=None, mask=None):
        """
        Everything but the output layer: the final vector of every position,
        (batch, seq, dim). Fine-tuning calls this directly, so that it can
        apply the output layer only at the positions it trains on.

        With a cache (minllm/inference.py), idx holds only the tokens that
        come after the ones already cached. mask, if given, is boolean and
        broadcasts to (batch, heads, new tokens, all tokens): True where a
        token may attend.

        """
        T = idx.shape[1]
        pos = cache.pos if cache is not None else 0  # tokens that came before idx
        assert pos + T <= self.cfg.max_seq_len, f"{pos + T} tokens exceed max_seq_len {self.cfg.max_seq_len}"
        if mask is None and pos > 0 and T > 1:
            # Several new tokens after cached ones: each sees the whole cache,
            # and of the new tokens only itself and the earlier ones.
            mask = torch.ones(T, pos + T, dtype=torch.bool, device=idx.device).tril(pos)
        x = self.embed(idx)
        cos, sin = self.rope_cos[pos : pos + T], self.rope_sin[pos : pos + T]
        for i, layer in enumerate(self.layers):
            x = layer(x, cos, sin, mask, cache.layers[i] if cache is not None else None)
        return self.norm(x)

    def forward(self, idx, targets=None, cache=None, mask=None):
        """
        idx and targets are (batch, seq) token ids. Returns the logits
        (batch, seq, vocab) and, given targets, the mean cross-entropy loss.
        Positions whose target is -100 do not count towards the loss.
        """
        logits = self.lm_head(self.hidden(idx, cache, mask))
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100)
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Append max_new_tokens sampled tokens to idx (batch, seq). This is the
        simplest correct sampler: it reruns the whole sequence for every new
        token. minllm/inference.py does the same with a cache, and the tests
        use this one as the reference it has to match.
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
