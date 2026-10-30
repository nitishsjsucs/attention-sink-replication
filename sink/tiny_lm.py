"""A small LLaMA-style decoder (RMSNorm, RoPE, SwiGLU, pre-norm) with
swappable attention, for the paper's Section 7 ablations.

Attention variants:
  softmax        standard scaled dot-product attention
  sigmoid        sigmoid(q.k / sqrt(d)) with no normalization over keys,
                 the variant the paper reports removes the sink
  softmax_kbias  softmax plus a learnable key per head with a zero value,
                 so the head has somewhere to dump attention that is not a
                 real token (the paper's "learnable sink token / key bias")
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    vocab_size: int = 50257
    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 4
    ffn_mult: float = 8 / 3
    max_len: int = 256
    attention: str = "softmax"


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return self.weight * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


def rope_tables(head_dim, max_len, base=10000.0):
    inv = 1.0 / base ** (torch.arange(0, head_dim, 2).float() / head_dim)
    freqs = torch.outer(torch.arange(max_len).float(), inv)
    return freqs.cos(), freqs.sin()


def apply_rope(x, cos, sin):
    x1, x2 = x[..., ::2], x[..., 1::2]
    out = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return out.flatten(-2)


class Attention(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.h = cfg.n_heads
        self.dh = cfg.d_model // cfg.n_heads
        self.kind = cfg.attention
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.out = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        if self.kind == "softmax_kbias":
            self.sink_key = nn.Parameter(torch.zeros(self.h, 1, self.dh))

    def forward(self, x, cos, sin, return_attn=False):
        B, T, _ = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        q, k = apply_rope(q, cos[:T], sin[:T]), apply_rope(k, cos[:T], sin[:T])
        if self.kind == "softmax" and not return_attn:
            # Fused kernel for training; same math as the explicit path below.
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            return self.out(y.transpose(1, 2).reshape(B, T, -1)), None
        scores = q @ k.transpose(-1, -2) / math.sqrt(self.dh)  # (B, h, T, T)
        causal = torch.ones(T, T, dtype=torch.bool, device=x.device).tril()

        if self.kind == "softmax":
            attn = scores.masked_fill(~causal, float("-inf")).softmax(-1)
            y = attn @ v
        elif self.kind == "sigmoid":
            attn = torch.sigmoid(scores) * causal
            y = attn @ v
        elif self.kind == "softmax_kbias":
            extra = (q @ self.sink_key.transpose(-1, -2)) / math.sqrt(self.dh)  # (B, h, T, 1)
            full = torch.cat([extra, scores.masked_fill(~causal, float("-inf"))], dim=-1)
            p = full.softmax(-1)
            attn = p[..., 1:]  # the zero-value sink key contributes nothing to y
            y = attn @ v
        else:
            raise ValueError(self.kind)

        y = y.transpose(1, 2).reshape(B, T, -1)
        return self.out(y), attn


class Block(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        hidden = int(cfg.ffn_mult * cfg.d_model)
        self.n1, self.n2 = RMSNorm(cfg.d_model), RMSNorm(cfg.d_model)
        self.attn = Attention(cfg)
        self.gate = nn.Linear(cfg.d_model, hidden, bias=False)
        self.up = nn.Linear(cfg.d_model, hidden, bias=False)
        self.down = nn.Linear(hidden, cfg.d_model, bias=False)

    def forward(self, x, cos, sin, return_attn=False):
        a, attn = self.attn(self.n1(x), cos, sin, return_attn)
        x = x + a
        h = self.n2(x)
        return x + self.down(F.silu(self.gate(h)) * self.up(h)), attn


class TinyLM(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.norm = RMSNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        cos, sin = rope_tables(cfg.d_model // cfg.n_heads, cfg.max_len)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, ids, return_attn=False):
        x = self.embed(ids)
        attns = []
        for block in self.blocks:
            x, attn = block(x, self.cos, self.sin, return_attn)
            if return_attn:
                attns.append(attn)
        logits = self.head(self.norm(x))
        return (logits, torch.stack(attns, dim=1)) if return_attn else logits  # attn: (B, L, h, T, T)
