"""Attention sink metric from Gu et al., ICLR 2025 (Section 3.1).

For head h in layer l, the importance score of key position k is the mean
attention that every query at or after k puts on k:

    alpha_k^{l,h} = 1 / (T - k + 1) * sum_{i >= k} A^{l,h}[i, k]

A head "sinks" on k when alpha_k > epsilon. Sink_k^epsilon is the share of
heads (over layers, heads and samples) that sink on k, in percent. The paper
reports Sink_1 with epsilon = 0.3 and T = 64.
"""

import torch


def importance_scores(attn: torch.Tensor) -> torch.Tensor:
    """attn: (..., T, T) causal attention probabilities, rows are queries.

    Returns (..., T): alpha_k for every key position k.
    """
    T = attn.shape[-1]
    # Query rows i >= k can see key k, so column k has T - k nonzero entries
    # (0-indexed). This matches `ratios = arange(T, 0, -1)` in the official code.
    n_queries = torch.arange(T, 0, -1, device=attn.device, dtype=attn.dtype)
    return attn.sum(dim=-2) / n_queries


def sink_ratio(attn: torch.Tensor, epsilon: float = 0.3) -> torch.Tensor:
    """attn: (samples, layers, heads, T, T). Returns (T,) Sink_k in percent."""
    return sink_from_alpha(importance_scores(attn), epsilon)


def sink_from_alpha(alpha: torch.Tensor, epsilon: float = 0.3) -> torch.Tensor:
    """alpha: (samples, layers, heads, T). Returns (T,) Sink_k in percent."""
    return (alpha > epsilon).float().mean(dim=(0, 1, 2)) * 100
