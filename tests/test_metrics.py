import torch

from sink.metrics import importance_scores, sink_ratio


def causal_uniform(T):
    """Each query attends uniformly to itself and all earlier keys."""
    mask = torch.tril(torch.ones(T, T))
    return mask / mask.sum(-1, keepdim=True)


def all_on_first(T):
    a = torch.zeros(T, T)
    a[:, 0] = 1.0
    return a


def test_all_attention_on_first_token_scores_one():
    alpha = importance_scores(all_on_first(8))
    assert torch.allclose(alpha[0], torch.tensor(1.0))
    assert torch.all(alpha[1:] == 0)


def test_uniform_causal_attention_matches_harmonic_mean():
    T = 6
    alpha = importance_scores(causal_uniform(T))
    # Column 0 gets 1/(i+1) from query i; averaged over all T queries.
    expected = sum(1 / (i + 1) for i in range(T)) / T
    assert torch.isclose(alpha[0], torch.tensor(expected))
    # The last key is only seen by the last query, with weight 1/T.
    assert torch.isclose(alpha[-1], torch.tensor(1 / T))


def test_sink_ratio_counts_share_of_heads():
    T = 8
    heads = torch.stack([all_on_first(T), causal_uniform(T), causal_uniform(T), all_on_first(T)])
    attn = heads.view(1, 1, 4, T, T)
    ratio = sink_ratio(attn, epsilon=0.3)
    assert torch.isclose(ratio[0], torch.tensor(100.0))  # uniform alpha_1 = 0.34 > 0.3
    ratio = sink_ratio(attn, epsilon=0.5)
    assert torch.isclose(ratio[0], torch.tensor(50.0))
