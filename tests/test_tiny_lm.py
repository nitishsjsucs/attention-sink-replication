import pytest
import torch

from sink.tiny_lm import Config, TinyLM


@pytest.mark.parametrize("kind", ["softmax", "sigmoid", "softmax_kbias"])
def test_attention_is_causal(kind):
    torch.manual_seed(0)
    model = TinyLM(Config(vocab_size=100, d_model=32, n_layers=2, n_heads=2, max_len=16, attention=kind))
    ids = torch.randint(0, 100, (1, 16))
    logits, attn = model(ids, return_attn=True)
    assert attn.shape == (1, 2, 2, 16, 16)
    assert torch.all(attn.triu(1) == 0)
    # Changing a later token must not change earlier logits.
    ids2 = ids.clone()
    ids2[0, -1] = (ids[0, -1] + 1) % 100
    assert torch.allclose(model(ids2)[0, :-1], logits[0, :-1], atol=1e-5)


def test_softmax_rows_sum_to_one_and_kbias_rows_do_not_exceed_one():
    torch.manual_seed(0)
    ids = torch.randint(0, 100, (1, 8))
    base = dict(vocab_size=100, d_model=32, n_layers=1, n_heads=2, max_len=8)
    _, a = TinyLM(Config(**base, attention="softmax"))(ids, return_attn=True)
    assert torch.allclose(a.sum(-1), torch.ones(1, 1, 2, 8), atol=1e-5)
    _, a = TinyLM(Config(**base, attention="softmax_kbias"))(ids, return_attn=True)
    assert torch.all(a.sum(-1) < 1)


def test_fused_softmax_matches_explicit_path():
    torch.manual_seed(0)
    model = TinyLM(Config(vocab_size=100, d_model=32, n_layers=2, n_heads=2, max_len=16)).eval()
    ids = torch.randint(0, 100, (2, 16))
    fused = model(ids)
    explicit, _ = model(ids, return_attn=True)
    assert torch.allclose(fused, explicit, atol=1e-5)
