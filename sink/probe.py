"""Run a Hugging Face causal LM over token sequences and collect attention
maps plus L2 norms of hidden states, keys and values per position."""

from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from sink.metrics import importance_scores


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load(model_id: str, device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # Eager attention is the only implementation that returns attention maps.
    model = AutoModelForCausalLM.from_pretrained(
        model_id, attn_implementation="eager", dtype=torch.float32
    ).to(device)
    model.eval()
    return model, tokenizer


def _layer_kv(cache, layer: int):
    if hasattr(cache, "layers"):  # transformers >= 4.56
        return cache.layers[layer].keys, cache.layers[layer].values
    if hasattr(cache, "key_cache"):
        return cache.key_cache[layer], cache.value_cache[layer]
    return cache[layer]


@dataclass
class ProbeResult:
    alpha: torch.Tensor  # (samples, layers, heads, T) importance scores
    hidden_norm: torch.Tensor  # (layers + 1, T) mean over samples
    key_norm: torch.Tensor  # (layers, T) mean over samples and kv heads
    value_norm: torch.Tensor  # (layers, T)


@torch.no_grad()
def probe(model, input_ids: torch.Tensor, device: torch.device) -> ProbeResult:
    """input_ids: (samples, T). Sequences run one at a time, so the sample
    count is bounded by time, not memory."""
    alphas, hiddens, keys, values = [], [], [], []
    for ids in input_ids:
        out = model(
            ids[None].to(device),
            output_attentions=True,
            output_hidden_states=True,
            use_cache=True,
        )
        # out.attentions: tuple over layers of (1, heads, T, T)
        attn = torch.cat(out.attentions, dim=0)  # (layers, heads, T, T)
        alphas.append(importance_scores(attn.float()).cpu())
        hiddens.append(torch.stack([h[0].norm(dim=-1) for h in out.hidden_states]).cpu())
        n_layers = len(out.attentions)
        kv = [_layer_kv(out.past_key_values, l) for l in range(n_layers)]
        keys.append(torch.stack([k[0].norm(dim=-1).mean(0) for k, _ in kv]).cpu())
        values.append(torch.stack([v[0].norm(dim=-1).mean(0) for _, v in kv]).cpu())
    return ProbeResult(
        alpha=torch.stack(alphas),
        hidden_norm=torch.stack(hiddens).mean(0),
        key_norm=torch.stack(keys).mean(0),
        value_norm=torch.stack(values).mean(0),
    )

