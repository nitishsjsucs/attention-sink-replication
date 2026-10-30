"""Experiments 1-3: attention sink in pretrained open LMs.

For each model and input type (natural text, random tokens, one repeated
token), records Sink_k for the first positions plus the L2 norms of hidden
states, keys and values per position. Results go to results/<model>.json.

    uv run python experiments/open_models.py --models gpt2 EleutherAI/pythia-70m
"""

import argparse
import json
import time
from pathlib import Path

import torch

from sink import data
from sink.metrics import sink_from_alpha
from sink.probe import load, pick_device, probe

DEFAULT_MODELS = [
    "openai-community/gpt2",
    "openai-community/gpt2-medium",
    "EleutherAI/pythia-14m",
    "EleutherAI/pythia-70m",
    "EleutherAI/pythia-160m",
    "EleutherAI/pythia-410m",
    "facebook/opt-125m",
    "facebook/opt-350m",
    "HuggingFaceTB/SmolLM2-135M",
    "HuggingFaceTB/SmolLM2-360M",
]
RESULTS = Path(__file__).resolve().parent.parent / "results"


def run(model_id: str, n: int, T: int, epsilon: float, device) -> dict:
    model, tokenizer = load(model_id, device)
    vocab = len(tokenizer)
    inputs = {
        "natural": data.natural(tokenizer, n, T),
        "random": data.random_tokens(vocab, n, T),
        "repeated": data.repeated_token(vocab, n, T),
    }
    out = {"model": model_id, "T": T, "epsilon": epsilon, "n": n, "inputs": {}}
    for name, ids in inputs.items():
        r = probe(model, ids, device)
        sink = sink_from_alpha(r.alpha, epsilon)
        out["inputs"][name] = {
            "sink_first8": sink[:8].tolist(),
            "hidden_norm": r.hidden_norm.tolist(),
            "key_norm": r.key_norm.tolist(),
            "value_norm": r.value_norm.tolist(),
            # Per-layer Sink_1, to see where in depth the sink forms.
            "sink1_by_layer": ((r.alpha[..., 0] > epsilon).float().mean(dim=(0, 2)) * 100).tolist(),
        }
        print(f"  {name:9s} Sink_1 = {sink[0]:5.1f}%   Sink_2 = {sink[1]:5.1f}%")
    del model
    if device.type == "mps":
        torch.mps.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--n", type=int, default=170, help="sequences per input type (the probe set has 170)")
    ap.add_argument("--T", type=int, default=64)
    ap.add_argument("--epsilon", type=float, default=0.3)
    args = ap.parse_args()

    device = pick_device()
    RESULTS.mkdir(exist_ok=True)
    for model_id in args.models:
        print(f"{model_id} on {device}")
        t0 = time.time()
        result = run(model_id, args.n, args.T, args.epsilon, device)
        path = RESULTS / f"{model_id.split('/')[-1]}.json"
        path.write_text(json.dumps(result))
        print(f"  saved {path.name} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
