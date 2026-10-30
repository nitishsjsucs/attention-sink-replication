"""Input sequences for the probes.

Natural text comes from the paper's own probe set (datasets/probe_valid.jsonl
in sail-sg/Attention-Sink, MIT licensed), downloaded on first use. The paper
also probes random-token and repeated-token inputs (Section 3.2), which are
generated here.
"""

import json
import urllib.request
from pathlib import Path

import torch

PROBE_URL = (
    "https://raw.githubusercontent.com/sail-sg/Attention-Sink/main/datasets/probe_valid.jsonl"
)
CACHE = Path(__file__).resolve().parent.parent / "data" / "probe_valid.jsonl"


def probe_texts() -> list[str]:
    if not CACHE.exists():
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(PROBE_URL, CACHE)
    with CACHE.open() as f:
        return [json.loads(line)["text"] for line in f]


def natural(tokenizer, n: int, T: int = 64) -> torch.Tensor:
    """First T tokens of the first n probe texts that are at least T long.

    Like the official script, no BOS token is prepended: whatever the text's
    first token is sits at position 1.
    """
    rows = []
    for text in probe_texts():
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if len(ids) >= T:
            rows.append(ids[:T])
        if len(rows) == n:
            break
    return torch.tensor(rows)


def random_tokens(vocab_size: int, n: int, T: int = 64, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, vocab_size, (n, T), generator=g)


def repeated_token(vocab_size: int, n: int, T: int = 64, seed: int = 0) -> torch.Tensor:
    """Each sequence is one random token repeated T times."""
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, vocab_size, (n, 1), generator=g).expand(n, T).clone()
