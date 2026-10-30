# Attention Sink Replication

A laptop-scale replication of **[When Attention Sink Emerges in Language Models: An Empirical View](https://arxiv.org/abs/2410.10781)**
(Gu, Pang, Du, Liu, Zhang, Du, Wang, Lin; ICLR 2025 Spotlight).

Autoregressive LMs put a large share of attention on the first token, whatever that token is. The paper
measures this "attention sink" across open models and then pretrains small LLaMA models to find which
training choices make it appear. This repo reimplements the measurement from scratch and reruns the
experiments that fit on one Apple Silicon machine (MPS) or a single consumer GPU.

Official code: [sail-sg/Attention-Sink](https://github.com/sail-sg/Attention-Sink). This repo shares no code
with it. The natural-text probe set (`datasets/probe_valid.jsonl`, MIT) is downloaded from there on
first run so the inputs match the paper's.

## The metric

For head *h* in layer *l*, the importance score of key position *k* is the mean attention that the
queries able to see *k* put on it:

$$\alpha_k^{l,h} = \frac{1}{T-k+1}\sum_{i=k}^{T} A^{l,h}_{i,k}$$

A head sinks on *k* when $\alpha_k > \epsilon$. **Sink_k** is the percentage of heads, over all layers,
heads and samples, that sink on *k*. As in the paper: ε = 0.3, T = 64. The implementation is in
[`sink/metrics.py`](sink/metrics.py) and checked against hand-computed cases in
[`tests/test_metrics.py`](tests/test_metrics.py).

## Plan

| # | Paper claim | Experiment here | Status |
|---|---|---|---|
| 1 | Sinks exist across open LMs and grow with scale (Sec. 3.1) | Sink_1 for GPT-2, Pythia 14M-410M, OPT, SmolLM2, Qwen2.5-0.5B on the paper's probe text | done |
| 2 | Sinks appear even on random tokens; repeated tokens behave differently (Sec. 3.2) | Same models on random-token and one-repeated-token inputs | done |
| 3 | The sink token has massive hidden-state and key norms but a small value norm, so it acts like a key bias (Sec. 4) | L2 norms of hidden states, keys and values per position and layer | done |
| 4 | Sigmoid attention without normalization prevents the sink; a learnable key bias absorbs it (Sec. 7) | Pretrain a ~30M LLaMA-style model on TinyStories with `softmax`, `sigmoid`, `softmax_kbias` | in progress |
| 5 | Weight decay and learning rate affect sink strength (Sec. 5-6) | `--weight-decay` / `--lr` sweeps on the same model | planned |

The paper's own pretraining runs use 60M-1B models on 5B tokens across 4 A100s. Experiments 4-5 here
train on tens of millions of tokens, so they test whether the *direction* of each effect holds at small
scale, not the paper's exact numbers.

## Run it

```bash
uv sync
uv run pytest

# Experiments 1-3: ~5 min on an M-series Mac
PYTHONPATH=. uv run python experiments/open_models.py

# Experiment 4: one run per attention variant
PYTHONPATH=. uv run python experiments/pretrain.py --attention softmax --steps 3000
PYTHONPATH=. uv run python experiments/pretrain.py --attention sigmoid --steps 3000
PYTHONPATH=. uv run python experiments/pretrain.py --attention softmax_kbias --steps 3000

# Table + figures into results/summary.md and figures/
PYTHONPATH=. uv run python experiments/report.py
```

## Layout

```
sink/metrics.py      importance score and Sink_k
sink/probe.py        runs an HF model, collects attention maps and hidden/key/value norms
sink/data.py         natural, random-token and repeated-token inputs
sink/tiny_lm.py      small LLaMA-style decoder with swappable attention
experiments/         one script per experiment, plus report.py
results/             JSON per run and summary.md
figures/             plots from report.py
```

## Results

See [`results/summary.md`](results/summary.md) and [`figures/`](figures/).
