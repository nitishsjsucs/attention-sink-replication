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
| 1 | Sinks exist across open LMs, even Pythia-14M; within Pythia they grow with scale (Sec. 3.3) | Sink_1 for GPT-2, Pythia 14M-410M, OPT, SmolLM2 on the paper's probe text | done |
| 2 | Sinks persist on random tokens; on one repeated token they vanish in rotary-PE models but not learned-PE models (Sec. 3.2, App. C.1) | Same models on random-token and one-repeated-token inputs | done |
| 3 | The first token has a massive hidden-state norm but small key and value norms, so it acts like a key bias (Sec. 3.1) | L2 norms of hidden states, keys and values per position and layer | done |
| 4 | Sigmoid attention without normalization prevents the sink; a learnable key bias ("K biases") absorbs it (Sec. 7) | Pretrain a ~30M LLaMA-style model on TinyStories with `softmax`, `sigmoid`, `softmax_kbias` | in progress |
| 5 | Weight decay and learning rate affect sink strength (Sec. 5-6) | `--weight-decay` / `--lr` sweeps on the same model | planned |

The paper's own pretraining runs use 60M-1B models on 5B tokens across 4 A100s. Experiments 4-5 here
train on tens of millions of tokens, so they test whether the *direction* of each effect holds at small
scale, not the paper's exact numbers.

### Experiment 4 log

- **Run 1** (softmax, 3k steps, 25M tokens, lr 1e-3): val loss 10.9 → 2.11, but **Sink_1 stayed at
  0.0% at every checkpoint**. This fits the paper's finding that the sink only emerges "after effective
  optimization on sufficient training data". It also means the softmax baseline has no sink yet for
  sigmoid to remove. Result: `results/pretrain/softmax_lr0.001_wd0.1_steps3000_seed0.json`.
- **Run 2** (softmax, 15k steps, 123M tokens, 1M stories, no data repeated): val loss fell to 1.75,
  and **Sink_1 was still 0.0% at every checkpoint**. `Sink_eos` (heads sinking on an end-of-story
  token) never went above 2.4%, so the sink didn't move to story boundaries either. In the final
  model every head's α₁ is 0.03-0.08, at or below the 0.074 that uniform causal attention would give
  over 64 tokens, so position 1 gets no special treatment. The heads do learn real structure: some
  put 30-56% of their attention on the previous token.
- **This matches the paper.** Its Appendix D.2 / Figure 28: with only 50M or 100M training tokens,
  Sink_1 "maintains a very small value (less than 1%)". In the default setup (60M model, 1M-token
  batches, 2048 context, lr 4e-4) the sink appears between 1k and 2k steps, i.e. after about
  1-2B tokens. Runs 1-2 here saw 25M and 123M tokens.
- **Next:** reaching a sink at this throughput (~17k tokens/s) would take about 16 hours per
  billion tokens. Cheaper levers the paper identifies: larger weight decay (Table 2) and a higher
  learning rate (Table 9). The paper also trains with 2048-token context versus 256 here.

## Run it

```bash
uv sync
uv run pytest

# Experiments 1-3: ~5 min on an M-series Mac (downloads ~5 GB of models)
PYTHONPATH=. uv run python experiments/open_models.py

# Experiment 4: one run per attention variant (~2 h each on an M5 with bf16).
# Checkpoints every eval; rerunning the same command resumes an interrupted run.
for a in softmax sigmoid softmax_kbias; do
  PYTHONPATH=. uv run python experiments/pretrain.py --attention $a --steps 15000 --eval-every 500 --train-docs 1000000
done

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

## Results so far

### Experiments 1-2: Sink_1 (%) on pretrained models

| Model | Position encoding | Natural | Random | Repeated |
|---|---|---|---|---|
| GPT-2 (124M) | learned | 71.5 | 72.5 | 60.6 |
| GPT-2 Medium | learned | 79.5 | 77.0 | 66.0 |
| OPT-125M | learned | 89.6 | 87.4 | 77.0 |
| OPT-350M | learned | 85.2 | 86.5 | 79.2 |
| Pythia-14M | rotary | 10.2 | 7.8 | 0.0 |
| Pythia-70M | rotary | 15.5 | 6.1 | 0.0 |
| Pythia-160M | rotary | 33.4 | 56.6 | 0.0 |
| Pythia-410M | rotary | 68.2 | 67.0 | 0.0 |
| SmolLM2-135M | rotary | 63.7 | 78.7 | 0.0 |
| SmolLM2-360M | rotary | 83.7 | 80.6 | 0.0 |

170 sequences per input type, T = 64, ε = 0.3. For reference, the paper reports GPT2-XL at
77.0 / 70.3 / 62.3 and Mistral-7B at 97.5 / 75.2 / 0.0.

- **Replicates:** a sink on token 1 exists in every model, including Pythia-14M. Sink_2 stays near 0
  for every model except Pythia, where it peaks at 11.9%.
- **Replicates:** within Pythia, the sink grows with size (10% → 68%).
- **Replicates:** random tokens still produce a sink about as strong as natural text.
- **Replicates, including the mechanism:** with one token repeated 64 times, the sink disappears
  in all four rotary models (0.0%) and stays in all four learned-PE models (60-79%). This is the
  paper's Appendix C.1 argument: under rotary, identical tokens have identical hidden states, so
  nothing marks position 1. A learned position embedding still does.

### Experiment 3: first-token norms vs. the mean of the other tokens

| Model | Hidden state (max over layers) | Key (median layer) | Value (median layer) |
|---|---|---|---|
| GPT-2 | 40.7× | 0.44× | 0.27× |
| OPT-125M | 130.3× | 0.57× | 0.02× |
| SmolLM2-360M | 167.2× | 0.37× | 0.18× |
| Pythia-410M | 31.8× | 0.90× | 0.16× |

All models are in [`results/`](results/). As in the paper's Figure 2, the first token has massive
activations in the residual stream but small keys and values: it draws attention without
contributing much to the output. Two deviations to discuss:
- Pythia's first-token keys are about the same size as the other tokens' keys (0.9-1.0×), not
  smaller. Only the value norm drops.
- OPT-350M shows no massive activation (1.0×). It is the one post-LayerNorm model here
  (`do_layer_norm_before=False`), so the hidden states it returns are already normalized.

Not yet replicated: the paper's QK-angle analysis, which says the sink comes from
cos(q, k₁) rather than from key norms. This needs per-head queries, which Hugging Face doesn't
return, so it needs forward hooks for each architecture.

Full tables: [`results/summary.md`](results/summary.md). Plots: [`figures/`](figures/).
