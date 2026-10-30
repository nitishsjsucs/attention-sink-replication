"""Builds results/summary.md and figures/ from the JSON results."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"


def first_vs_rest(norms):
    """norms: (layers, T). Per layer, token-1 norm over the mean of the rest."""
    return [row[0] / (sum(row[1:]) / len(row[1:])) for row in norms]


def open_models():
    rows = [json.loads(p.read_text()) for p in sorted(RESULTS.glob("*.json"))]
    if not rows:
        return ""
    lines = [
        "| Model | Sink_1 natural | Sink_1 random | Sink_1 repeated | Sink_2 natural |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        i = r["inputs"]
        lines.append(
            f"| {r['model']} | {i['natural']['sink_first8'][0]:.1f} | {i['random']['sink_first8'][0]:.1f} "
            f"| {i['repeated']['sink_first8'][0]:.1f} | {i['natural']['sink_first8'][1]:.1f} |"
        )

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for r in rows:
        nat = r["inputs"]["natural"]
        name = r["model"].split("/")[-1]
        axes[0].plot(nat["sink1_by_layer"], label=name)
        # hidden_norm has an extra entry for the embeddings; drop it to align with layers.
        axes[1].plot(first_vs_rest(nat["hidden_norm"][1:]), label=name)
        axes[2].plot(first_vs_rest(nat["value_norm"]), label=name)
    axes[0].set(title="Sink_1 by layer (natural text)", xlabel="layer", ylabel="% of heads")
    axes[1].set(title="Hidden-state norm, token 1 / mean of others", xlabel="layer", yscale="log")
    axes[2].set(title="Value norm, token 1 / mean of others", xlabel="layer", yscale="log")
    for ax in axes[1:]:
        ax.axhline(1, color="gray", lw=0.5)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / "open_models.png", dpi=150)
    return "## Pretrained open models (T = 64, epsilon = 0.3)\n\n" + "\n".join(lines) + "\n"


def pretrain():
    runs = [json.loads(p.read_text()) for p in sorted((RESULTS / "pretrain").glob("*.json"))]
    if not runs:
        return ""
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4))
    lines = ["| Attention | Params | Tokens | Final val loss | Final Sink_1 |", "|---|---|---|---|---|"]
    for r in runs:
        log, name = r["log"], r["args"]["attention"]
        tokens = [e["tokens"] / 1e6 for e in log]
        a.plot(tokens, [e["sink1"] for e in log], label=name)
        b.plot(tokens, [e["val_loss"] for e in log], label=name)
        lines.append(f"| {name} | {r['params'] / 1e6:.1f}M | {log[-1]['tokens'] / 1e6:.1f}M | {log[-1]['val_loss']:.3f} | {log[-1]['sink1']:.1f} |")
    a.set(title="Sink_1 during pretraining", xlabel="tokens (M)", ylabel="% of heads")
    b.set(title="Validation loss", xlabel="tokens (M)")
    a.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "pretrain.png", dpi=150)
    return "## Pretraining from scratch on TinyStories\n\n" + "\n".join(lines) + "\n"


if __name__ == "__main__":
    text = "# Results\n\n" + open_models() + "\n" + pretrain()
    (RESULTS / "summary.md").write_text(text)
    print(text)
