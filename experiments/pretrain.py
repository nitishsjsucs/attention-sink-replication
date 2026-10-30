"""Experiment 4: does the attention variant decide whether a sink forms?

Pretrains TinyLM from scratch on TinyStories with each attention variant and
tracks validation loss and Sink_1 (epsilon = 0.3, T = 64) during training.
The paper does this with 60M-1B LLaMA models on 5B tokens across 4 A100s;
this is the laptop-scale version, so read the trend, not the exact numbers.

    uv run python experiments/pretrain.py --attention softmax --steps 3000
"""

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer

from sink.metrics import importance_scores, sink_ratio
from sink.probe import pick_device
from sink.tiny_lm import Config, TinyLM

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data"


def token_stream(split: str, tokenizer, max_docs: int) -> torch.Tensor:
    """All stories in the split joined with EOS, as one long token tensor.
    Training windows are cut at random offsets, so position 1 of a window is
    usually mid-story: a sink there cannot come from a special first token."""
    path = CACHE / f"tinystories_{split}_{max_docs}.pt"
    if path.exists():
        return torch.load(path)
    ds = load_dataset("roneneldan/TinyStories", split=split)
    ds = ds.select(range(min(max_docs, len(ds))))
    eos = tokenizer.eos_token_id
    ids = []
    for i in range(0, len(ds), 10_000):
        for doc in tokenizer(ds[i : i + 10_000]["text"], add_special_tokens=False)["input_ids"]:
            ids.extend(doc)
            ids.append(eos)
    t = torch.tensor(ids, dtype=torch.int32)
    CACHE.mkdir(exist_ok=True)
    torch.save(t, path)
    return t


def batch(stream: torch.Tensor, B: int, T: int, g: torch.Generator):
    starts = torch.randint(0, len(stream) - T - 1, (B,), generator=g)
    x = torch.stack([stream[s : s + T + 1] for s in starts]).long()
    return x[:, :-1], x[:, 1:]


@torch.no_grad()
def evaluate(model, val, device, eos_id, T_sink=64, n=64, epsilon=0.3):
    model.eval()
    g = torch.Generator().manual_seed(1234)
    x, y = batch(val, n, model.cfg.max_len, g)
    x, y = x.to(device), y.to(device)
    logits = model(x)
    loss = F.cross_entropy(logits.flatten(0, 1), y.flatten()).item()
    _, attn = model(x[:, :T_sink], return_attn=True)
    attn = attn.float()
    sink = sink_ratio(attn, epsilon)
    # The sink may form on story boundaries instead of position 1. For windows
    # with an EOS in the first half (enough later queries to average over),
    # count heads whose importance score on that EOS exceeds epsilon.
    alpha = importance_scores(attn)  # (n, L, h, T)
    eos = (x[:, : T_sink // 2] == eos_id)
    rows = eos.any(-1)
    if rows.any():
        a = alpha[..., : T_sink // 2].masked_fill(~eos[:, None, None, :], 0).amax(-1)  # (n, L, h)
        sink_eos = ((a[rows] > epsilon).float().mean() * 100).item()
    else:
        sink_eos = float("nan")
    model.train()
    return loss, sink[0].item(), sink[1].item(), sink_eos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attention", default="softmax", choices=["softmax", "sigmoid", "softmax_kbias"])
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--train-docs", type=int, default=150_000)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = pick_device()
    tok = AutoTokenizer.from_pretrained("openai-community/gpt2")
    train = token_stream("train", tok, args.train_docs)
    val = token_stream("validation", tok, 5_000)
    print(f"train tokens {len(train):,}  val tokens {len(val):,}  device {device}", flush=True)

    cfg = Config(vocab_size=len(tok), attention=args.attention)
    model = TinyLM(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{args.attention}: {n_params / 1e6:.1f}M params", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay)
    warmup = args.steps // 20
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(s, args.steps) / args.steps)),
    )

    name = f"{args.attention}_lr{args.lr:g}_wd{args.weight_decay:g}_steps{args.steps}_seed{args.seed}"
    ckpt_path = CACHE / "checkpoints" / f"{name}.pt"
    g = torch.Generator().manual_seed(args.seed)
    log, start = [], 0
    if ckpt_path.exists():
        # Resume after an interruption: the step's eval already ran, so continue training from it.
        ckpt = torch.load(ckpt_path, weights_only=False)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        sched.load_state_dict(ckpt["sched"])
        g.set_state(ckpt["rng"])
        log, start = ckpt["log"], ckpt["step"]
        print(f"resumed {name} at step {start}", flush=True)
    t0 = time.time()
    for step in range(start, args.steps + 1):
        if step % args.eval_every == 0 and not (log and log[-1]["step"] == step):
            loss, s1, s2, se = evaluate(model, val, device, tok.eos_token_id)
            log.append({"step": step, "tokens": step * args.batch * cfg.max_len, "val_loss": loss,
                        "sink1": s1, "sink2": s2, "sink_eos": se})
            print(f"step {step:5d}  val_loss {loss:.3f}  Sink_1 {s1:5.1f}%  Sink_2 {s2:5.1f}%  "
                  f"Sink_eos {se:5.1f}%  {time.time() - t0:.0f}s", flush=True)
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                 "rng": g.get_state(), "log": log, "step": step},
                ckpt_path,
            )
        if step == args.steps:
            break
        x, y = batch(train, args.batch, cfg.max_len, g)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=args.bf16):
            logits = model(x.to(device))
            loss = F.cross_entropy(logits.flatten(0, 1).float(), y.to(device).flatten())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

    out = ROOT / "results" / "pretrain"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}.json").write_text(json.dumps({"args": vars(args), "params": n_params, "log": log}))


if __name__ == "__main__":
    main()
