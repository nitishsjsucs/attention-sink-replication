#!/bin/sh
# Runs experiment 4: one pretraining run per attention variant, in sequence.
set -e
for a in softmax sigmoid softmax_kbias; do
  PYTHONPATH=. uv run python experiments/pretrain.py --attention "$a" --steps 3000 "$@"
done
