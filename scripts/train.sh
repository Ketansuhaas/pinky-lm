#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
python3 train.py \
    --steps 5000 \
    --eval-every 10 \
    --block-size 128 \
    --batch-size 4 \
    --lr 1e-3 \
    --embed-dim 64 \
    --n-heads 4 \
    --n-layers 4 \
    --ckpt-dir checkpoints \
    --max-val-tokens 500000 \
    --tokenizer data/tokenizers/fineweb_1024_bpe.model \
    --train data/datasets/fineweb10B_sp1024 \
    --val   data/datasets/fineweb10B_sp1024 \
    --wandb-project  pinky-lm \
    --wandb-run-name run-01
