#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

modal run modal_train.py \
    --run         run-01 \
    --steps       5000 \
    --eval-every  500 \
    --log-every   1 \
    --block-size  1024 \
    --batch-size  32 \
    --lr          1e-3 \
    --embed-dim   64 \
    --n-heads     4 \
    --n-layers    4 \
    --max-val-tokens 500000
