#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [ -z "${1:-}" ]; then
    echo "usage: ./scripts/eval.sh <checkpoint_path>"
    echo "  e.g. ./scripts/eval.sh checkpoints/20260329_run-01/best.pt"
    exit 1
fi

python3 eval.py \
    --checkpoint "$1" \
    --block-size 128 \
    --batch-size 64 \
    --embed-dim  64 \
    --n-heads    4 \
    --n-layers   4
