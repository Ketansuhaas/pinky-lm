#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [ -z "${1:-}" ]; then
    echo "usage: ./scripts/eval_quantized.sh <zlib_path>"
    echo "  e.g. ./scripts/eval_quantized.sh checkpoints/20260329_run-01/best_int8.zlib"
    exit 1
fi

python3 eval_quantized.py \
    --checkpoint "$1" \
    --block-size 1024 \
    --batch-size 64 \
    --embed-dim  64 \
    --n-heads    4 \
    --n-layers   4
