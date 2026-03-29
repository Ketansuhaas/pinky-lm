#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [ -z "${1:-}" ]; then
    echo "usage: ./scripts/quantize.sh <checkpoint_path>"
    echo "  e.g. ./scripts/quantize.sh checkpoints/20260329_run-01/best.pt"
    exit 1
fi

python3 quantize.py --checkpoint "$1"
