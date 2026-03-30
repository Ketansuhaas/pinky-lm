#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

TRAIN_SHARDS=${1:-1}
VARIANT=${2:-sp1024}

modal run --detach modal_train.py::download_data \
    --train-shards "$TRAIN_SHARDS" \
    --variant      "$VARIANT"
