#!/usr/bin/env bash
# Download FineWeb shards for training.
# Usage: ./scripts/download.sh [num_train_shards]
#   num_train_shards: number of training shards to download (default: 1, max: 195)
#
# Each shard is ~191MB (~100M tokens). The full val set (1 shard) is always downloaded.

set -euo pipefail

TRAIN_SHARDS=${1:-1}

python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards "$TRAIN_SHARDS"

echo "Done. Downloaded $TRAIN_SHARDS train shard(s) + val set."
