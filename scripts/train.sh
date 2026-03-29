#!/usr/bin/env bash
# Train PinkyLM on FineWeb.
# Usage: ./scripts/train.sh [extra llm.py args]
#   e.g. ./scripts/train.sh --steps 10000 --batch-size 64

set -euo pipefail

python3 llm.py "$@"
