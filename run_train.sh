#!/usr/bin/env bash
set -euo pipefail
CONFIG=${1:-config.yaml}
python -m fine-tuning.fine-tuning.train "$CONFIG"
