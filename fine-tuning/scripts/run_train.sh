#!/usr/bin/env bash
set -euo pipefail
CONFIG=${1:-config.yaml}
python -m src.llm_finetune.train "$CONFIG"
