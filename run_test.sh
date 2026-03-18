#!/bin/bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <ckpt_path> [extra lightning args...]" >&2
  exit 1
fi

CKPT_PATH="$1"
shift

pixi run python -m hepattn.experiments.ecal.main test \
  -c src/hepattn/experiments/ecal/configs/base.yaml \
  --ckpt_path "$CKPT_PATH" \
  "$@"
