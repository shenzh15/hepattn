#!/bin/bash
set -e

pixi run python src/hepattn/experiments/lhcb/main.py fit --config src/hepattn/experiments/lhcb/config/base.yaml

# pixi run python src/hepattn/experiments/lhcb/main.py fit --config src/hepattn/experiments/lhcb/config/base.yaml --trainer.fast_dev_run=true