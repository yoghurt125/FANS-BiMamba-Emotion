#!/usr/bin/env bash
set -euo pipefail

PROCESSED_DIR=${1:-./data/DEAP}
RESULTS_DIR=${2:-./results}

python src/nas_eval.py   --processed_de_dir "${PROCESSED_DIR}"   --results_dir "${RESULTS_DIR}"   --seed 42   --batch_size 32   --max_epochs 150   --patience 30   --fusion_mode multihead   --num_heads 8   --selection_metric val_acc
