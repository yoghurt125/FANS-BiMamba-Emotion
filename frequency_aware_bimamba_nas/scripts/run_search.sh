#!/usr/bin/env bash
set -euo pipefail

PROCESSED_DIR=${1:-./data/DEAP}
RESULTS_DIR=${2:-./results}

python src/nas_search.py   --processed_de_dir "${PROCESSED_DIR}"   --results_dir "${RESULTS_DIR}"   --seed 42   --batch_size 32   --num_steps 50   --search_epochs 80   --search_patience 15
