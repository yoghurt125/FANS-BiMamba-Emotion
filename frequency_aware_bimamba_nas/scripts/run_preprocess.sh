#!/usr/bin/env bash
set -euo pipefail

RAW_DATA_PATH=${1:-/path/to/DEAP_Preprocessed_EEG}
PROCESSED_DIR=${2:-./data/DEAP}
LABEL_TYPE=${3:-valence}

python src/deap_dataset.py   --raw_data_path "${RAW_DATA_PATH}"   --processed_dir "${PROCESSED_DIR}"   --label_type "${LABEL_TYPE}"   --test_loader
