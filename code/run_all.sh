#!/usr/bin/env bash
set -euo pipefail
# Run the complete C-MAPSS model/seed/lambda matrix from this directory.
# Full C-MAPSS files must be present in ../4_dataset/CMAPSS.
DATA_DIR="../4_dataset/CMAPSS"
python -m model.experiment --data_dir "$DATA_DIR" --fds FD002 FD004 \
  --seeds 0 1 2 3 4 --epochs "${EPOCHS:-20}" \
  --output_dir results/experiment_matrix
python summarize.py
