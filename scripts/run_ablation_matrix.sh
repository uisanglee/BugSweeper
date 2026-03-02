#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/run_ablation_matrix.sh DATASET COVERAGE GPU
# Example:
#   bash scripts/run_ablation_matrix.sh smart-contract 4 0

DATASET="${1:-smart-contract}"
COVERAGE="${2:-4}"
GPU="${3:-0}"

SEEDS=(42 7 13)

# single: baseline (function-level only)
# hier + max_pool_levels: ablations on hierarchy depth
POOL_CONFIGS=(
  "single 0"
  "hier 2"
  "hier 4"
  "hier 0"
)

mkdir -p logs

for seed in "${SEEDS[@]}"; do
  for cfg in "${POOL_CONFIGS[@]}"; do
    pool_mode="$(echo "$cfg" | awk '{print $1}')"
    max_levels="$(echo "$cfg" | awk '{print $2}')"

    run_name="${DATASET}_cov${COVERAGE}_seed${seed}_${pool_mode}_L${max_levels}"
    echo "[RUN] ${run_name}"

    python train.py \
      -S train \
      -M POOL \
      -L function \
      --coverage "${COVERAGE}" \
      --seed "${seed}" \
      --pool_mode "${pool_mode}" \
      --max_pool_levels "${max_levels}" \
      --gpu "${GPU}" \
      2>&1 | tee "logs/${run_name}.log"
  done
done
