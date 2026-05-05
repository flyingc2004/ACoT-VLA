#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export LD_LIBRARY_PATH="${CONDA_PREFIX:-}/lib:${CONDA_PREFIX:-}/lib/python3.11/site-packages/nvidia/cu13/lib:${CONDA_PREFIX:-}/lib/python3.11/site-packages/nvidia/cuda_nvrtc/lib:${LD_LIBRARY_PATH:-}"
export DEBUG_MODE=false
export WANDB_MODE=offline
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

BASELINE_DIR="${ACOT_BASELINE_CHECKPOINT_DIR:-${PROJECT_ROOT}/checkpoints/baseline/30000}"
export BASELINE_PARAMS="${BASELINE_PARAMS:-${BASELINE_DIR}/params}"
export BASELINE_NORM_ASSETS_DIR="${BASELINE_NORM_ASSETS_DIR:-${BASELINE_DIR}/assets}"

python yrm/train_original_style.py \
  --exp-name "${EXP_NAME:-baseline_5tasks_original_style}" \
  --overwrite \
  --batch-size "${BATCH_SIZE:-16}" \
  --num-workers "${NUM_WORKERS:-8}" \
  --num-train-steps "${NUM_TRAIN_STEPS:-20000}" \
  --save-interval "${SAVE_INTERVAL:-5000}" \
  --warmup-steps "${WARMUP_STEPS:-0}"
