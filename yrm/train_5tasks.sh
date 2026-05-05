#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# 保持与 scripts/train.sh 风格一致，仅用于 baseline + 5任务
export LD_LIBRARY_PATH="${CONDA_PREFIX:-}/lib:${CONDA_PREFIX:-}/lib/python3.11/site-packages/nvidia/cu13/lib:${CONDA_PREFIX:-}/lib/python3.11/site-packages/nvidia/cuda_nvrtc/lib:${LD_LIBRARY_PATH:-}"
export DEBUG_MODE=false
export WANDB_MODE=offline
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95

# 默认单卡，避免当前环境的 NCCL 多卡报错；需要多卡时自行改为 0,1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

BASELINE_DIR="${ACOT_BASELINE_CHECKPOINT_DIR:-${PROJECT_ROOT}/checkpoints/baseline/30000}"
DATASET_ROOT="${R2A_DATASET_ROOT:-${PROJECT_ROOT}/datasets/Reasoning2Action-Sim/dataset_without_depth}"
export BASELINE_PARAMS="${BASELINE_PARAMS:-${BASELINE_DIR}/params}"
export BASELINE_NORM_ASSETS_DIR="${BASELINE_NORM_ASSETS_DIR:-${BASELINE_DIR}/assets}"
export FIVE_TASK_REPO_PATHS="${FIVE_TASK_REPO_PATHS:-${DATASET_ROOT}/pour_workpiece:${DATASET_ROOT}/open_door:${DATASET_ROOT}/take_wrong_item_shelf:${DATASET_ROOT}/scoop_popcorn:${DATASET_ROOT}/scoop_popcorn_part_2:${DATASET_ROOT}/hold_pot}"

python yrm/train_5tasks.py \
  --exp-name "${EXP_NAME:-baseline_5tasks}" \
  --overwrite \
  --batch-size "${BATCH_SIZE:-256}" \
  --num-workers "${NUM_WORKERS:-24}" \
  --num-train-steps "${NUM_TRAIN_STEPS:-20000}" \
  --save-interval "${SAVE_INTERVAL:-5000}" \
  --warmup-steps "${WARMUP_STEPS:-0}"
