#!/usr/bin/env bash
set -euo pipefail

# 保持与 scripts/train.sh 风格一致，仅用于 baseline + 5任务
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cu13/lib:$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cuda_nvrtc/lib:${LD_LIBRARY_PATH:-}"
export DEBUG_MODE=false
export WANDB_MODE=offline
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95

# 默认单卡，避免当前环境的 NCCL 多卡报错；需要多卡时自行改为 0,1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

export BASELINE_PARAMS="${BASELINE_PARAMS:-/data/checkpoints/params}"
export BASELINE_NORM_ASSETS_DIR="${BASELINE_NORM_ASSETS_DIR:-/data/checkpoints/assets}"
export FIVE_TASK_REPO_PATHS="${FIVE_TASK_REPO_PATHS:-/data/Dataset/Reasoning2Action-Sim/dataset_without_depth/pour_workpiece:/data/Dataset/Reasoning2Action-Sim/dataset_without_depth/open_door:/data/Dataset/Reasoning2Action-Sim/dataset_without_depth/take_wrong_item_shelf:/data/Dataset/Reasoning2Action-Sim/dataset_without_depth/scoop_popcorn:/data/Dataset/Reasoning2Action-Sim/dataset_without_depth/scoop_popcorn_part_2:/data/Dataset/Reasoning2Action-Sim/dataset_without_depth/hold_pot}"

python yrm/train_5tasks.py \
  --exp-name "${EXP_NAME:-baseline_5tasks}" \
  --overwrite \
  --batch-size "${BATCH_SIZE:-256}" \
  --num-workers "${NUM_WORKERS:-24}" \
  --num-train-steps "${NUM_TRAIN_STEPS:-20000}" \
  --save-interval "${SAVE_INTERVAL:-5000}" \
  --warmup-steps "${WARMUP_STEPS:-0}"
