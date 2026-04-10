#!/usr/bin/env bash
set -euo pipefail

# 独立于脚本：多任务（含 open_door）续训启动脚本
# 用法：
#   conda activate agibot
#   cd /home/fudan/ACoT-VLA
#   bash yrm/train_four_tasks.sh

# 与原 train.sh 对齐：优先 conda lib，并补 torchcodec 依赖的 nvrtc 目录
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cu13/lib:$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cuda_nvrtc/lib:${LD_LIBRARY_PATH:-}"
export DEBUG_MODE=false
export WANDB_MODE="${WANDB_MODE:-offline}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.95}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

export FOUR_TASK_REPO_PATHS="${FOUR_TASK_REPO_PATHS:-/data/Dataset/Reasoning2Action-Sim/dataset_without_depth/pour_workpiece:/data/Dataset/Reasoning2Action-Sim/dataset_without_depth/open_door:/data/Dataset/Reasoning2Action-Sim/dataset_without_depth/take_wrong_item_shelf:/data/Dataset/Reasoning2Action-Sim/dataset_without_depth/scoop_popcorn:/data/Dataset/Reasoning2Action-Sim/dataset_without_depth/scoop_popcorn_part_2:/data/Dataset/Reasoning2Action-Sim/dataset_without_depth/hold_pot}"

EXP_NAME="${EXP_NAME:-four_tasks_bs256_20k_nowarmup}"
BATCH_SIZE="${BATCH_SIZE:-256}"
NUM_WORKERS="${NUM_WORKERS:-24}"
NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-20000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-5000}"
WARMUP_STEPS="${WARMUP_STEPS:-0}"

python yrm/train_four_tasks.py \
  --exp-name "$EXP_NAME" \
  --overwrite \
  --batch-size "$BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --num-train-steps "$NUM_TRAIN_STEPS" \
  --save-interval "$SAVE_INTERVAL" \
  --warmup-steps "$WARMUP_STEPS"
