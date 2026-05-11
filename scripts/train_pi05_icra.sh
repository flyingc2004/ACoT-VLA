#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,2,3,4,6,7}"
export PI05_BASE_PARAMS="${PI05_BASE_PARAMS:-/mnt/a/ljz/.cache/openpi/openpi-assets/checkpoints/pi05_base/params}"
export R2A_DATASET_ROOT="${R2A_DATASET_ROOT:-/mnt/a/ljz/.tmp/agibot_r2a_lerobot}"
export PI05_CHECKPOINT_BASE_DIR="${PI05_CHECKPOINT_BASE_DIR:-/mnt/a/ljz/ACoT-VLA/checkpoints}"
export BASELINE_NORM_ASSETS_DIR="${BASELINE_NORM_ASSETS_DIR:-/mnt/a/ljz/ACoT-VLA/checkpoints/pi05_icra_norm_assets}"
export HF_HOME="${HF_HOME:-/mnt/a/ljz/.tmp/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/mnt/a/ljz/.tmp/matplotlib}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/mnt/a/ljz/.tmp/uv}"
export FFMPEG_LIB_DIR="${FFMPEG_LIB_DIR:-/mnt/a/ljz/miniconda/lib}"
if [[ -d "${FFMPEG_LIB_DIR}" ]]; then
  export LD_LIBRARY_PATH="${FFMPEG_LIB_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_autotune_level=0}"

CONFIG_NAME="${CONFIG_NAME:-pi05_icra_simulation_challenge}"
EXP_NAME="${EXP_NAME:-pi05_icra_v1}"
BATCH_SIZE="${BATCH_SIZE:-6}"
NUM_WORKERS="${NUM_WORKERS:-16}"
NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-50000}"
FSDP_DEVICES="${FSDP_DEVICES:-6}"
EMA_DECAY="${EMA_DECAY:-None}"
WANDB_ENABLED="${WANDB_ENABLED:-true}"

cd "$(dirname "$0")/.."

train_args=(
  "${CONFIG_NAME}"
  --exp-name "${EXP_NAME}"
  --overwrite
  --batch-size "${BATCH_SIZE}"
  --num-workers "${NUM_WORKERS}"
  --num-train-steps "${NUM_TRAIN_STEPS}"
  --fsdp-devices "${FSDP_DEVICES}"
  --ema-decay "${EMA_DECAY}"
)

case "${WANDB_ENABLED,,}" in
  true|1|yes|on)
    train_args+=(--wandb-enabled)
    ;;
  false|0|no|off)
    train_args+=(--no-wandb-enabled)
    ;;
  *)
    echo "Invalid WANDB_ENABLED value: ${WANDB_ENABLED}. Expected true/false." >&2
    exit 1
    ;;
esac

uv run python scripts/train.py "${train_args[@]}"
