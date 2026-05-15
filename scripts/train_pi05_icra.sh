#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export PI05_BASE_PARAMS="${PI05_BASE_PARAMS:-/datadrive4/qid/checkpoint/pi05_base/params}"
export R2A_DATASET_ROOT="${R2A_DATASET_ROOT:-/datadrive4/qid/agibot_r2a_lerobot}"
export PI05_CHECKPOINT_BASE_DIR="${PI05_CHECKPOINT_BASE_DIR:-/datadrive4/qid/checkpoint}"
export BASELINE_NORM_ASSETS_DIR="${BASELINE_NORM_ASSETS_DIR:-/datadrive4/qid/checkpoint/pi05_icra_norm_assets}"
export HF_HOME="${HF_HOME:-/datadrive4/qid/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/datadrive4/qid/matplotlib}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/datadrive4/qid/uv_cache}"
export TMPDIR="${TMPDIR:-/datadrive4/qid/tmp}"
export TEMP="${TEMP:-$TMPDIR}"
export TMP="${TMP:-$TMPDIR}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/datadrive4/qid/cache}"
export JAX_CACHE_DIR="${JAX_CACHE_DIR:-$XDG_CACHE_HOME/jax}"
export XLA_AUTOTUNE_CACHE_DIR="${XLA_AUTOTUNE_CACHE_DIR:-$JAX_CACHE_DIR/xla_gpu_per_fusion_autotune_cache_dir}"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$JAX_CACHE_DIR" "$XLA_AUTOTUNE_CACHE_DIR"
export WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-/datadrive4/qid/wandb_config}"
export WANDB_BASE_URL="${WANDB_BASE_URL:-https://api.wandb.ai}"
mkdir -p "$WANDB_CONFIG_DIR"
export FFMPEG_LIB_DIR="${FFMPEG_LIB_DIR:-/opt/miniconda/lib}"
if [[ -d "${FFMPEG_LIB_DIR}" ]]; then
  export LD_LIBRARY_PATH="${FFMPEG_LIB_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.95}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-true}"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_autotune_level=2} --xla_gpu_per_fusion_autotune_cache_dir=${XLA_AUTOTUNE_CACHE_DIR}"

CONFIG_NAME="${CONFIG_NAME:-pi05_icra_sorting_packages}"
EXP_NAME="${EXP_NAME:-pi05_sorting_packages_v1}"
BATCH_SIZE="${BATCH_SIZE:-256}"
NUM_WORKERS="${NUM_WORKERS:-32}"
NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-12500}"
SAVE_INTERVAL="${SAVE_INTERVAL:-2500}"
FSDP_DEVICES="${FSDP_DEVICES:-1}"
EMA_DECAY="${EMA_DECAY:-0.999}"
WANDB_ENABLED="${WANDB_ENABLED:-true}"
RESUME="${RESUME:-false}"
OVERWRITE="${OVERWRITE:-false}"
if [[ "${WANDB_ENABLED,,}" == "true" || "${WANDB_ENABLED,,}" == "1" || "${WANDB_ENABLED,,}" == "yes" || "${WANDB_ENABLED,,}" == "on" ]]; then
  if [[ -n "${WANDB_API_KEY:-}" ]]; then
    wandb login --relogin "${WANDB_API_KEY}"
  fi
fi

cd "$(dirname "$0")/.."

train_args=(
  "${CONFIG_NAME}"
  --exp-name "${EXP_NAME}"
  --batch-size "${BATCH_SIZE}"
  --num-workers "${NUM_WORKERS}"
  --num-train-steps "${NUM_TRAIN_STEPS}"
  --save-interval "${SAVE_INTERVAL}"
  --fsdp-devices "${FSDP_DEVICES}"
  --ema-decay "${EMA_DECAY}"
)

overwrite_enabled=false
resume_enabled=true
case "${OVERWRITE,,}" in
  true|1|yes|on) overwrite_enabled=true ;;
esac
case "${RESUME,,}" in
  true|1|yes|on) resume_enabled=true ;;
esac
if $overwrite_enabled && $resume_enabled; then
  echo "Cannot use --overwrite and --resume together." >&2
  exit 1
fi
if $overwrite_enabled; then
  train_args+=(--overwrite)
fi
if $resume_enabled; then
  train_args+=(--resume)
fi

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
