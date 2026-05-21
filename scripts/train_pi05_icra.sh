#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export PI05_BASE_PARAMS="${PI05_BASE_PARAMS:-/home/qid/agibot/checkpoint/pi05-base/params}"
export R2A_DATASET_ROOT="${R2A_DATASET_ROOT:-/home/qid/agibot/agibot_r2a_lerobot}"
export PI05_CHECKPOINT_BASE_DIR="${PI05_CHECKPOINT_BASE_DIR:-/home/qid/agibot/checkpoint}"
export BASELINE_NORM_ASSETS_DIR="${BASELINE_NORM_ASSETS_DIR:-/home/qid/agibot/checkpoint/pi05_icra_norm_assets}"
export HF_HOME="${HF_HOME:-/datadrive4/qid/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/datadrive4/qid/matplotlib}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/home/qid/agibot/uv_cache}"
export TMPDIR="${TMPDIR:-/datadrive4/qid/tmp}"
export TEMP="${TEMP:-$TMPDIR}"
export TMP="${TMP:-$TMPDIR}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/datadrive4/qid/cache}"
export JAX_CACHE_DIR="${JAX_CACHE_DIR:-$XDG_CACHE_HOME/jax}"
export XLA_AUTOTUNE_CACHE_DIR="${XLA_AUTOTUNE_CACHE_DIR:-$JAX_CACHE_DIR/xla_gpu_per_fusion_autotune_cache_dir}"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$JAX_CACHE_DIR" "$XLA_AUTOTUNE_CACHE_DIR"
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
OVERWRITE="${OVERWRITE:-}"

cd "$(dirname "$0")/.."
export PYTHONPATH="/home/qid/agibot/ACoT-VLA/lerobot/build/lib:${PYTHONPATH:-}"

parse_bool() {
  local name="$1"
  local value="${2,,}"
  case "${value}" in
    true|1|yes|on)
      echo "true"
      ;;
    false|0|no|off)
      echo "false"
      ;;
    *)
      echo "Invalid ${name} value: ${2}. Expected true/false." >&2
      exit 1
      ;;
  esac
}

resume_enabled="$(parse_bool RESUME "${RESUME}")"
if [[ -z "${OVERWRITE}" ]]; then
  if [[ "${resume_enabled}" == "true" ]]; then
    overwrite_enabled="false"
  else
    overwrite_enabled="true"
  fi
else
  overwrite_enabled="$(parse_bool OVERWRITE "${OVERWRITE}")"
fi

if [[ "${resume_enabled}" == "true" && "${overwrite_enabled}" == "true" ]]; then
  echo "RESUME=true cannot be used with OVERWRITE=true." >&2
  exit 1
fi

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

if [[ "${resume_enabled}" == "true" ]]; then
  train_args+=(--resume)
elif [[ "${overwrite_enabled}" == "true" ]]; then
  train_args+=(--overwrite)
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

VENV_PYTHON="${VENV_PYTHON:-$(pwd)/.venv/bin/python}"
if [[ ! -x "${VENV_PYTHON}" ]]; then
  echo "Virtual environment Python not found: ${VENV_PYTHON}" >&2
  exit 1
fi

"${VENV_PYTHON}" scripts/train.py "${train_args[@]}"
