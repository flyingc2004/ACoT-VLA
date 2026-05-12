#!/usr/bin/env bash
set -euo pipefail

export R2A_DATASET_ROOT="${R2A_DATASET_ROOT:-/datadrive4/qid/agibot_r2a_lerobot}"
export BASELINE_NORM_ASSETS_DIR="${BASELINE_NORM_ASSETS_DIR:-/datadrive4/qid/checkpoint/pi05_icra_norm_assets}"
export HF_HOME="${HF_HOME:-/datadrive4/qid/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/datadrive4/qid/matplotlib}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/datadrive4/qid/uv_cache}"
export FFMPEG_LIB_DIR="${FFMPEG_LIB_DIR:-/opt/miniconda/lib}"
if [[ -d "${FFMPEG_LIB_DIR}" ]]; then
  export LD_LIBRARY_PATH="${FFMPEG_LIB_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

CONFIG_NAME="${CONFIG_NAME:-pi05_icra_simulation_challenge}"
MAX_FRAMES="${MAX_FRAMES:-}"

cd "$(dirname "$0")/.."

if [[ -n "${MAX_FRAMES}" ]]; then
  uv run python scripts/compute_norm_stats.py --config-name "${CONFIG_NAME}" --max-frames "${MAX_FRAMES}"
else
  uv run python scripts/compute_norm_stats.py --config-name "${CONFIG_NAME}"
fi
