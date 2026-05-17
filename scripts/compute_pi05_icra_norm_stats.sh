#!/usr/bin/env bash
set -euo pipefail

export R2A_DATASET_ROOT="${R2A_DATASET_ROOT:-/home/qid/agibot/agibot_r2a_lerobot}"
export BASELINE_NORM_ASSETS_DIR="${BASELINE_NORM_ASSETS_DIR:-/home/qid/agibot/checkpoint/pi05_icra_norm_assets}"
export HF_HOME="${HF_HOME:-/home/qid/agibot/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/home/qid/agibot/matplotlib}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/home/qid/agibot/uv_cache}"
export FFMPEG_LIB_DIR="${FFMPEG_LIB_DIR:-/opt/miniconda/lib}"
if [[ -d "${FFMPEG_LIB_DIR}" ]]; then
  export LD_LIBRARY_PATH="${FFMPEG_LIB_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

CONFIG_NAME="${CONFIG_NAME:-pi05_icra_simulation_challenge}"
MAX_FRAMES="${MAX_FRAMES:-}"

cd "$(dirname "$0")/.."
repo_root="$(pwd)"

if [[ -d "${repo_root}/lerobot/build/lib" ]]; then
  export PYTHONPATH="${repo_root}/lerobot/build/lib:${PYTHONPATH:-}"
fi

if [[ -n "${VENV_PYTHON:-}" ]]; then
  python_bin="${VENV_PYTHON}"
elif [[ -n "${VIRTUAL_ENV:-}" || -n "${CONDA_PREFIX:-}" ]]; then
  python_bin="$(command -v python)"
elif [[ -x "${repo_root}/.venv/bin/python" ]]; then
  python_bin="${repo_root}/.venv/bin/python"
else
  python_bin="$(command -v python)"
fi

compute_args=(scripts/compute_norm_stats.py --config-name "${CONFIG_NAME}")
if [[ -n "${MAX_FRAMES}" ]]; then
  compute_args+=(--max-frames "${MAX_FRAMES}")
fi

"${python_bin}" "${compute_args[@]}"

generated_stats="${repo_root}/assets/${CONFIG_NAME}/norm_stats.json"
if [[ ! -f "${generated_stats}" ]]; then
  echo "Expected norm stats were not generated: ${generated_stats}" >&2
  exit 1
fi

mkdir -p "${BASELINE_NORM_ASSETS_DIR}"
install -m 0644 "${generated_stats}" "${BASELINE_NORM_ASSETS_DIR}/norm_stats.json"
echo "Wrote norm stats to: ${BASELINE_NORM_ASSETS_DIR}/norm_stats.json"
