#!/usr/bin/env bash
set -euo pipefail

# Build a continuous-style sorting dataset without cross-episode stitching.
# It merges part_1/2/3 into one output dataset, rewrites reset-like instruction text,
# and keeps each source episode self-consistent.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_CMD="${PYTHON_CMD:-python}"

SOURCE_BASE="${SOURCE_BASE:-/mnt/sdb/xhz/Datasets/AgiBot/Reasoning2Action-Sim/dataset_without_depth}"
OUTPUT_DIR="${OUTPUT_DIR:-${SOURCE_BASE}/sorting_packages_continuous_v3}"

SOURCE_DIRS=(
  "${SOURCE_BASE}/sorting_packages_part_1"
  "${SOURCE_BASE}/sorting_packages_part_2"
  "${SOURCE_BASE}/sorting_packages_part_3"
)

for d in "${SOURCE_DIRS[@]}"; do
  if [[ ! -d "$d" ]]; then
    echo "[ERROR] Missing source directory: $d" >&2
    exit 1
  fi
done

cmd=(
  "$PYTHON_CMD" "${PROJECT_ROOT}/scripts/make_sorting_continuous_dataset.py"
  --source-dirs "${SOURCE_DIRS[@]}"
  --output-dir "$OUTPUT_DIR"
  --rewrite-only
  --sanitize-reset-instructions
  --overwrite
)

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  cmd+=(--dry-run)
fi

if [[ -n "${MAX_GROUPS:-}" ]]; then
  cmd+=(--max-groups "$MAX_GROUPS")
fi

if [[ -n "${TASK_NAME:-}" ]]; then
  cmd+=(--task-name "$TASK_NAME")
fi

if [[ -n "${RESET_INSTRUCTION_ALIAS:-}" ]]; then
  cmd+=(--reset-instruction-alias "$RESET_INSTRUCTION_ALIAS")
fi

echo "[INFO] Running command:"
printf ' %q' "${cmd[@]}"
echo
"${cmd[@]}"

echo "[INFO] Done. Output: $OUTPUT_DIR"
