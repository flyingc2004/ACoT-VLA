#!/usr/bin/env bash
set -euo pipefail

source_root="${R2A_SOURCE_ROOT:-/datadrive4/qid/agibot_challenge/data/Reasoning2Action-Sim/dataset_without_depth}"
target_root="${R2A_PREPARED_ROOT:-${R2A_DATASET_ROOT:-/datadrive4/qid/agibot_r2a_lerobot}}"

if [[ ! -d "${source_root}" ]]; then
  echo "Source dataset root does not exist: ${source_root}" >&2
  exit 1
fi

mkdir -p "${target_root}"

for source_dataset_dir in "${source_root}"/*; do
  [[ -d "${source_dataset_dir}" ]] || continue

  task_name="$(basename "${source_dataset_dir}")"
  target_dataset_dir="${target_root}/${task_name}"
  mkdir -p "${target_dataset_dir}"

  for part in meta data videos; do
    source_dir="${source_dataset_dir}/${part}"
    archive="${source_dataset_dir}/${part}.tar.gz.000"
    output_dir="${target_dataset_dir}/${part}"

    if [[ -d "${output_dir}" ]]; then
      echo "[skip] ${output_dir}"
      continue
    fi

    if [[ -d "${source_dir}" ]]; then
      echo "[link] ${source_dir} -> ${output_dir}"
      ln -s "${source_dir}" "${output_dir}"
      continue
    fi

    if [[ ! -f "${archive}" ]]; then
      echo "[missing] ${archive}"
      continue
    fi

    echo "[extract] ${archive} -> ${target_dataset_dir}"
    tar -xzf "${archive}" -C "${target_dataset_dir}"
  done
done

echo "Prepared dataset root: ${target_root}"
