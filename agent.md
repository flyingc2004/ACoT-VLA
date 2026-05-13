# ACoT-VLA to PI05 Training Agent Guide

This document is for a fresh agent or engineer who needs to run the ACoT-VLA repository as a PI05 training job on another server.

The goal is not to re-analyze the whole project. Start from this guide, verify the listed files, adapt paths to the new server, then run data preparation, norm stats, and training.

## Goal

Convert/use the ICRA simulation challenge training path as a PI05 baseline:

- Config name: `pi05_icra_simulation_challenge`
- Model: `pi0.Pi0Config(pi05=True, action_dim=32, action_horizon=30)`
- Dataset root: local LeRobot-format Reasoning2Action-Sim task directories
- Norm stats: one shared `norm_stats.json`
- Training entrypoint: `scripts/train_pi05_icra.sh`

## Server-Specific Paths

On the new server, set these paths explicitly before running scripts:

```bash
export R2A_SOURCE_ROOT=/path/to/Reasoning2Action-Sim/dataset_without_depth
export R2A_DATASET_ROOT=/path/to/writable/prepared_agibot_r2a_lerobot
export BASELINE_NORM_ASSETS_DIR=/path/to/ACoT-VLA/checkpoints/pi05_icra_norm_assets
export PI05_BASE_PARAMS=/path/to/pi05_base/params
export PI05_CHECKPOINT_BASE_DIR=/path/to/ACoT-VLA/checkpoints
export HF_HOME=/path/to/writable/huggingface
export UV_CACHE_DIR=/path/to/writable/uv
export MPLCONFIGDIR=/path/to/writable/matplotlib
```

If `torchcodec` cannot find FFmpeg libraries, also set:

```bash
export FFMPEG_LIB_DIR=/path/to/conda-or-system/lib
export LD_LIBRARY_PATH="${FFMPEG_LIB_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
```

The source dataset directory may be read-only. The prepared dataset root must be writable.

## Required Code State

Check that these changes exist on the new server. If not, implement them before running training.

### `src/openpi/models/gemma.py`

`Variant` must include all variants implemented by `get_config()`, especially `gemma_300m_lora`; otherwise `tyro` can warn or reject valid configs.

Expected variants:

```python
Variant = Literal[
    "dummy",
    "gemma_50m",
    "gemma_150m",
    "gemma_250m",
    "gemma_300m",
    "gemma_300m_lora",
    "gemma_2b",
    "gemma_2b_lora",
]
```

### `src/openpi/training/config.py`

`repo_id` types must allow multi-dataset training:

```python
class DataConfig:
    repo_id: str | Sequence[str] | None = None

class DataConfigFactory:
    repo_id: str | Sequence[str] = tyro.MISSING
```

The `pi05_icra_simulation_challenge` config should use `LerobotPi05Go2DataConfig`, shared norm assets, and freeze the large frozen modules:

```python
TrainConfig(
    name="pi05_icra_simulation_challenge",
    checkpoint_base_dir=_env_path("PI05_CHECKPOINT_BASE_DIR", "./checkpoints"),
    model=pi0.Pi0Config(pi05=True, action_dim=32, action_horizon=30),
    data=LerobotPi05Go2DataConfig(
        repo_id=_r2a_repo_ids(...),
        assets=AssetsConfig(
            assets_dir=os.getenv("BASELINE_NORM_ASSETS_DIR", str(_baseline_checkpoint_dir() / "assets")),
            asset_id=".",
        ),
        prompt_map_inject_to_training=_icra_prompt_map(),
        base_config=DataConfig(dataloader_sampler="subtask", prompt_from_task=True),
        extra_delta_transform=True,
    ),
    weight_loader=weight_loaders.CheckpointWeightLoader(
        os.getenv("PI05_BASE_PARAMS", "gs://openpi-assets-preview/checkpoints/pi05_may21_280k_v1/params")
    ),
    freeze_filter=pi0.Pi0Config(
        pi05=True,
        action_dim=32,
        action_horizon=30,
    ).get_freeze_filter(freeze_vision=True, freeze_llm=True),
)
```

The freeze filter is important for 24 GB GPUs. Without it, the run becomes full AdamW fine-tuning of the VLM/vision/action modules and can OOM even with FSDP.

### `src/openpi/training/data_loader.py`

The dataloader must support local absolute LeRobot dataset paths. Newer `lerobot` treats the first constructor argument as a HuggingFace repo id, so this is wrong:

```python
LeRobotDatasetMetadata("/abs/path/to/task")
LeRobotDataset("/abs/path/to/task")
```

The local-path logic should convert it to:

```python
LeRobotDatasetMetadata("task", root="/abs/path/to/task")
LeRobotDataset("task", root="/abs/path/to/task")
```

For multi-dataset local paths:

```python
MultiLeRobotDataset(["task1", "task2"], root="/abs/path/to/prepared_root")
```

Also validate that each local task directory has at least:

```text
meta/
data/
```

and `videos/` when videos are packaged or required. If a directory only has `meta.tar.gz.000`, `data.tar.gz.000`, `videos.tar.gz.000`, it must be prepared first.

### `scripts/compute_norm_stats.py`

Norm stats should not decode videos/images. It only needs numeric fields. The script should:

- Disable visual feature loading for norm computation.
- Drop image/image-mask repack fields.
- Handle PI05 batches that do not contain `coarse_actions`.
- Save only keys actually observed, normally `state` and `actions`.

The important behavior is:

```python
candidate_keys = ["state", "actions", "coarse_actions"]
active_keys = set()
...
for key in candidate_keys:
    if key not in batch:
        continue
    stats[key].update(...)
    active_keys.add(key)
norm_stats = {key: stats[key].get_statistics() for key in active_keys}
```

## Required Helper Scripts

Create these scripts if they are absent.

### `scripts/prepare_r2a_lerobot_data.sh`

Purpose: create a writable prepared dataset root. If the source task already has `meta/data/videos`, create symlinks. If it only has `*.tar.gz.000`, extract into the prepared root.

```bash
#!/usr/bin/env bash
set -euo pipefail

source_root="${R2A_SOURCE_ROOT:-/mnt/a/sharedata/agibot_challenge/data/Reasoning2Action-Sim/dataset_without_depth}"
target_root="${R2A_PREPARED_ROOT:-${R2A_DATASET_ROOT:-/mnt/a/ljz/.tmp/agibot_r2a_lerobot}}"

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
```

### `scripts/compute_pi05_icra_norm_stats.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail

export R2A_DATASET_ROOT="${R2A_DATASET_ROOT:-/mnt/a/ljz/.tmp/agibot_r2a_lerobot}"
export BASELINE_NORM_ASSETS_DIR="${BASELINE_NORM_ASSETS_DIR:-/mnt/a/ljz/ACoT-VLA/checkpoints/pi05_icra_norm_assets}"
export HF_HOME="${HF_HOME:-/mnt/a/ljz/.tmp/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/mnt/a/ljz/.tmp/matplotlib}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/mnt/a/ljz/.tmp/uv}"
export FFMPEG_LIB_DIR="${FFMPEG_LIB_DIR:-/mnt/a/ljz/miniconda/lib}"
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
```

### `scripts/train_pi05_icra.sh`

Use conservative defaults for 6x RTX 4090 24 GB:

```bash
#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5}"
export PI05_BASE_PARAMS="${PI05_BASE_PARAMS:-/path/to/pi05_base/params}"
export R2A_DATASET_ROOT="${R2A_DATASET_ROOT:-/path/to/prepared_agibot_r2a_lerobot}"
export PI05_CHECKPOINT_BASE_DIR="${PI05_CHECKPOINT_BASE_DIR:-/path/to/ACoT-VLA/checkpoints}"
export BASELINE_NORM_ASSETS_DIR="${BASELINE_NORM_ASSETS_DIR:-/path/to/ACoT-VLA/checkpoints/pi05_icra_norm_assets}"
export HF_HOME="${HF_HOME:-/path/to/writable/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/path/to/writable/matplotlib}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/path/to/writable/uv}"
export FFMPEG_LIB_DIR="${FFMPEG_LIB_DIR:-/path/to/conda/lib}"
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
```

Important:

- `BATCH_SIZE` must be divisible by `jax.device_count()`.
- `FSDP_DEVICES` must divide `jax.device_count()`.
- On 6 GPUs, start with `BATCH_SIZE=6 FSDP_DEVICES=6 EMA_DECAY=None`.
- Do not pass `--wandb-enabled true`; `tyro` expects boolean flags as `--wandb-enabled` or `--no-wandb-enabled`.

## Execution Order

From the repository root:

```bash
chmod +x scripts/prepare_r2a_lerobot_data.sh
chmod +x scripts/compute_pi05_icra_norm_stats.sh
chmod +x scripts/train_pi05_icra.sh

./scripts/prepare_r2a_lerobot_data.sh
./scripts/compute_pi05_icra_norm_stats.sh
./scripts/train_pi05_icra.sh
```

Expected norm output:

```text
${BASELINE_NORM_ASSETS_DIR}/norm_stats.json
```

Do not recompute norm stats unless the dataset, action/state masks, action/state dimensions, delta transform, or normalization code changed.

## Validation Checks

Before a long run:

```bash
bash -n scripts/prepare_r2a_lerobot_data.sh scripts/compute_pi05_icra_norm_stats.sh scripts/train_pi05_icra.sh
python -m py_compile src/openpi/training/config.py src/openpi/training/data_loader.py scripts/compute_norm_stats.py
uv run python scripts/train.py pi05_icra_simulation_challenge --help
```

Check prepared data:

```bash
find "${R2A_DATASET_ROOT}" -maxdepth 2 -type d | head
```

Every task used by `_r2a_repo_ids(...)` must have:

```text
meta/
data/
videos/
```

## Common Failures and Fixes

### `HFValidationError: Repo id must be in the form ... /abs/path/...`

Cause: local absolute path was passed as HuggingFace `repo_id`.

Fix: implement the `data_loader.py` local path handling described above.

### `Local LeRobot dataset is not ready ... Missing directories: meta, data`

Cause: prepared root is empty or source data was not linked/extracted.

Fix:

```bash
./scripts/prepare_r2a_lerobot_data.sh
```

If source is read-only, do not extract in place. Extract/link into a writable prepared root.

### `Permission denied` while extracting tar files

Cause: source dataset directory is read-only.

Fix: set `R2A_DATASET_ROOT` or `R2A_PREPARED_ROOT` to a writable location and run the prepare script.

### `Unrecognized arguments: true`

Cause: `--wandb-enabled true` was passed to `tyro`.

Fix: use `--wandb-enabled` or `--no-wandb-enabled`. The provided training script already handles this.

### `Could not load libtorchcodec ... libavutil.so.59 not found`

Cause: FFmpeg shared libraries are installed but not visible in `LD_LIBRARY_PATH`, or FFmpeg is not installed.

Fix:

```bash
find /path/to/conda -name 'libavutil.so*' | head
export FFMPEG_LIB_DIR=/path/to/conda/lib
export LD_LIBRARY_PATH="${FFMPEG_LIB_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
```

Quick test:

```bash
LD_LIBRARY_PATH="${FFMPEG_LIB_DIR}" python - <<'PY'
import torchcodec
from torchcodec.decoders import VideoDecoder
print("torchcodec ok")
PY
```

### GPU OOM on 6x RTX 4090 24 GB

Use all of these together:

```bash
BATCH_SIZE=6 FSDP_DEVICES=6 EMA_DECAY=None ./scripts/train_pi05_icra.sh
```

Also ensure the PI05 config freezes vision and base LLM:

```python
freeze_filter=pi0.Pi0Config(
    pi05=True,
    action_dim=32,
    action_horizon=30,
).get_freeze_filter(freeze_vision=True, freeze_llm=True)
```

Why this matters:

- Full AdamW fine-tuning stores gradients and optimizer state for the whole VLM.
- EMA stores an additional full parameter copy.
- FSDP helps, but it may not compensate for full-model AdamW + EMA on 24 GB cards.

### `nvidia-smi` shows other processes using memory

Free the GPUs or choose clean devices:

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
```

Then keep `BATCH_SIZE` divisible by the number of visible GPUs.

## Final Notes

- The PI05 norm stats are independent of the FFmpeg fix; do not recompute just because video decoding was fixed.
- The prepared dataset root can mix symlinks to already-extracted source data and extracted local directories.
- Prefer explicit environment variables on the new server instead of editing hard-coded local paths repeatedly.
- If training fails again, identify whether it fails during:
  - config parsing,
  - dataset construction,
  - weight loading / `init_train_state`,
  - first forward/backward step.

Those stages have different fixes.
