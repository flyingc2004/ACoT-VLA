# Switch GenieSim Inference Server from ACoT to PI05

This document is for the server that previously ran the standard ACoT inference server with GenieSim. The new goal is to run the PI05 version instead, on the same server as GenieSim, to avoid cross-machine network issues.

Do not re-use the old ACoT server command blindly. The old `G2SIM` default points to the ACoT config/checkpoint. PI05 must be launched with an explicit PI05 config and PI05 checkpoint.

## What Changes

Old standard ACoT path:

```text
config: acot_icra_simulation_challenge_reasoning_to_action
server: scripts/server.sh or scripts/serve_policy.py --env G2SIM
checkpoint: checkpoints/acot_icra_simulation_challenge_reasoning_to_action/...
```

New PI05 path:

```text
config: pi05_icra_simulation_challenge
server: scripts/serve_policy.py policy:checkpoint ...
checkpoint: checkpoints/pi05_icra_simulation_challenge/pi05_icra_v1/<step>
port: 8999
```

The GenieSim side still calls the same websocket endpoint:

```text
localhost:8999
```

when GenieSim and the model server run on the same host with host networking.

## Required Code State

Before launching PI05 inference, make sure the repository on the GenieSim server includes the PI05 migration changes.

Minimum required checks:

```bash
cd /path/to/ACoT-VLA

python -m py_compile \
  src/openpi/models/gemma.py \
  src/openpi/training/config.py \
  src/openpi/training/data_loader.py \
  src/openpi/policies/policy.py \
  src/openpi/policies/policy_config.py \
  scripts/serve_policy.py
```

The following must be true:

- `src/openpi/training/config.py` contains `pi05_icra_simulation_challenge`.
- `pi05_icra_simulation_challenge` uses `pi0.Pi0Config(pi05=True, action_dim=32, action_horizon=30)`.
- `pi05_icra_simulation_challenge` uses `LerobotPi05Go2DataConfig`.
- `policy.py` defines `Policy` and `PolicyRecorder`; it must not be empty.
- `DataConfig.repo_id` and `DataConfigFactory.repo_id` allow `Sequence[str]`.
- `data_loader.py` handles local absolute LeRobot dataset paths with `repo_id=name, root=path`.
- `gemma.py` `Variant` includes `gemma_300m_lora`.
- `go2_policy.py` keeps the Go2 state/action slicing and output transforms aligned with the training dataset.

If this server is behind the current working server, copy the updated repository or apply the patches from the PI05 migration branch before continuing.

## Checkpoint Layout

Set the PI05 checkpoint step explicitly:

```bash
export CKPT_DIR=/path/to/ACoT-VLA/checkpoints/pi05_icra_simulation_challenge/pi05_icra_v1/50000
```

The checkpoint directory must contain:

```text
${CKPT_DIR}/params/
${CKPT_DIR}/assets/norm_stats.json
```

If `assets/norm_stats.json` is missing, copy the norm stats that were computed for PI05:

```bash
mkdir -p "${CKPT_DIR}/assets"
cp /path/to/ACoT-VLA/checkpoints/pi05_icra_norm_assets/norm_stats.json "${CKPT_DIR}/assets/norm_stats.json"
```

Do not recompute norm stats just for inference. Recompute only if dataset, action/state masks, delta transform, or normalization logic changed.

## Start PI05 Policy Server

Use one GPU for the PI05 server. Leave the other GPU(s) for GenieSim if possible.

```bash
cd /path/to/ACoT-VLA

export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.3
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS=--xla_gpu_autotune_level=0

export UV_CACHE_DIR=/path/to/writable/uv
export HF_HOME=/path/to/writable/huggingface
export MPLCONFIGDIR=/path/to/writable/matplotlib
export TOPHEAD_RECORD_DIR=/path/to/ACoT-VLA/policy_records/top_head
export TOPHEAD_RECORD_EVERY=1

# Needed if torchcodec/PyAV cannot find FFmpeg shared libs.
export FFMPEG_LIB_DIR=/path/to/conda/lib
if [[ -d "${FFMPEG_LIB_DIR}" ]]; then
  export LD_LIBRARY_PATH="${FFMPEG_LIB_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

export CKPT_DIR=/path/to/ACoT-VLA/checkpoints/pi05_icra_simulation_challenge/pi05_icra_v1/50000

GIT_LFS_SKIP_SMUDGE=1 uv run python scripts/serve_policy.py \
  --port 8999 \
  policy:checkpoint \
  --policy.config pi05_icra_simulation_challenge \
  --policy.dir "${CKPT_DIR}"
```

Successful startup should include:

```text
server listening on 0.0.0.0:8999
```

If you want a reusable launcher, create `scripts/server_pi05_icra.sh` with the command above. Do not use the old `scripts/server.sh` unless you also rewrite it to use `policy:checkpoint` and the PI05 config.

### Optional: Record Top-Head Camera Images

To inspect the simulation visually, enable top-head image recording on the policy server:

```bash
export TOPHEAD_RECORD_DIR=/path/to/ACoT-VLA/policy_records/top_head
export TOPHEAD_RECORD_EVERY=1
```

The server will continuously overwrite one latest image:

```text
policy_records/top_head/tophead.png
```

Use a larger interval if disk I/O becomes heavy:

```bash
export TOPHEAD_RECORD_EVERY=5
```

This records the raw `images.top_head` observation received from GenieSim before policy transforms. It does not change model inputs or actions.

If you really need a frame sequence for debugging, explicitly enable:

```bash
export TOPHEAD_RECORD_SEQUENCE=1
```

By default, sequence recording is disabled.

## Start GenieSim on the Same Server

Use the normal GenieSim Docker workflow from the GenieSim repo:

```bash
cd /path/to/genie_sim
./scripts/start_gui.sh
./scripts/into.sh
```

Inside the GenieSim container:

```bash
cd /geniesim/main
./scripts/run_icra_tasks.sh --infer-host localhost:8999
```

Use `localhost:8999` only when:

- GenieSim container uses host networking, or
- the container can reach the host loopback endpoint.

If `localhost` fails from inside the container, use the host LAN IP instead:

```bash
./scripts/run_icra_tasks.sh --infer-host <host-ip>:8999
```

## VLM Checker for Scoring

GenieSim docs say `scoop_popcorn` and `clean_the_desktop` use VLM evaluation. Missing VLM config should not block simulation, but scores for those tasks may be missing.

Inside GenieSim container, set these if available:

```bash
export BASE_URL=xxx
export VL_MODEL=xxx
export API_KEY=xxx
```

## Important Differences from the Old ACoT Server

### Do not use `--env G2SIM` for PI05 unless you changed defaults

This old pattern loads the default `G2SIM` checkpoint from `scripts/serve_policy.py`:

```bash
uv run python scripts/serve_policy.py --env G2SIM --port 8999
```

In the current code, `G2SIM` defaults to:

```text
config: acot_icra_simulation_challenge_reasoning_to_action
```

That is the old ACoT config. For PI05, use:

```bash
policy:checkpoint \
--policy.config pi05_icra_simulation_challenge \
--policy.dir "${CKPT_DIR}"
```

### Old checkpoint routing JSON is probably ACoT-specific

If the old server used:

```bash
--checkpoint-routing yrm/checkpoint_routing.example.json
```

check that JSON. If it points to ACoT checkpoints/configs, do not use it for PI05. For a single PI05 checkpoint, routing is unnecessary.

### Action post-processing still matters

The Go2 policy output transform/post-process may:

- clip non-sorting tasks to arm actions,
- preserve waist actions for `sorting_packages*` tasks using the same sliced state layout as training.

So keep the Go2/ICRA policy transform code from the current PI05 migration.

The current compact Go2 action layout is trained from the 40-D raw dataset action as:

```text
raw action[16:30]  -> 14 arm joints
raw action[0:2]    -> 2 grippers
raw action[33:38]  -> 5 waist joints
```

It does not train or output `raw action[38:40]` (`action/robot/velocity`). If you need chassis/base movement, this is an action-space change: update the transform, recompute norm stats, and retrain the PI05 checkpoint. Do not expect an old 21-D checkpoint to produce meaningful base velocity.

### Image dtype matters

GenieSim may send camera images as `float32` in either `0..1` or `0..255`. The Go2 policy image conversion must handle both. A bad conversion can make `tophead.png` look normal while the model receives corrupted images. The current migration version uses a robust `_to_uint8_hwc()` helper in `go2_policy.py`; make sure the target server has it.

For action-shape and value-range debugging, start the server with:

```bash
export POLICY_DEBUG_ACTIONS=1
```

The server will print the task name, raw state shape, final action shape, first action preview, and min/max values for the first few calls and then periodically.

To record the actual images passed into the model after Go2 policy conversion and resize:

```bash
export POLICY_DEBUG_IMAGES_DIR=/path/to/ACoT-VLA/policy_records/model_images
export POLICY_DEBUG_IMAGES_EVERY=1
```

Check these files while GenieSim is running:

```text
policy_records/model_images/model_base_0_rgb.png
policy_records/model_images/model_left_wrist_0_rgb.png
policy_records/model_images/model_right_wrist_0_rgb.png
```

These are more important than raw `tophead.png` for diagnosing image recognition, because they show what PI05 actually receives.

### Prompt is a secondary diagnostic

The current PI05 migration does not inject task prompts from GenieSim `task_name` during inference. This matches the old Go2/ACoT inference style more closely. If behavior is badly misaligned, first check checkpoint/norm consistency, camera mapping, state slicing, and action post-processing before changing prompt logic.

PI05 training still uses dataset task prompts through `prompt_from_task=True`, so prompt conditioning remains a possible experiment later, but it should not be the first explanation for a grossly wrong grasp.

For an A/B test with the same checkpoint, enable deterministic task-name prompt injection:

```bash
export PI05_INFER_PROMPT_FROM_TASK_NAME=1
export POLICY_DEBUG_PROMPT=1
```

If this improves PI05 noticeably, the issue is an inference/train prompt distribution mismatch rather than action dimensions or image conversion.

## Quick Smoke Tests

Before running GenieSim:

```bash
cd /path/to/ACoT-VLA

uv run python scripts/serve_policy.py --help
uv run python scripts/serve_policy.py policy:checkpoint --help
```

Expected checkpoint subcommand help:

```text
--policy.config STR
--policy.dir STR
```

Check port after starting server:

```bash
ss -ltnp | grep 8999
```

From the host:

```bash
python - <<'PY'
import socket
s = socket.create_connection(("127.0.0.1", 8999), timeout=3)
print("port open")
s.close()
PY
```

## Common Failures

### `AttributeError: module 'openpi.policies.policy' has no attribute 'Policy'`

`src/openpi/policies/policy.py` is empty or stale. Copy the current PI05 migration version of `policy.py`.

### `AttributeError: module 'openpi.models.gemma' has no attribute 'Variant'`

The target server has a stale `src/openpi/models/gemma.py`. `pi0.py` expects `_gemma.Variant`, so `gemma.py` must define it near the top, after `Config`.

Patch `src/openpi/models/gemma.py`:

```python
from typing import Literal

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

Then make sure the function signature uses it:

```python
def get_config(variant: Variant) -> Config:
    ...
```

Quick validation:

```bash
python - <<'PY'
from openpi.models import gemma
print(gemma.Variant)
print(gemma.get_config("gemma_300m"))
PY
```

### `tyro` says `policy:checkpoint` arguments are missing

Use this exact ordering:

```bash
uv run python scripts/serve_policy.py \
  --port 8999 \
  policy:checkpoint \
  --policy.config pi05_icra_simulation_challenge \
  --policy.dir "${CKPT_DIR}"
```

### Norm stats missing from checkpoint

Policy loading reads norm stats from:

```text
${CKPT_DIR}/assets/norm_stats.json
```

Copy it from the PI05 norm assets directory.

### OOM when model server and GenieSim run together

Use separate GPUs:

```bash
# terminal 1: PI05 policy server
export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.3

# terminal 2: GenieSim
# configure GenieSim/Docker to use a different GPU if possible
```

If GenieSim must see all GPUs, keep the model server memory fraction low and start the policy server first.

### GenieSim cannot connect to `localhost:8999`

The container may not be using host networking. Try:

```bash
./scripts/run_icra_tasks.sh --infer-host <host-ip>:8999
```

Also check firewall rules and that `serve_policy.py` printed `0.0.0.0:8999`.

## Minimal Runbook

1. Copy/update ACoT-VLA repo to the PI05 migration version.
2. Put PI05 checkpoint on the server.
3. Verify `${CKPT_DIR}/assets/norm_stats.json`.
4. Start PI05 server on port `8999` with `policy:checkpoint`.
5. Start GenieSim Docker.
6. Run:

```bash
./scripts/run_icra_tasks.sh --infer-host localhost:8999
```

7. Collect scores:

```bash
python3 scripts/stat_average.py
```
