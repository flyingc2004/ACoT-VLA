# Agent Task: Switch PI05 Training Prompt To Segment-Specific JSON Prompt

## Goal

Modify PI05 ICRA training so the model uses the segment-specific prompt from the LeRobot JSON metadata, not `prompt_map_inject_to_training`.

The desired training prompt source is:

```text
meta/info.json -> instruction_segments[episode_index][segment]["instruction"]
```

The prompt should vary by frame/segment. For sorting, examples should look like:

```text
Grab the package on the table with right arm black.
Place the package on the scanning table with the barcode facing up.
```

Do not use the fixed `_icra_prompt_map()` template during PI05 training.

## Current Context

Main training script:

```text
scripts/train_pi05_icra.sh
```

The script usually trains:

```text
CONFIG_NAME=pi05_icra_sorting_packages
```

Relevant files:

```text
src/openpi/training/config.py
src/openpi/training/data_loader.py
src/openpi/transforms.py
src/openpi/policies/go2_policy.py
```

Do not recompute norm stats for this prompt-only change. Norm stats depend on state/action, not prompt text.

## Required Behavior

For `pi05_icra_sorting_packages`, the training data flow should be:

1. LeRobot dataset sample is loaded.
2. `PromptFromHighlevelInstruction` reads the current segment instruction from `meta/info.json`.
3. That segment instruction becomes `data["prompt"]`.
4. `RepackTransform` keeps `"prompt": "prompt"`.
5. `Go2ACOTInputs` passes `inputs["prompt"]` through unchanged.
6. `TokenizePrompt` tokenizes that segment-specific prompt.

The training prompt must not be replaced by `_icra_prompt_map()`.

## Patch Plan

### 1. Update `pi05_icra_sorting_packages` config

In `src/openpi/training/config.py`, find:

```python
TrainConfig(
    name="pi05_icra_sorting_packages",
    ...
)
```
Delete `_icra_prompt_map()`

Inside its `LerobotPi05Go2DataConfig(...)`, remove:

```python
prompt_map_inject_to_training=_icra_prompt_map(),
```

Then set its base config to use segment-specific JSON prompts:

```python
base_config=DataConfig(
    dataloader_sampler="subtask",
    prompt_from_hl_instruction=True,
),
```

Do not set:

```python
prompt_from_task=True
prompt_from_episode_instruction=True
```

Reason:

- `prompt_from_task=True` gives only the generic task name, for example `Sort logistics parcels`.
- `prompt_from_episode_instruction=True` gives the full episode-level instruction, not the segment-specific prompt.
- `prompt_from_hl_instruction=True` uses the current segment instruction selected by `episode_index` and `frame_index`.

If training uses `pi05_icra_simulation_challenge` instead of `pi05_icra_sorting_packages`, apply the same change there too.

### 2. Make `PromptFromHighlevelInstruction` repack-safe

In `src/openpi/transforms.py`, find `PromptFromHighlevelInstruction.__call__`.

It should return the segment instruction as `prompt`, and also keep it under `segment_instruction` for any repack config that expects that key.

Use this behavior:

```python
result = {
    **data,
    "prompt": instruction,
    "subtask": instruction,
    "segment_instruction": instruction,
}
return result
```

Do not preserve the old generic task prompt in this mode.

### 3. Ensure PI05 does not inject `prompt_map_inject_to_training`

In `src/openpi/policies/go2_policy.py`, check `Go2ACOTInputs.__call__`.

For this task, it should not call any helper like:

```python
self.inject_prompt_from_training_map(data)
```

If such logic exists, remove it or make it inactive when `prompt_map_inject_to_training` is empty.

Expected behavior:

```python
if "prompt" in data:
    inputs["prompt"] = data["prompt"]
```

No template replacement. No `<color>` replacement. No random prompt injection.

### 4. Repack expectations

`LerobotPi05Go2DataConfig.repack_transforms` should include at least:

```python
{
    "images": {
        "top_head": "observation.images.top_head",
        "hand_left": "observation.images.hand_left",
        "hand_right": "observation.images.hand_right",
    },
    "state": "observation.state",
    "actions": "action",
    "prompt": "prompt",
    "segment_instruction": "segment_instruction",
    "task": "task",
    "episode_index": "episode_index",
}
```

This is safe if step 2 sets `segment_instruction`. So you do not need to change .

## Validation

Run syntax check:

```bash
python -m compileall src/openpi/training/config.py src/openpi/transforms.py src/openpi/policies/go2_policy.py src/openpi/training/data_loader.py
```

Then run a data prompt sanity check on the training server:

```bash
R2A_DATASET_ROOT=/path/to/agibot_r2a_lerobot \
BASELINE_NORM_ASSETS_DIR=/path/to/pi05_icra_norm_assets \
HF_HOME=/path/to/writable/huggingface \
HF_DATASETS_CACHE=/path/to/writable/huggingface/datasets \
MPLCONFIGDIR=/tmp/matplotlib \
uv run python - <<'PY'
from openpi.training import config as cfg
from openpi.training import data_loader

tc = cfg.get_config("pi05_icra_sorting_packages")
dc = tc.data.create(tc.assets_dirs, tc.model)

print("prompt_from_task:", dc.prompt_from_task)
print("prompt_from_hl_instruction:", dc.prompt_from_hl_instruction)
print("dataloader_sampler:", dc.dataloader_sampler)

dataset = data_loader.create_torch_dataset(dc, tc.model)
item = dataset[0]

print("prompt:", item.get("prompt"))
print("segment_instruction:", item.get("segment_instruction"))
print("task:", item.get("task"))
PY
```

Expected:

```text
prompt_from_task: False
prompt_from_hl_instruction: True
prompt: <a segment-specific instruction from info.json>
segment_instruction: same as prompt
task: Sort logistics parcels
```

The prompt should not be:

```text
Sort logistics parcels
```

The prompt should not be:

```text
Grab the <color> package ...
```

It should be the actual segment text from JSON with concrete wording/color when that segment contains color.

## Training Notes

After this change, train from a fresh experiment directory or use `--overwrite`.

Recommended current PI05 training settings:

```text
warmup_steps = 1_000
ema_decay = 0.999
freeze_vision = False
freeze_llm = True
```

If `torchcodec` fails while reading videos, fix FFmpeg/library path on the training server before judging the prompt change. The prompt code can be correct even if local video decoding fails.

