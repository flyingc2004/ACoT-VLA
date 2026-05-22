# Sorting Segment Prompt Switching Chain

## Branch And Isolation Rule

This work should live on a separate branch:

```text
sorting-segment-prompt-controller
```

Do not modify or replace the existing coarse phase chain:

```text
src/openpi/policies/sorting_phase_state_machine.py
scripts/build_sorting_phase_dataset.py
scripts/train_sorting_phase_classifier.py
```

Those files remain the old continuous-prompt implementation:

```text
already_reset / in_progress / task_terminal
```

The segment-prompt logic must use new files:

```text
scripts/build_sorting_segment_prompt_dataset.py
scripts/train_sorting_segment_prompt_classifier.py
src/openpi/policies/sorting_segment_prompt_controller.py
```

`src/openpi/policies/policy.py` may only receive a small feature-gated integration point. The old controller must remain available and unchanged. Recommended environment switch:

```text
SORTING_PROMPT_CONTROLLER=phase     # old behavior
SORTING_PROMPT_CONTROLLER=segment   # new segment-specific prompt behavior
SORTING_PROMPT_CONTROLLER=off       # no automatic prompt switching
```

Implemented new files:

```text
scripts/build_sorting_segment_prompt_dataset.py
scripts/train_sorting_segment_prompt_classifier.py
src/openpi/policies/sorting_segment_prompt_controller.py
```

Implemented gated integration:

```text
src/openpi/policies/policy.py
```

Default behavior remains:

```text
SORTING_PROMPT_CONTROLLER=phase
```

This preserves the old chain unless `SORTING_PROMPT_CONTROLLER=segment` is explicitly set.

## Goal

Current continuous sorting inference uses `SortingContinuousPromptController` in
`src/openpi/policies/sorting_phase_state_machine.py`.

The existing logic is:

1. Use a visual classifier on `top_head`.
2. Predict coarse phase: `already_reset`, `in_progress`, `task_terminal`.
3. Use a small state machine to switch the whole continuous task prompt to the next package color.

This was suitable when the policy prompt was one long continuous instruction. It is not aligned with the current PI05 training setup, because PI05 training now uses the segment-specific prompt from LeRobot `meta/info.json`.

The new inference chain should therefore switch prompt inside one sorting task:

```text
top_head image
  -> segment/stage classifier
  -> transition-guarded state machine
  -> current segment prompt
  -> policy server input prompt
  -> PI05 action
```

The important constraint is that the classifier training images must be cut from the exact `instruction_segments` frame ranges in `meta/info.json`, not from manually defined broad task phases.

## Dataset Metadata Source

Use the LeRobot metadata file:

```text
/mnt/a/ljz/.tmp/agibot_r2a_lerobot/sorting_packages_part_1/meta/info.json
```

Relevant structure:

```json
{
  "fps": 30,
  "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
  "features": {
    "observation.images.top_head": {
      "dtype": "video",
      "shape": [400, 640, 3]
    }
  },
  "instruction_segments": {
    "0": [
      {
        "instruction": "Grab the package on the table with right arm black.",
        "start_frame_index": 0,
        "success_frame_index": 239,
        "end_frame_index": 239
      }
    ]
  }
}
```

Observed from `sorting_packages_part_1`:

- `total_episodes`: 196
- Each episode has 7 instruction segments.
- There are 27 unique raw instruction strings.
- Color-bearing first-grab segments:
  - black: 56
  - red: 59
  - yellow: 36
  - white: 45

Example episode 0:

| Segment | Frame Range | Instruction |
| --- | --- | --- |
| 0 | `[0, 239)` | `Grab the package on the table with right arm black.` |
| 1 | `[239, 302)` | `Turn the waist right to face the barcode scanner.` |
| 2 | `[302, 552)` | `Place the package on the scanning table with the barcode facing up.` |
| 3 | `[552, 811)` | `The right arm grabs the package.` |
| 4 | `[811, 899)` | `Rotate the waist with the right arm.` |
| 5 | `[899, 922)` | `Place the package in the blue bin.` |
| 6 | `[922, 998]` | `Both arms coordinate and the waist returns to the initial posture.` |

Use half-open intervals `[start_frame_index, end_frame_index)` for all non-final segments. This matches the training transform in `PromptFromHighlevelInstruction`, which selects a segment when:

```python
frame_index >= start and frame_index < end
```

For the final segment in an episode, including `end_frame_index` is acceptable.

## Current Training Prompt Alignment

PI05 training prompt comes from:

```text
src/openpi/training/data_loader.py
src/openpi/transforms.py
```

The relevant transform is `PromptFromHighlevelInstruction`. It reads:

```python
dataset_meta.info["instruction_segments"]
```

and replaces the model prompt with the segment instruction for the current `episode_index` and `frame_index`.

Therefore, inference must feed prompts from the same semantic space:

```text
Grab the package on the table with right arm <color>.
Turn the waist right to face the barcode scanner.
Place the package on the scanning table with the barcode facing up.
The right arm grabs the package.
Rotate the waist with the right arm.
Place the package in the blue bin.
Both arms coordinate and the waist returns to the initial posture.
```

The old long prompt should not be used for PI05 segment-prompt inference.

## New Classifier Dataset Build

The old `scripts/build_sorting_phase_dataset.py` creates coarse classes:

```text
already_reset
in_progress
task_terminal
```

For segment prompt switching, this is too coarse. The new dataset builder should produce segment/stage classes from `instruction_segments`.

Recommended class set:

```text
grab_table
turn_to_scanner
place_on_scanning_table
grab_from_scanning_table
rotate_to_bin
place_in_blue_bin
return_initial
```

Do not split `grab_table` into `grab_table_black`, `grab_table_red`, `grab_table_yellow`, `grab_table_white` unless there is a specific need. The target color is already known from the continuous color cycle or from the incoming GenieSim prompt. Keeping one `grab_table` class gives more data per class and makes the classifier focus on phase rather than color recognition.

The prompt for `grab_table` is generated with the active color:

```text
Grab the package on the table with right arm <active_color>.
```

Other stage prompts are fixed canonical prompts.

### Required Build Logic

For each dataset root:

```text
/mnt/a/ljz/.tmp/agibot_r2a_lerobot/sorting_packages_part_1
/mnt/a/ljz/.tmp/agibot_r2a_lerobot/sorting_packages_part_2
/mnt/a/ljz/.tmp/agibot_r2a_lerobot/sorting_packages_part_3
```

read:

```text
meta/info.json
```

For each episode key in `instruction_segments`:

1. Read each segment:
   - `instruction`
   - `start_frame_index`
   - `end_frame_index`
   - fallback to `success_frame_index` if `end_frame_index` is missing
2. Convert `instruction` to a canonical stage label.
3. Open the matching top-head video:

```text
videos/chunk-{episode_index // 1000:03d}/observation.images.top_head/episode_{episode_index:06d}.mp4
```

4. Sample frames only from that segment's frame window.
5. Save images under:

```text
outputs/sorting_segment_prompt_dataset/
  train/
    grab_table/
    turn_to_scanner/
    ...
  val/
    grab_table/
    turn_to_scanner/
    ...
```

6. Write `metadata.csv` with at least:

```text
split,label,dataset,episode,segment_index,start_frame,end_frame,frame,path,instruction,prompt
```

### Instruction To Label Mapping

Use regex/normalization instead of exact string matching, because `info.json` contains variants:

```text
Place the package in the blue bin.
Put the package in the blue box.
Both arms coordinate and the waist returns to the initial posture.
Arms and waist return to the initial posture.
Return both arms and waist to the initial posture.
```

Recommended mapping:

| Label | Match |
| --- | --- |
| `grab_table` | contains `grab` and `package on the table` |
| `turn_to_scanner` | contains `turn the waist` and `barcode scanner` |
| `place_on_scanning_table` | contains `place` and `scanning table` |
| `grab_from_scanning_table` | contains `right arm grabs` or `right arm grabbed` |
| `rotate_to_bin` | contains `rotate` and `waist` |
| `place_in_blue_bin` | contains `place` or `put`, and `blue bin` or `blue box` |
| `return_initial` | contains `return` and `initial`, or `initial posture` |

### Sampling Rules

Use segment-specific frame windows:

```python
start = int(segment["start_frame_index"])
end = int(segment.get("end_frame_index", segment.get("success_frame_index", start)))
```

For non-final segments:

```python
frame_idx in range(start, end)
```

For final segment:

```python
frame_idx in range(start, end + 1)
```

Avoid transition ambiguity:

- Trim 5-10 percent from both sides of long segments.
- For very short segments, keep all frames.
- Keep `max_frames_per_segment` to avoid overrepresenting long actions.
- Split train/val by episode, not by frame, to avoid leakage.

Recommended initial parameters:

```text
frame_stride: 5 or 10
max_frames_per_segment: 40
val_ratio: 0.1
camera_key: observation.images.top_head
trim_ratio: 0.05
place_scan_window: 0.30,0.85
grab_scan_window: 0.25,0.75
```

Do not use random horizontal flip for this classifier. The scanner/bin direction is semantically meaningful; flipping can corrupt the stage geometry.

`place_on_scanning_table` and `grab_from_scanning_table` are visually similar in `top_head`, so the implemented builder samples them from stricter windows:

```text
place_on_scanning_table: middle/late segment frames
grab_from_scanning_table: middle segment frames
```

This avoids training on ambiguous boundary frames where the package is already on the scanning table but the arm is about to grab it again.

Implemented command:

```bash
python scripts/build_sorting_segment_prompt_dataset.py \
  --dataset-roots \
    /mnt/a/ljz/.tmp/agibot_r2a_lerobot/sorting_packages_part_1 \
    /mnt/a/ljz/.tmp/agibot_r2a_lerobot/sorting_packages_part_2 \
    /mnt/a/ljz/.tmp/agibot_r2a_lerobot/sorting_packages_part_3 \
  --output-dir outputs/sorting_segment_prompt_dataset \
  --camera-key observation.images.top_head \
  --frame-stride 10 \
  --max-frames-per-segment 40 \
  --trim-ratio 0.05 \
  --place-scan-window 0.30,0.85 \
  --grab-scan-window 0.25,0.75
```

Outputs:

```text
outputs/sorting_segment_prompt_dataset/metadata.csv
outputs/sorting_segment_prompt_dataset/stats.json
outputs/sorting_segment_prompt_dataset/train/<stage>/*.jpg
outputs/sorting_segment_prompt_dataset/val/<stage>/*.jpg
```

## Classifier Training

`scripts/train_sorting_phase_classifier.py` can be reused structurally, but it should be pointed at the new segment dataset and should expect the new class names.

Recommended changes:

1. Rename output folder to avoid overwriting old phase classifier:

```text
outputs/sorting_segment_prompt_classifier/
```

2. Save checkpoint as:

```text
sorting_segment_prompt_classifier_best.pt
```

3. Store these fields in the checkpoint:

```python
{
    "arch": args.arch,
    "class_names": class_names,
    "image_size": args.image_size,
    "mean": mean,
    "std": std,
    "model_state_dict": ...,
    "stage_prompts": STAGE_PROMPTS,
}
```

4. Prefer:

```text
arch: vit_b_16 or resnet18
pretrained: true if network access/checkpoint cache is available
image_size: 224
```

5. Training augmentation should be conservative:

```text
Resize(224, 224)
ColorJitter(brightness=0.15, contrast=0.20, saturation=0.10, hue=0.0)
ToTensor()
Normalize(ImageNet mean/std)
```

Avoid hue jitter because prompt switching depends on color semantics at the first grab stage.

Implemented command:

```bash
python scripts/train_sorting_segment_prompt_classifier.py \
  --data-dir outputs/sorting_segment_prompt_dataset \
  --output-dir outputs/sorting_segment_prompt_classifier \
  --arch resnet18 \
  --epochs 20 \
  --batch-size 128
```

For ViT:

```bash
python scripts/train_sorting_segment_prompt_classifier.py \
  --data-dir outputs/sorting_segment_prompt_dataset \
  --output-dir outputs/sorting_segment_prompt_classifier \
  --arch vit_b_16 \
  --pretrained \
  --epochs 20 \
  --batch-size 64
```

Checkpoint:

```text
outputs/sorting_segment_prompt_classifier/sorting_segment_prompt_classifier_best.pt
```

## Inference Controller

Implement the new controller as a separate file:

```text
src/openpi/policies/sorting_segment_prompt_controller.py
```

Do not put this implementation in:

```text
src/openpi/policies/sorting_phase_state_machine.py
```

That file is the old coarse phase controller and must remain unchanged.

The new class name is:

```text
SortingSegmentPromptController
```

Implemented file:

```text
src/openpi/policies/sorting_segment_prompt_controller.py
```

Recommended runtime state:

```python
STAGE_ORDER = (
    "grab_table",
    "turn_to_scanner",
    "place_on_scanning_table",
    "grab_from_scanning_table",
    "rotate_to_bin",
    "place_in_blue_bin",
    "return_initial",
)
```

The controller should maintain:

```text
current_stage_index
current_color
color_cycle
last_prediction
prediction_confidence
debounce_counter
current_prompt
```

### First Prompt Color Logic

The segment classifier is responsible for recognizing the task stage only. It should not decide the target package color.

The first stage prompt is:

```text
Grab the package on the table with right arm <color>.
```

`<color>` is resolved by the controller in this order:

1. Extract color from the incoming GenieSim prompt or task text.
   - If the original prompt contains `yellow`, use `yellow`.
   - If it contains `white`, use `white`.
   - Same for `black` and `red`.
2. If no color exists in the incoming prompt, use:

```text
SORTING_INITIAL_COLOR
```

3. If `SORTING_INITIAL_COLOR` is not set, use the first color in:

```text
SORTING_COLOR_CYCLE
```

Recommended defaults:

```text
SORTING_INITIAL_COLOR=black
SORTING_COLOR_CYCLE=black,red,yellow,white
```

The controller therefore has two separate pieces of state:

```text
current_color
current_stage
```

This is intentional. A `grab_table` prediction only means "the robot is at the first stage"; it does not mean "the target color is visible or recognized correctly".

Prompt mapping:

```python
STAGE_PROMPTS = {
    "grab_table": "Grab the package on the table with right arm <color>.",
    "turn_to_scanner": "Turn the waist right to face the barcode scanner.",
    "place_on_scanning_table": "Place the package on the scanning table with the barcode facing up.",
    "grab_from_scanning_table": "The right arm grabs the package.",
    "rotate_to_bin": "Rotate the waist with the right arm.",
    "place_in_blue_bin": "Place the package in the blue bin.",
    "return_initial": "Both arms coordinate and the waist returns to the initial posture.",
}
```

Transition rule:

```text
Only accept current stage or next stage.
Do not jump arbitrarily across multiple stages.
Advance only when classifier predicts next stage with confidence >= threshold for N consecutive frames.
When return_initial advances back to grab_table, advance current_color to the next color in color_cycle.
```

Recommended defaults:

```text
confidence_threshold: 0.70
debounce_frames: 2 or 3
scan_table_debounce_frames: 4
color_cycle: black,red,yellow,white
```

Implemented environment variables:

```text
SORTING_SEGMENT_MODEL_PATH
SORTING_SEGMENT_DEVICE
SORTING_SEGMENT_CONFIDENCE
SORTING_SEGMENT_DEBOUNCE_FRAMES
SORTING_SEGMENT_SCAN_TABLE_DEBOUNCE_FRAMES
SORTING_SEGMENT_TASK_KEYWORDS
SORTING_INITIAL_COLOR
SORTING_COLOR_CYCLE
```

## Policy Integration

Current integration is in:

```text
src/openpi/policies/policy.py
```

The policy uses `SORTING_PROMPT_CONTROLLER`:

```text
SORTING_PROMPT_CONTROLLER=phase
```

keeps the old controller:

```text
SortingContinuousPromptController
```

```text
SORTING_PROMPT_CONTROLLER=segment
```

uses the new controller:

```text
SortingSegmentPromptController
```

```text
SORTING_PROMPT_CONTROLLER=off
```

disables automatic sorting prompt switching.

Segment-mode launch example:

```bash
export SORTING_PROMPT_CONTROLLER=segment
export SORTING_SEGMENT_MODEL_PATH=/mnt/a/ljz/ACoT-VLA/outputs/sorting_segment_prompt_classifier/sorting_segment_prompt_classifier_best.pt
export SORTING_INITIAL_COLOR=black
export SORTING_COLOR_CYCLE=black,red,yellow,white
export SORTING_SEGMENT_CONFIDENCE=0.70
export SORTING_SEGMENT_DEBOUNCE_FRAMES=2
export SORTING_SEGMENT_SCAN_TABLE_DEBOUNCE_FRAMES=4
```

Runtime flow:

```text
GenieSim observation
  -> policy.py receives images + prompt/task_name
  -> controller extracts images["top_head"]
  -> classifier predicts segment stage
  -> state machine validates transition
  -> controller emits segment prompt
  -> policy.py overwrites inputs["prompt"]
  -> OpenPI transforms tokenize prompt
  -> PI05 predicts action
```

## Debugging Requirements

During inference, log every accepted transition:

```text
[SORTING_SEGMENT] pred=turn_to_scanner conf=0.84 state=grab_table->turn_to_scanner prompt="Turn the waist right to face the barcode scanner."
```

Also log rejected predictions when confidence is high but transition is invalid:

```text
[SORTING_SEGMENT_REJECT] pred=place_in_blue_bin conf=0.91 current=grab_table reason=invalid_jump
```

Keep updating:

```text
top_head.png
```

for visual inspection, but do not save every frame unless explicitly debugging the classifier dataset.

## Validation Checklist

Before using the controller for PI05 inference:

1. Build the segment classifier dataset from `instruction_segments`.
2. Inspect `metadata.csv`; verify frames are inside the correct segment range.
3. Check class counts; no class should be near zero.
4. Train classifier and verify validation confusion matrix.
5. Run controller offline on recorded episodes and verify predicted stage order:

```text
grab_table
turn_to_scanner
place_on_scanning_table
grab_from_scanning_table
rotate_to_bin
place_in_blue_bin
return_initial
```

6. Run policy server with debug logs and confirm `inputs["prompt"]` changes inside one sorting task.
7. Compare the emitted prompts against the PI05 training prompts from `info.json`.

## Key Difference From The Old Phase Classifier

Old classifier:

```text
already_reset / in_progress / task_terminal
```

Old controller:

```text
switch full continuous prompt only after terminal/reset
```

New classifier:

```text
7 segment/stage labels from info.json instruction frame ranges
```

New controller:

```text
switch the prompt inside one package cycle to match PI05 segment-specific training
```

This is the required alignment:

```text
training prompt source == inference prompt source
info.json instruction_segments == controller stage prompts
```
