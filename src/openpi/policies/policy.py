from collections.abc import Sequence
import logging
import os
import pathlib
import time
from typing import Any, TypeAlias
import copy
import traceback
import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.policies.sorting_phase_state_machine import SortingContinuousPromptController
from openpi.policies.sorting_segment_prompt_controller import SortingSegmentPromptController
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils
from PIL import Image

BasePolicy: TypeAlias = _base_policy.BasePolicy


class Policy(BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        self._sample_actions = nnx_utils.module_jit(model.sample_actions)
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._rng = rng or jax.random.key(0)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self._sorting_prompt_controller: Any | None = None
        sorting_prompt_mode = os.getenv("SORTING_PROMPT_CONTROLLER", "phase").strip().lower()
        try:
            if sorting_prompt_mode in ("", "phase", "old"):
                self._sorting_prompt_controller = SortingContinuousPromptController.from_env()
            elif sorting_prompt_mode == "segment":
                self._sorting_prompt_controller = SortingSegmentPromptController.from_env()
            elif sorting_prompt_mode == "off":
                self._sorting_prompt_controller = None
            else:
                logging.warning(
                    "Unknown SORTING_PROMPT_CONTROLLER=%r. Expected phase, segment, or off. "
                    "Automatic sorting prompt switching is disabled.",
                    sorting_prompt_mode,
                )
        except Exception:  # pylint: disable=broad-exception-caught
            logging.warning("Failed to initialize sorting prompt controller:\n%s", traceback.format_exc())

    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]
        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)
        logging.info("Task name: %s", inputs.get("task_name", ""))

        # Sorting prompt controller is feature-gated by SORTING_PROMPT_CONTROLLER.
        # phase keeps the old coarse controller; segment uses segment-specific prompts.
        if self._sorting_prompt_controller is not None:
            updated_prompt, prompt_pred = self._sorting_prompt_controller.step(inputs)
            if prompt_pred is not None:
                logging.info(
                    "Sorting prompt prediction: label=%s conf=%.4f",
                    prompt_pred.label,
                    prompt_pred.confidence,
                )
            if updated_prompt is not None:
                print(f"updated prompt: {updated_prompt}")
                inputs["prompt"] = updated_prompt
                logging.info("Updated sorting prompt to: %s", updated_prompt)

        # There are other tasks that requires prompt mapping, we need to add them here

        prompt_mapping = {
            "pour_workpiece": "Pour the workpiece into the box",
            "open_door": "Turn the doorknob and push the door",
            "scoop_popcorn": "Scoop the popcorn and pour it into the popcorn bucket",
            "hold_pot": "Grasp the two handles of the pot and place it on the stove",
            "place_block_into_box": (
                "Left arm pick up the yellow circular block from the table and "
                "place it into the round hole of the block box"
            ),
            "take_wrong_item_shelf": (
                "Right arm picks up the incorrectly placed item from the shelf "
                "and place it into the shopping basket"
            ),
            "stock_and_straighten_shelf": (
                "Right arm pick up the wei-chuan orange juice in the shopping basket "
                "and place it on the shelf, Then, right arm straighten the toppled "
                "wei-chuan grape juice"
            ),
            "clean_the_desktop": (
                "Pick up the pen on the left side and place it into the pen holder, "
                "close the laptop, pick up the tissue on the table and place it into "
                "the trash bin on the right size. Then, pick up the mouse and place "
                "it on the right side of the laptop. Finally, straighten the colored "
                "pencil box"
            ),
        }

        task_name = str(inputs.get("task_name", "")).strip()
        if task_name in prompt_mapping:
            new_prompt = prompt_mapping[task_name]
            inputs["prompt"] = new_prompt

        # Debug: save top_head to PNG. PIL needs (H,W) or (H,W,C) with C in {1,3,4}.
        img = inputs["images"]["top_head"]
        img_np = np.asarray(img)
        while img_np.ndim == 4 and img_np.shape[0] == 1:
            img_np = img_np[0]
        # CHW (C in {1,3,4}) -> HWC when spatial dims look like H,W
        if (
            img_np.ndim == 3
            and img_np.shape[0] in (1, 3, 4)
            and img_np.shape[1] >= 8
            and img_np.shape[2] >= 8
            and img_np.shape[-1] not in (1, 3, 4)
        ):
            img_np = np.transpose(img_np, (1, 2, 0))
        if img_np.ndim == 2 or (img_np.ndim == 3 and img_np.shape[-1] in (1, 3, 4)):
            Image.fromarray(img_np.astype(np.uint8)).save("top_head.png")
        else:
            logging.warning(
                "Skipping top_head.png debug save: top_head shape %s is not HWC-compatible "
                "(client should send uint8 HxWx3 or 1xHxWx3).",
                img_np.shape,
            )

        inputs = self._input_transform(inputs)
        # Make a batch and convert to jax.Array.
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)

        start_time = time.monotonic()
        self._rng, sample_rng = jax.random.split(self._rng)         
        outputs = {
            "state": inputs["state"]
        }
        sample_kwargs = copy.deepcopy(self._sample_kwargs)
        if noise is not None:
            noise_jax = jnp.asarray(noise, dtype=inputs["state"].dtype)
            if noise_jax.ndim == 2:
                noise_jax = noise_jax[None, ...]
            sample_kwargs["noise"] = noise_jax

        try:
            result = self._sample_actions(sample_rng, _model.Observation.from_dict(inputs), **sample_kwargs)
        except TypeError:
            # Some models do not expose a `noise` kwarg.
            result = self._sample_actions(sample_rng, _model.Observation.from_dict(inputs), **self._sample_kwargs)

        if isinstance(result, dict):
            outputs.update(result)    
        else:
            outputs["actions"] = result
        # outputs["actions"] = inputs["actions"]

        # Unbatch and convert to np.ndarray.
        outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)
        model_time = time.monotonic() - start_time

        outputs = self._output_transform(outputs)
        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }
        return self.post_process(obs, outputs)

    def post_process(self, obs: dict, outputs: dict) -> dict:
        task_name_requiring_waist = ["sorting_packages", "sorting_packages_continuous"]
        task_name = jax.tree.map(lambda x: x, obs).get("task_name", None)

        if task_name is None:
            return outputs

        print(f"Policy infering for task: {task_name}, with inference time: {outputs['policy_timing']['infer_ms']:.3f} ms")
        if task_name not in task_name_requiring_waist:
            # cut off waist actions for tasks that don't require it
            outputs["actions"] = outputs["actions"][:, :16]

        else:
            raw_state = jax.tree.map(lambda x: x, obs).get("state", None)
            assert raw_state is not None, "State is required for post-processing waist actions"
            # freeze four waist actions to the current state, utilizing only the last action for policy output
            outputs["actions"][:, 16:20] = raw_state[16:20]

        return outputs

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs)

        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results
