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
from openpi.models import tokenizer as _tokenizer
from openpi.policies.sorting_phase_state_machine import SortingContinuousPromptController
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
        high_level_transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        self._sample_actions = nnx_utils.module_jit(model.sample_actions)
        self._sample_low_level_task = (
            nnx_utils.module_jit(model.sample_low_level_task, static_argnums=(3,))
            if high_level_transforms and hasattr(model, "sample_low_level_task")
            else None
        )
        self._input_transform = _transforms.compose(transforms)
        self._high_level_input_transform = _transforms.compose(high_level_transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._rng = rng or jax.random.key(0)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self._subtask_max_decoding_steps = int(getattr(model, "subtask_max_decoding_steps", 25))
        self._subtask_temperature = float(getattr(model, "subtask_temperature", 0.0))
        self._subtask_detokenizer = (
            _tokenizer.PaligemmaTokenizer(max_len=max(model.max_token_len, self._subtask_max_decoding_steps))
            if self._sample_low_level_task is not None
            else None
        )
        self._sorting_prompt_controller: SortingContinuousPromptController | None = None
        try:
            self._sorting_prompt_controller = SortingContinuousPromptController.from_env()
        except Exception:  # pylint: disable=broad-exception-caught
            logging.warning("Failed to initialize sorting prompt controller:\n%s", traceback.format_exc())

    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]
        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)
        logging.info("Task name: %s", inputs.get("task_name", ""))

        # For sorting continuous tasks, update prompt by two-state machine:
        # state0 --(task_terminal)--> state1 --(already_reset)--> state0 and switch to next color.
        if self._sorting_prompt_controller is not None:
            updated_prompt, phase_pred = self._sorting_prompt_controller.step(inputs)
            if phase_pred is not None:
                logging.info(
                    "Sorting phase prediction: label=%s conf=%.4f",
                    phase_pred.label,
                    phase_pred.confidence,
                )
            if updated_prompt is not None:
                print(f"updated prompt: {updated_prompt}")
                inputs["prompt"] = updated_prompt
                logging.info("Updated sorting continuous prompt to: %s", updated_prompt)

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

        if os.getenv("ACOT_SAVE_TOP_HEAD_DEBUG", "").lower() in {"1", "true", "yes"}:
            # Debug: save top_head to PNG. PIL needs (H,W) or (H,W,C) with C in {1,3,4}.
            img = inputs.get("images", {}).get("top_head")
            if img is not None:
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

        start_time = time.monotonic()
        self._rng, sample_rng = jax.random.split(self._rng)
        if self._sample_low_level_task is not None:
            sample_rng, subtask_rng = jax.random.split(sample_rng)
            subtask_inputs = jax.tree.map(lambda x: x, inputs)
            # Placeholder label is masked out before decoding; it only builds the high/low token layout.
            subtask_inputs["subtask"] = np.asarray("placeholder subtask")
            subtask_inputs = self._high_level_input_transform(subtask_inputs)
            subtask_inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], subtask_inputs)
            subtask_observation = _model.Observation.from_dict(subtask_inputs)

            loss_mask = jnp.asarray(subtask_observation.token_loss_mask)
            tokenized_prompt = subtask_observation.tokenized_prompt.at[loss_mask].set(0)
            tokenized_prompt_mask = subtask_observation.tokenized_prompt_mask.at[loss_mask].set(False)
            subtask_observation = _model.Observation(
                images=subtask_observation.images,
                image_masks=subtask_observation.image_masks,
                state=subtask_observation.state,
                tokenized_prompt=tokenized_prompt,
                tokenized_prompt_mask=tokenized_prompt_mask,
                token_ar_mask=subtask_observation.token_ar_mask,
                token_loss_mask=subtask_observation.token_loss_mask,
            )
            subtask_tokens, _, _, _ = self._sample_low_level_task(
                subtask_rng,
                subtask_observation,
                self._subtask_max_decoding_steps,
                1,
                self._subtask_temperature,
            )
            subtask_tokens = np.asarray(subtask_tokens)
            subtask_texts = [
                self._subtask_detokenizer.detokenize(np.asarray(tokens, dtype=np.int32))
                for tokens in subtask_tokens
            ]
        else:
            subtask_tokens = None
            subtask_texts = None

        inputs = self._input_transform(inputs)
        # Make a batch and convert to jax.Array.
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)

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
        if subtask_texts is not None:
            outputs["subtask_tokens"] = np.asarray(subtask_tokens[0], dtype=np.int32)
            outputs["subtask_text"] = subtask_texts[0]
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
