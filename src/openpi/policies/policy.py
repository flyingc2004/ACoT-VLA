from collections.abc import Sequence
import logging
import pathlib
import time
from typing import Any, TypeAlias
import copy
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
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

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

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        start_time = time.monotonic()
        self._rng, sample_rng = jax.random.split(self._rng)
        subtask_rng = None
        if self._sample_low_level_task is not None:
            sample_rng, subtask_rng = jax.random.split(sample_rng)
            subtask_inputs = jax.tree.map(lambda x: x, obs)
            # Placeholder label used only to create the train-time high/low prompt shape.
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

        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        # Make a batch and convert to jax.Array.
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)

        outputs = {
            "state": inputs["state"]
        }
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
