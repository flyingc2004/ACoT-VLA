"""Policy transforms for the Go2 robot."""

import dataclasses
from typing import ClassVar
from collections.abc import Sequence
import numpy as np
import torch
import copy
import re

import openpi.models.model as _model
import openpi.transforms as transforms


SORT_PACKAGE_COLOR_PATTERN = re.compile(r"\b(white|red|black|yellow)\b", re.IGNORECASE)


@dataclasses.dataclass(frozen=True)
class Go2Inputs(transforms.DataTransformFn):
    """Inputs for the Go2 policy.
    """

    # The action dimension of the model. Will be used to pad state and actions.
    action_dim: int

    state_mask: np.ndarray | None = None
    action_mask: np.ndarray | None = None

    # The expected cameras names. All input cameras must be in this set. Missing cameras will be
    # replaced with black images and the corresponding `image_mask` will be set to False.
    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = ("top_head", "hand_left", "hand_right")

    rename_map = {
        "top_head": "base_0_rgb",
        "hand_left": "left_wrist_0_rgb",
        "hand_right": "right_wrist_0_rgb"
    }

    def __call__(self, data: dict) -> dict:

        # Pad the proprioceptive input to the action dimension of the model
        state = transforms.pad_to_dim(data["state"], self.action_dim)
        state = copy.deepcopy(state)
        # state[14:]  = state[14:] * 120
        if self.state_mask is not None:
            state[self.state_mask] = 0

        # Ensure state has correct shape [batch_size, state_dim]
        state = state.squeeze()

        # Parse images to uint8 (H,W,C) since LeRobot automatically stores as float32 (C,H,W)
        images = {}
        for camera in self.EXPECTED_CAMERAS:
            if camera in data["images"]:
                img = data["images"][camera]
                # Convert torch tensor to numpy array if needed
                if isinstance(img, torch.Tensor):
                    img = img.cpu().numpy()
                # Ensure image is in uint8 format
                if np.issubdtype(img.dtype, np.floating):
                    img = (255 * img).astype(np.uint8)
                # Convert from [C,H,W] to [H,W,C] if needed
                if img.shape[0] == 3:
                    img = np.transpose(img, (1, 2, 0))
                images[self.rename_map[camera]] = img
            else:
                raise ValueError(f"Camera {camera} not found in data")

        # Create image mask based on available cameras
        image_mask = {self.rename_map[camera]: np.True_ for camera in self.EXPECTED_CAMERAS}


        # Prepare inputs dictionary
        inputs = {
            "image": images,
            "image_mask": image_mask,
            "state": state,
        }

        # Add actions if present
        if "actions" in data:
            actions = data["actions"]
            if self.action_mask is not None:
                actions[:, self.action_mask[:actions.shape[1]]] = 0
            actions = transforms.pad_to_dim(actions, self.action_dim)



            inputs["actions"] = actions.squeeze()

        # Add prompt if present
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class Go2Outputs(transforms.DataTransformFn):
    """Outputs for the Go2 policy."""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :22])} 


@dataclasses.dataclass(frozen=True)
class Go2ACOTInputs(transforms.DataTransformFn):
    """Inputs for the Go2 policy.
    """

    action_dim: int

    state_mask: np.ndarray | None = None
    action_mask: np.ndarray | None = None
    prompt_map_inject_to_training: dict[str, Sequence[object]] | None = None

    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = ("top_head", "hand_left", "hand_right")

    rename_map = {
        "top_head": "base_0_rgb",
        "hand_left": "left_wrist_0_rgb",
        "hand_right": "right_wrist_0_rgb"
    }
    acot_action_generation: Sequence[Sequence[int]] | None = None

    def _extract_color_from_segment(self, data: dict) -> str | None:
        # Prefer segment-level instruction text. Fallback to current prompt when available.
        candidate_fields = ("segment_instruction", "segment_instructions", "prompt")
        for key in candidate_fields:
            raw_text = data.get(key)
            if not isinstance(raw_text, str):
                continue
            match = SORT_PACKAGE_COLOR_PATTERN.search(raw_text)
            if match is not None:
                return match.group(1).lower()
        return None

    def slice_state_and_action(self, data):
        # Slice the state and action to the expected dimensions based on the original data shape
        state_indices = None
        if len(data["state"]) == 183:
            state_indices = list(range(54, 68)) + [0, 1] + list(range(99, 104))

        if len(data["state"]) == 159:
            state_indices = list(range(30, 44)) + [0, 1] + list(range(75, 80))
        if state_indices is not None:
            data["state"] = data["state"][state_indices]

        if "actions" in data:
            assert data["actions"].shape[1] == 40
            data["actions"] = np.column_stack((data["actions"][:, 16:30], data["actions"][:, 0:2], data["actions"][:, 33:38]))
        return data
    
    def random_inject_prompt(self, data):
        task_name = data["task"]
        if self.prompt_map_inject_to_training is not None and task_name in self.prompt_map_inject_to_training:
            mapping = self.prompt_map_inject_to_training[task_name]
            if len(mapping) < 2:
                return data

            default_prompt = mapping[0]
            inject_prob = mapping[1]
            if not isinstance(default_prompt, str):
                return data
            if isinstance(inject_prob, (int, float, np.floating, str)):
                try:
                    inject_prob_value = float(inject_prob)
                except ValueError:
                    return data
            else:
                return data

            if isinstance(default_prompt, str) and "<color>" in default_prompt:
                detected_color = self._extract_color_from_segment(data)
                if detected_color is None:
                    return data
                default_prompt = default_prompt.replace("<color>", detected_color)

            if np.random.rand() < inject_prob_value:
                data["prompt"] = default_prompt
    
        return data

    def __call__(self, data: dict) -> dict:
        data = self.slice_state_and_action(data)
        state = copy.deepcopy(transforms.pad_to_dim(data["state"], self.action_dim))
        if self.state_mask is not None:
            state[np.array(self.state_mask)] = 0

        # Parse images to uint8 (H,W,C) since LeRobot automatically stores as float32 (C,H,W)
        images = {}
        for camera in self.EXPECTED_CAMERAS:
            if camera in data["images"]:
                img = data["images"][camera]
                if isinstance(img, torch.Tensor):
                    img = img.cpu().numpy()
                if np.issubdtype(img.dtype, np.floating):
                    img = (255 * img).astype(np.uint8)
                if img.shape[0] == 3:
                    img = np.transpose(img, (1, 2, 0))
                images[self.rename_map[camera]] = img
            else:
                raise ValueError(f"Camera {camera} not found in data")

        # Create image mask based on available cameras
        image_mask = {self.rename_map[camera]: np.True_ for camera in self.EXPECTED_CAMERAS}

        # Prepare inputs dictionary
        inputs = {
            "image": images,
            "image_mask": image_mask,
            "state": state,
        }

        if self.acot_action_generation is not None and "actions" in data:
            action_horizons = self.acot_action_generation[0]
            joint_action_shifts = self.acot_action_generation[1]

            raw_data = data["actions"]
            keys = ["coarse_actions", "actions"]
            for idx, key in enumerate(keys):
                action_horizon = action_horizons[idx]
                joint_action_shift = joint_action_shifts[idx]
                required_length = (action_horizon - 1) * joint_action_shift + 1
                data[key] = copy.deepcopy(raw_data[:required_length:joint_action_shift])
                assert len(data[key]) == action_horizon
        for key in ['coarse_actions', 'actions']:
            if key in data:
                if self.action_mask is not None:
                    data[key][:, np.array(self.action_mask)[:data[key].shape[1]]] = 0
                data[key] = transforms.pad_to_dim(data[key], self.action_dim)
                inputs[key] = data[key]

        if "task" in data: # training
            data = self.random_inject_prompt(data)

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        return inputs


@dataclasses.dataclass(frozen=True)
class Go2ACOTOutputs(transforms.DataTransformFn):
    """Outputs for the Go2 policy."""

    def __call__(self, data: dict) -> dict:
        keys = ['coarse_actions', 'actions']
        return {key: np.asarray(data[key][:, :21]) for key in keys if key in data}