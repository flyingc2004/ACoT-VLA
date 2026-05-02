from __future__ import annotations

import dataclasses
import logging
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as tv_models
import torchvision.transforms as tv_transforms

LOGGER = logging.getLogger(__name__)

SORTING_PHASE_LABELS = ("already_reset", "in_progress", "task_terminal")
SORTING_COLOR_PATTERN = re.compile(r"\b(white|red|black|yellow)\b", re.IGNORECASE)


@dataclasses.dataclass(frozen=True)
class SortingPhasePrediction:
    label: str
    confidence: float


def _coerce_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError:
            return ""
    return str(value)


def _to_hwc_uint8(image: Any) -> np.ndarray:
    arr = np.asarray(image)

    while arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]

    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
        arr = np.transpose(arr, (1, 2, 0))

    if arr.ndim != 3:
        raise ValueError(f"Expected image with 3 dims, got shape {arr.shape}")

    if np.issubdtype(arr.dtype, np.floating):
        scale = 255.0 if arr.max() <= 1.0 else 1.0
        arr = np.clip(arr * scale, 0, 255).astype(np.uint8)
    elif arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]

    return arr


def _build_model(arch: str, num_classes: int, *, pretrained: bool = False) -> nn.Module:
    arch = arch.lower()
    if arch == "resnet18":
        weights = tv_models.ResNet18_Weights.DEFAULT if pretrained else None
        model = tv_models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    if arch == "vit_b_16":
        weights = tv_models.ViT_B_16_Weights.DEFAULT if pretrained else None
        model = tv_models.vit_b_16(weights=weights)
        in_features = model.heads.head.in_features
        model.heads.head = nn.Linear(in_features, num_classes)
        return model

    raise ValueError(f"Unsupported architecture: {arch}")


class SortingPhaseClassifier:
    def __init__(
        self,
        model: nn.Module,
        class_names: list[str],
        image_size: int,
        mean: list[float],
        std: list[float],
        device: str,
    ) -> None:
        self._model = model
        self._class_names = class_names
        self._device = torch.device(device)
        self._model.to(self._device)
        self._model.eval()

        self._transform = tv_transforms.Compose(
            [
                tv_transforms.ToPILImage(),
                tv_transforms.Resize((image_size, image_size)),
                tv_transforms.ToTensor(),
                tv_transforms.Normalize(mean=mean, std=std),
            ]
        )

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, *, device: str = "cuda") -> "SortingPhaseClassifier":
        ckpt_path = Path(checkpoint_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Sorting phase checkpoint not found: {ckpt_path}")

        checkpoint = torch.load(ckpt_path, map_location="cpu")
        class_names = checkpoint.get("class_names", list(SORTING_PHASE_LABELS))
        arch = checkpoint.get("arch", "resnet18")
        image_size = int(checkpoint.get("image_size", 224))
        mean = list(checkpoint.get("mean", [0.485, 0.456, 0.406]))
        std = list(checkpoint.get("std", [0.229, 0.224, 0.225]))

        model = _build_model(arch, len(class_names), pretrained=False)
        model.load_state_dict(checkpoint["model_state_dict"])

        return cls(
            model=model,
            class_names=class_names,
            image_size=image_size,
            mean=mean,
            std=std,
            device=device,
        )

    def predict(self, image: Any) -> SortingPhasePrediction:
        hwc = _to_hwc_uint8(image)
        tensor = self._transform(hwc).unsqueeze(0).to(self._device)

        with torch.no_grad():
            logits = self._model(tensor)
            probs = torch.softmax(logits, dim=1)
            conf, cls_idx = torch.max(probs, dim=1)

        index = int(cls_idx.item())
        return SortingPhasePrediction(label=self._class_names[index], confidence=float(conf.item()))


@dataclasses.dataclass
class SortingContinuousPromptController:
    classifier: SortingPhaseClassifier
    color_cycle: tuple[str, ...] = ("black", "red", "yellow")
    prompt_template: str = "Grab the <color> package on the table, turn the waist right to face the barcode scanner, place the package on the scanning table with the barcode facing up. Then, grab the package, rotate the waist and place the package in the blue bin. Finally, return the waist back to face the initial table"
    task_keywords: tuple[str, ...] = ("sorting_packages_continuous", "sort logistics parcels continuous")
    white_stuck_timeout_steps: int = 30
    forced_reset_anchor_color: str = "black"
    forced_reset_detect_steps: int = 3
    forced_reset_grace_steps: int = 20
    forced_reset_grace_extra_steps: int = 2
    forced_reset_streak_decay: int = 1

    _state: int = dataclasses.field(default=0, init=False)
    _color_index: int = dataclasses.field(default=0, init=False)
    current_prompt: str | None = dataclasses.field(default=None, init=False)
    is_initialized: bool = dataclasses.field(default=False, init=False)
    _white_stuck_steps: int = dataclasses.field(default=0, init=False)
    _already_reset_streak: int = dataclasses.field(default=0, init=False)
    _forced_reset_grace_left: int = dataclasses.field(default=0, init=False)

    def _anchor_color_index(self) -> int:
        anchor = self.forced_reset_anchor_color.strip().lower()
        if not anchor:
            return 0
        try:
            return self.color_cycle.index(anchor)
        except ValueError:
            return 0

    def _set_color_prompt_by_index(self, color_index: int) -> str:
        self._color_index = color_index % len(self.color_cycle)
        color = self.color_cycle[self._color_index]
        self.current_prompt = self.prompt_template.replace("<color>", color)
        return color

    @staticmethod
    def _default_model_path() -> str | None:
        # Repository root: .../ACoT-VLA/src/openpi/policies/sorting_phase_state_machine.py
        repo_root = Path(__file__).resolve().parents[3]
        outputs_dir = repo_root / "outputs"

        # Prefer the newest retrain checkpoint if available.
        retrain_candidates = sorted(
            outputs_dir.glob("sorting_phase_classifier_retrain_*/sorting_phase_classifier_best.pt"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for candidate in retrain_candidates:
            if candidate.exists():
                return str(candidate)

        baseline = outputs_dir / "sorting_phase_classifier" / "sorting_phase_classifier_best.pt"
        if baseline.exists():
            return str(baseline)

        return None

    @classmethod
    def from_env(cls) -> SortingContinuousPromptController | None:
        model_path = os.getenv("SORTING_PHASE_MODEL_PATH", "").strip()
        if not model_path:
            default_path = cls._default_model_path()
            if default_path is None:
                return None
            model_path = default_path

        device = os.getenv("SORTING_PHASE_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
        prompt_template = os.getenv("SORTING_CONTINUOUS_PROMPT_TEMPLATE", "Grab the <color> package on the table, turn the waist right to face the barcode scanner, place the package on the scanning table with the barcode facing up. Then, grab the package, rotate the waist and place the package in the blue bin. Finally, return the waist back to face the initial table")

        raw_cycle = os.getenv("SORTING_COLOR_CYCLE", "black, red, yellow")
        color_cycle = tuple([c.strip().lower() for c in raw_cycle.split(",") if c.strip()])
        if not color_cycle:
            color_cycle = ("black", "red", "yellow")

        raw_keywords = os.getenv(
            "SORTING_CONTINUOUS_TASK_KEYWORDS",
            "sorting_packages_continuous,sort logistics parcels continuous",
        )
        task_keywords = tuple([k.strip().lower() for k in raw_keywords.split(",") if k.strip()])

        white_stuck_timeout_steps = int(os.getenv("SORTING_WHITE_STUCK_TIMEOUT_STEPS", "30"))
        if white_stuck_timeout_steps < 0:
            white_stuck_timeout_steps = 0

        forced_reset_anchor_color = os.getenv("SORTING_FORCED_RESET_ANCHOR_COLOR", "black").strip().lower()

        forced_reset_detect_steps = int(os.getenv("SORTING_FORCED_RESET_DETECT_STEPS", "3"))
        if forced_reset_detect_steps < 0:
            forced_reset_detect_steps = 0

        forced_reset_grace_steps = int(os.getenv("SORTING_FORCED_RESET_GRACE_STEPS", "20"))
        if forced_reset_grace_steps < 0:
            forced_reset_grace_steps = 0

        forced_reset_grace_extra_steps = int(os.getenv("SORTING_FORCED_RESET_GRACE_EXTRA_STEPS", "2"))
        if forced_reset_grace_extra_steps < 0:
            forced_reset_grace_extra_steps = 0

        forced_reset_streak_decay = int(os.getenv("SORTING_FORCED_RESET_STREAK_DECAY", "1"))
        if forced_reset_streak_decay < 0:
            forced_reset_streak_decay = 0

        classifier = SortingPhaseClassifier.from_checkpoint(model_path, device=device)
        LOGGER.info(
            "Loaded sorting phase classifier from %s (colors=%s, white_timeout_steps=%s, forced_reset_anchor=%s, forced_reset_detect_steps=%s, forced_reset_grace_steps=%s, forced_reset_grace_extra_steps=%s, forced_reset_streak_decay=%s)",
            model_path,
            color_cycle,
            white_stuck_timeout_steps,
            forced_reset_anchor_color,
            forced_reset_detect_steps,
            forced_reset_grace_steps,
            forced_reset_grace_extra_steps,
            forced_reset_streak_decay,
        )

        return cls(
            classifier=classifier,
            color_cycle=color_cycle,
            prompt_template=prompt_template,
            task_keywords=task_keywords,
            white_stuck_timeout_steps=white_stuck_timeout_steps,
            forced_reset_anchor_color=forced_reset_anchor_color,
            forced_reset_detect_steps=forced_reset_detect_steps,
            forced_reset_grace_steps=forced_reset_grace_steps,
            forced_reset_grace_extra_steps=forced_reset_grace_extra_steps,
            forced_reset_streak_decay=forced_reset_streak_decay,
        )

    def _is_continuous_task(self, task_name: str, prompt: str) -> bool:
        task_name = task_name.lower()
        prompt = prompt.lower()
        if any(k in task_name for k in self.task_keywords) or "sort packages" in prompt:
            return True

        # Fallback for evaluation streams that omit task_name but keep sorting prompts.
        has_sorting_style_prompt = (
            SORTING_COLOR_PATTERN.search(prompt) is not None
            and ("package" in prompt)
        )
        return has_sorting_style_prompt

    def _extract_top_head_image(self, obs: dict) -> Any | None:
        images = obs.get("images")
        if not isinstance(images, dict):
            return None
        return images.get("top_head")

    def step(self, obs: dict) -> tuple[str | None, SortingPhasePrediction | None]:
        task_name = _coerce_str(obs.get("task_name"))
        prompt = _coerce_str(obs.get("prompt"))

        if not self._is_continuous_task(task_name, prompt):
            self._state = 0
            self._white_stuck_steps = 0
            self._already_reset_streak = 0
            self._forced_reset_grace_left = 0
            return None, None
        
        if not self.is_initialized:
            self.current_prompt = self.prompt_template.replace("<color>", self.color_cycle[self._color_index])
            self.is_initialized = True

        image = self._extract_top_head_image(obs)
        if image is None:
            print("No top_head image found in observation for sorting phase prediction")
            return None, None

        pred = self.classifier.predict(image)

        if self._forced_reset_grace_left > 0:
            self._forced_reset_grace_left -= 1

        if self._state == 0 and pred.label == "task_terminal":
            self._state = 1
            self._white_stuck_steps = 0
            self._already_reset_streak = 0
            print(f"Sorting phase prediction: label={pred.label} conf={pred.confidence:.4f}")
            LOGGER.info("Sorting prompt state transition: state0 -> state1 (terminal detected, conf=%.4f)", pred.confidence)
            return self.current_prompt, pred

        if self._state == 0 and pred.label == "already_reset":
            self._already_reset_streak += 1

            anchor_index = self._anchor_color_index()
            current_color = self.color_cycle[self._color_index].lower()
            anchor_color = self.color_cycle[anchor_index]
            required_streak = self.forced_reset_detect_steps
            if self._forced_reset_grace_left > 0:
                required_streak += self.forced_reset_grace_extra_steps
            if (
                self.forced_reset_detect_steps > 0
                and current_color != anchor_color
                and self._already_reset_streak >= required_streak
            ):
                self._state = 0
                self._white_stuck_steps = 0
                self._already_reset_streak = 0
                self._forced_reset_grace_left = 0
                reset_color = self._set_color_prompt_by_index(anchor_index)
                LOGGER.warning(
                    "Forced reset triggered by already_reset streak (required=%s), switching to anchor color=%s",
                    required_streak,
                    reset_color,
                )
                print("Forced reset triggered by already_reset prediction streak, switching to anchor color:", reset_color)
                return self.current_prompt, pred
        elif self._state == 0:
            self._already_reset_streak = max(0, self._already_reset_streak - self.forced_reset_streak_decay)

        if self._state == 0 and pred.label != "task_terminal":
            current_color = self.color_cycle[self._color_index].lower()
            if current_color == "white" and self.white_stuck_timeout_steps > 0:
                self._white_stuck_steps += 1
                if self._white_stuck_steps >= self.white_stuck_timeout_steps:
                    self._white_stuck_steps = 0
                    next_color = self._set_color_prompt_by_index(self._color_index + 1)
                    print("white stuck timeout reached, switching to next color:", next_color)
                    return self.current_prompt, pred
            else:
                self._white_stuck_steps = 0

        if self._state == 1 and pred.label == "already_reset":
            self._state = 0
            self._white_stuck_steps = 0
            self._already_reset_streak = 0
            self._forced_reset_grace_left = self.forced_reset_grace_steps
            color = self._set_color_prompt_by_index(self._color_index + 1)
            print(f"Sorting phase prediction: label={pred.label} conf={pred.confidence:.4f}")
            LOGGER.info(
                "Sorting prompt state transition: state1 -> state0 (reset detected, conf=%.4f), next color=%s",
                pred.confidence,
                color,
            )
            return self.current_prompt, pred

        return self.current_prompt, pred
