from __future__ import annotations

import dataclasses
import logging
import os
from pathlib import Path
import re
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as tv_models
import torchvision.transforms as tv_transforms

LOGGER = logging.getLogger(__name__)

SORTING_SEGMENT_LABELS = (
    "grab_table",
    "turn_to_scanner",
    "place_on_scanning_table",
    "grab_from_scanning_table",
    "rotate_to_bin",
    "place_in_blue_bin",
    "return_initial",
)

SORTING_COLOR_PATTERN = re.compile(r"\b(black|red|yellow|white)\b", re.IGNORECASE)

DEFAULT_STAGE_PROMPTS = {
    "grab_table": "Grab the package on the table with right arm <color>.",
    "turn_to_scanner": "Turn the waist right to face the barcode scanner.",
    "place_on_scanning_table": "Place the package on the scanning table with the barcode facing up.",
    "grab_from_scanning_table": "The right arm grabs the package.",
    "rotate_to_bin": "Rotate the waist with the right arm.",
    "place_in_blue_bin": "Place the package in the blue bin.",
    "return_initial": "Both arms coordinate and the waist returns to the initial posture.",
}


@dataclasses.dataclass(frozen=True)
class SortingSegmentPrediction:
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


def _extract_color(text: str) -> str | None:
    match = SORTING_COLOR_PATTERN.search(text)
    if match is None:
        return None
    return match.group(1).lower()


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
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
        return model

    raise ValueError(f"Unsupported architecture: {arch}")


class SortingSegmentClassifier:
    def __init__(
        self,
        model: nn.Module,
        class_names: list[str],
        image_size: int,
        mean: list[float],
        std: list[float],
        device: str,
        stage_prompts: dict[str, str] | None = None,
    ) -> None:
        self._model = model
        self._class_names = class_names
        self._device = torch.device(device)
        self.stage_prompts = dict(stage_prompts or DEFAULT_STAGE_PROMPTS)
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
    def from_checkpoint(cls, checkpoint_path: str, *, device: str = "cuda") -> "SortingSegmentClassifier":
        ckpt_path = Path(checkpoint_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Sorting segment checkpoint not found: {ckpt_path}")

        checkpoint = torch.load(ckpt_path, map_location="cpu")
        class_names = checkpoint.get("class_names", list(SORTING_SEGMENT_LABELS))
        arch = checkpoint.get("arch", "resnet18")
        image_size = int(checkpoint.get("image_size", 224))
        mean = list(checkpoint.get("mean", [0.485, 0.456, 0.406]))
        std = list(checkpoint.get("std", [0.229, 0.224, 0.225]))
        stage_prompts = checkpoint.get("stage_prompts", DEFAULT_STAGE_PROMPTS)

        model = _build_model(arch, len(class_names), pretrained=False)
        model.load_state_dict(checkpoint["model_state_dict"])

        return cls(
            model=model,
            class_names=class_names,
            image_size=image_size,
            mean=mean,
            std=std,
            device=device,
            stage_prompts=stage_prompts,
        )

    def predict(self, image: Any) -> SortingSegmentPrediction:
        hwc = _to_hwc_uint8(image)
        tensor = self._transform(hwc).unsqueeze(0).to(self._device)

        with torch.no_grad():
            logits = self._model(tensor)
            probs = torch.softmax(logits, dim=1)
            conf, cls_idx = torch.max(probs, dim=1)

        index = int(cls_idx.item())
        return SortingSegmentPrediction(label=self._class_names[index], confidence=float(conf.item()))


@dataclasses.dataclass
class SortingSegmentPromptController:
    classifier: SortingSegmentClassifier
    color_cycle: tuple[str, ...] = ("black", "red", "yellow", "white")
    initial_color: str = "black"
    confidence_threshold: float = 0.70
    debounce_frames: int = 2
    scan_table_debounce_frames: int = 4
    task_keywords: tuple[str, ...] = (
        "sorting_packages",
        "sorting_packages_continuous",
        "sort logistics parcels",
        "sort logistics parcels continuous",
        "sort packages",
    )
    stage_prompts: dict[str, str] = dataclasses.field(default_factory=lambda: dict(DEFAULT_STAGE_PROMPTS))

    _stage_index: int = dataclasses.field(default=0, init=False)
    _color_index: int = dataclasses.field(default=0, init=False)
    _candidate_label: str | None = dataclasses.field(default=None, init=False)
    _candidate_count: int = dataclasses.field(default=0, init=False)
    current_color: str | None = dataclasses.field(default=None, init=False)
    current_prompt: str | None = dataclasses.field(default=None, init=False)
    is_initialized: bool = dataclasses.field(default=False, init=False)

    @staticmethod
    def _default_model_path() -> str | None:
        repo_root = Path(__file__).resolve().parents[3]
        outputs_dir = repo_root / "outputs"

        candidates = sorted(
            outputs_dir.glob("sorting_segment_prompt_classifier*/sorting_segment_prompt_classifier_best.pt"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for candidate in candidates:
            if candidate.exists():
                return str(candidate)

        baseline = outputs_dir / "sorting_segment_prompt_classifier" / "sorting_segment_prompt_classifier_best.pt"
        if baseline.exists():
            return str(baseline)
        return None

    @classmethod
    def from_env(cls) -> "SortingSegmentPromptController | None":
        model_path = os.getenv("SORTING_SEGMENT_MODEL_PATH", "").strip()
        if not model_path:
            default_path = cls._default_model_path()
            if default_path is None:
                return None
            model_path = default_path

        device = os.getenv("SORTING_SEGMENT_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
        raw_cycle = os.getenv("SORTING_COLOR_CYCLE", "black,red,yellow,white")
        color_cycle = tuple(c.strip().lower() for c in raw_cycle.split(",") if c.strip())
        if not color_cycle:
            color_cycle = ("black", "red", "yellow", "white")

        initial_color = os.getenv("SORTING_INITIAL_COLOR", color_cycle[0]).strip().lower()
        if initial_color not in color_cycle:
            color_cycle = (initial_color, *color_cycle)

        raw_keywords = os.getenv(
            "SORTING_SEGMENT_TASK_KEYWORDS",
            "sorting_packages,sorting_packages_continuous,sort logistics parcels,sort logistics parcels continuous,sort packages",
        )
        task_keywords = tuple(k.strip().lower() for k in raw_keywords.split(",") if k.strip())

        classifier = SortingSegmentClassifier.from_checkpoint(model_path, device=device)
        LOGGER.info(
            "Loaded sorting segment classifier from %s (colors=%s, initial_color=%s)",
            model_path,
            color_cycle,
            initial_color,
        )

        return cls(
            classifier=classifier,
            color_cycle=color_cycle,
            initial_color=initial_color,
            confidence_threshold=float(os.getenv("SORTING_SEGMENT_CONFIDENCE", "0.70")),
            debounce_frames=max(1, int(os.getenv("SORTING_SEGMENT_DEBOUNCE_FRAMES", "2"))),
            scan_table_debounce_frames=max(1, int(os.getenv("SORTING_SEGMENT_SCAN_TABLE_DEBOUNCE_FRAMES", "4"))),
            task_keywords=task_keywords,
            stage_prompts=dict(classifier.stage_prompts),
        )

    @property
    def current_stage(self) -> str:
        return SORTING_SEGMENT_LABELS[self._stage_index]

    @property
    def next_stage(self) -> str:
        return SORTING_SEGMENT_LABELS[(self._stage_index + 1) % len(SORTING_SEGMENT_LABELS)]

    def reset(self) -> None:
        self._stage_index = 0
        self._candidate_label = None
        self._candidate_count = 0
        self.current_prompt = None
        self.current_color = None
        self.is_initialized = False

    def _is_sorting_task(self, task_name: str, prompt: str) -> bool:
        text = f"{task_name} {prompt}".lower()
        return any(keyword in text for keyword in self.task_keywords)

    def _extract_top_head_image(self, obs: dict) -> Any | None:
        images = obs.get("images")
        if not isinstance(images, dict):
            return None
        return images.get("top_head")

    def _initialize(self, task_name: str, prompt: str) -> None:
        incoming_color = _extract_color(f"{task_name} {prompt}")
        color = incoming_color or self.initial_color or self.color_cycle[0]

        self.current_color = color
        if color in self.color_cycle:
            self._color_index = self.color_cycle.index(color)
        else:
            self._color_index = 0

        self._stage_index = 0
        self.current_prompt = self._render_prompt(self.current_stage)
        self.is_initialized = True
        LOGGER.info(
            "[SORTING_SEGMENT_INIT] color=%s source=%s prompt=%s",
            self.current_color,
            "geniesim_prompt" if incoming_color else "fallback_cycle",
            self.current_prompt,
        )

    def _render_prompt(self, stage: str) -> str:
        prompt = self.stage_prompts[stage]
        color = self.current_color or self.initial_color
        return prompt.replace("<color>", color)

    def _advance_color(self) -> None:
        if self.current_color in self.color_cycle:
            self._color_index = self.color_cycle.index(self.current_color)
        self._color_index = (self._color_index + 1) % len(self.color_cycle)
        self.current_color = self.color_cycle[self._color_index]

    def _required_debounce_frames(self, next_label: str) -> int:
        if self.current_stage == "place_on_scanning_table" and next_label == "grab_from_scanning_table":
            return self.scan_table_debounce_frames
        return self.debounce_frames

    def _accept_next_stage_prediction(self, label: str) -> bool:
        if label != self.next_stage:
            self._candidate_label = None
            self._candidate_count = 0
            return False

        if self._candidate_label == label:
            self._candidate_count += 1
        else:
            self._candidate_label = label
            self._candidate_count = 1

        return self._candidate_count >= self._required_debounce_frames(label)

    def _advance_to_next_stage(self, pred: SortingSegmentPrediction) -> None:
        old_stage = self.current_stage
        new_stage = self.next_stage

        if old_stage == SORTING_SEGMENT_LABELS[-1] and new_stage == SORTING_SEGMENT_LABELS[0]:
            self._advance_color()

        self._stage_index = (self._stage_index + 1) % len(SORTING_SEGMENT_LABELS)
        self.current_prompt = self._render_prompt(self.current_stage)
        self._candidate_label = None
        self._candidate_count = 0
        LOGGER.info(
            '[SORTING_SEGMENT] pred=%s conf=%.4f state=%s->%s color=%s prompt="%s"',
            pred.label,
            pred.confidence,
            old_stage,
            self.current_stage,
            self.current_color,
            self.current_prompt,
        )

    def step(self, obs: dict) -> tuple[str | None, SortingSegmentPrediction | None]:
        task_name = _coerce_str(obs.get("task_name"))
        prompt = _coerce_str(obs.get("prompt"))

        if not self._is_sorting_task(task_name, prompt):
            self.reset()
            return None, None

        if not self.is_initialized:
            self._initialize(task_name, prompt)

        image = self._extract_top_head_image(obs)
        if image is None:
            LOGGER.warning("No top_head image found in observation for sorting segment prediction")
            return self.current_prompt, None

        pred = self.classifier.predict(image)
        if pred.label not in SORTING_SEGMENT_LABELS:
            LOGGER.warning("[SORTING_SEGMENT_REJECT] pred=%s conf=%.4f reason=unknown_label", pred.label, pred.confidence)
            return self.current_prompt, pred

        if pred.confidence < self.confidence_threshold:
            return self.current_prompt, pred

        if pred.label == self.current_stage:
            self._candidate_label = None
            self._candidate_count = 0
            return self.current_prompt, pred

        if self._accept_next_stage_prediction(pred.label):
            self._advance_to_next_stage(pred)
        else:
            LOGGER.info(
                "[SORTING_SEGMENT_REJECT] pred=%s conf=%.4f current=%s next=%s reason=debounce_or_invalid_jump",
                pred.label,
                pred.confidence,
                self.current_stage,
                self.next_stage,
            )

        return self.current_prompt, pred
