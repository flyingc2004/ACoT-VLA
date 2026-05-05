#!/usr/bin/env python3
"""Build a frame dataset for sorting phase classification.

Classes:
- already_reset: robot has returned to initial posture after a package cycle.
- in_progress: all intermediate operation phases.
- task_terminal: package has been placed into the blue bin.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
import os
from pathlib import Path
import re

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

LABEL_ALREADY_RESET = "already_reset"
LABEL_IN_PROGRESS = "in_progress"
LABEL_TASK_TERMINAL = "task_terminal"
LABELS = (LABEL_ALREADY_RESET, LABEL_IN_PROGRESS, LABEL_TASK_TERMINAL)

# Terminal instruction variants in this dataset include both "blue bin" and "blue box",
# and verbs like "place" / "put".
TERMINAL_PAT = re.compile(
    r"(?:place|put)\s+the\s+package\s+in\s+the\s+blue\s+(?:bin|box)",
    re.IGNORECASE,
)
RESET_PAT = re.compile(r"return(?:s)?\b.*\b(?:initial|origin(?:al)?)\b", re.IGNORECASE)

# Prefer terminal/reset labels over in-progress when segment boundaries overlap.
LABEL_PRIORITY = {
    LABEL_IN_PROGRESS: 0,
    LABEL_TASK_TERMINAL: 1,
    LABEL_ALREADY_RESET: 2,
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_dataset_roots() -> list[str]:
    raw = os.getenv("SORTING_PHASE_DATASET_ROOTS", "").strip()
    if raw:
        return [p for p in raw.split(os.pathsep) if p]

    dataset_root = Path(
        os.getenv(
            "R2A_DATASET_ROOT",
            str(_repo_root() / "datasets" / "Reasoning2Action-Sim" / "dataset_without_depth"),
        )
    ).expanduser()
    return [
        str(dataset_root / "sorting_packages_part_1"),
        str(dataset_root / "sorting_packages_part_2"),
        str(dataset_root / "sorting_packages_part_3"),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build sorting phase dataset from videos")
    parser.add_argument(
        "--dataset-roots",
        nargs="+",
        default=_default_dataset_roots(),
        help="Sorting package dataset roots",
    )
    parser.add_argument(
        "--output-dir",
        default=os.getenv("SORTING_PHASE_DATASET_DIR", str(_repo_root() / "outputs" / "sorting_phase_dataset")),
        help="Output directory with train/val class folders",
    )
    parser.add_argument("--frame-stride", type=int, default=10, help="Sample one frame every N frames per segment")
    parser.add_argument(
        "--max-frames-per-segment",
        type=int,
        default=40,
        help="Cap sampled frames per segment to keep dataset balanced",
    )
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation split ratio by episode")
    parser.add_argument("--camera-key", default="observation.images.top_head", help="Video key folder under videos/chunk-xxx")
    parser.add_argument("--jpeg-quality", type=int, default=95, help="JPEG quality for saved frames")
    parser.add_argument(
        "--reset-tail-ratio",
        type=float,
        default=0.4,
        help="For already_reset segments, keep only the tail ratio of frames (0,1].",
    )
    parser.add_argument(
        "--terminal-tail-ratio",
        type=float,
        default=0.7,
        help="For task_terminal segments, keep only the tail ratio of frames (0,1].",
    )
    parser.add_argument(
        "--inprogress-trim-ratio",
        type=float,
        default=0.15,
        help="Trim this ratio from both start/end of in_progress segments to avoid transition ambiguity.",
    )
    return parser.parse_args()


def segment_label(instruction: str) -> str:
    text = instruction.strip().lower()
    if TERMINAL_PAT.search(text):
        return LABEL_TASK_TERMINAL
    if RESET_PAT.search(text):
        return LABEL_ALREADY_RESET
    return LABEL_IN_PROGRESS


def sample_indices(start: int, end: int, stride: int, max_frames: int) -> list[int]:
    if end < start:
        return []

    indices = list(range(start, end + 1, stride))
    if not indices or indices[-1] != end:
        indices.append(end)

    if max_frames > 0 and len(indices) > max_frames:
        keep = np.linspace(0, len(indices) - 1, num=max_frames, dtype=int)
        indices = [indices[i] for i in sorted(set(keep.tolist()))]

    return indices


def segment_sampling_window(
    start: int,
    end: int,
    label: str,
    *,
    reset_tail_ratio: float,
    terminal_tail_ratio: float,
    inprogress_trim_ratio: float,
) -> tuple[int, int]:
    if end < start:
        return start, end

    seg_len = end - start + 1
    if seg_len <= 2:
        return start, end

    if label == LABEL_ALREADY_RESET:
        keep_len = max(1, int(round(seg_len * reset_tail_ratio)))
        new_start = max(start, end - keep_len + 1)
        return new_start, end

    if label == LABEL_TASK_TERMINAL:
        keep_len = max(1, int(round(seg_len * terminal_tail_ratio)))
        new_start = max(start, end - keep_len + 1)
        return new_start, end

    if label == LABEL_IN_PROGRESS:
        trim = int(round(seg_len * inprogress_trim_ratio))
        new_start = start + trim
        new_end = end - trim
        if new_end < new_start:
            return start, end
        return new_start, new_end

    return start, end


def split_from_episode_key(key: str, val_ratio: float) -> str:
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()
    score = int(digest[:8], 16) / float(16**8 - 1)
    return "val" if score < val_ratio else "train"


def ensure_output_dirs(root: Path) -> None:
    for split in ("train", "val"):
        for label in LABELS:
            (root / split / label).mkdir(parents=True, exist_ok=True)


def build_episode_frame_labels(
    segments: list[dict],
    frame_stride: int,
    max_frames_per_segment: int,
    *,
    reset_tail_ratio: float,
    terminal_tail_ratio: float,
    inprogress_trim_ratio: float,
) -> dict[int, tuple[str, str]]:
    # frame_idx -> (label, instruction)
    labels: dict[int, tuple[str, str]] = {}

    for seg in segments:
        instruction = str(seg.get("instruction", "")).strip()
        label = segment_label(instruction)

        start = int(seg.get("start_frame_index", 0))
        end = int(seg.get("end_frame_index", start))
        start, end = segment_sampling_window(
            start,
            end,
            label,
            reset_tail_ratio=reset_tail_ratio,
            terminal_tail_ratio=terminal_tail_ratio,
            inprogress_trim_ratio=inprogress_trim_ratio,
        )

        for frame_idx in sample_indices(start, end, frame_stride, max_frames_per_segment):
            prev = labels.get(frame_idx)
            if prev is None or LABEL_PRIORITY[label] >= LABEL_PRIORITY[prev[0]]:
                labels[frame_idx] = (label, instruction)

    return labels


def process_dataset(
    dataset_root: Path,
    output_root: Path,
    frame_stride: int,
    max_frames_per_segment: int,
    val_ratio: float,
    camera_key: str,
    jpeg_quality: int,
    reset_tail_ratio: float,
    terminal_tail_ratio: float,
    inprogress_trim_ratio: float,
    metadata_writer: csv.writer,
    class_counter: Counter,
) -> None:
    info_path = dataset_root / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"Missing info.json: {info_path}")

    with info_path.open("r", encoding="utf-8") as f:
        info = json.load(f)

    instruction_segments = info.get("instruction_segments", {})
    dataset_name = dataset_root.name

    episode_keys = sorted(instruction_segments.keys(), key=lambda x: int(x))
    pbar = tqdm(episode_keys, desc=f"Extracting {dataset_name}", dynamic_ncols=True)

    for ep_key in pbar:
        episode_idx = int(ep_key)
        split = split_from_episode_key(f"{dataset_name}:{episode_idx}", val_ratio)
        segments = instruction_segments[ep_key]
        frame_labels = build_episode_frame_labels(
            segments,
            frame_stride,
            max_frames_per_segment,
            reset_tail_ratio=reset_tail_ratio,
            terminal_tail_ratio=terminal_tail_ratio,
            inprogress_trim_ratio=inprogress_trim_ratio,
        )
        if not frame_labels:
            continue

        video_path = (
            dataset_root
            / "videos"
            / f"chunk-{episode_idx // 1000:03d}"
            / camera_key
            / f"episode_{episode_idx:06d}.mp4"
        )
        if not video_path.exists():
            pbar.write(f"[WARN] Missing video: {video_path}")
            continue

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            pbar.write(f"[WARN] Failed to open video: {video_path}")
            continue

        frame_idx = 0
        saved = 0
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            payload = frame_labels.get(frame_idx)
            if payload is not None:
                label, instruction = payload
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                out_name = f"{dataset_name}_ep{episode_idx:06d}_f{frame_idx:06d}.jpg"
                out_path = output_root / split / label / out_name
                Image.fromarray(frame_rgb).save(out_path, format="JPEG", quality=jpeg_quality)

                class_counter[(split, label)] += 1
                metadata_writer.writerow(
                    [
                        split,
                        label,
                        dataset_name,
                        episode_idx,
                        frame_idx,
                        str(out_path.relative_to(output_root)),
                        instruction,
                    ]
                )
                saved += 1

            frame_idx += 1

        cap.release()
        pbar.set_postfix(saved=saved)


def main() -> None:
    args = parse_args()

    if args.frame_stride <= 0:
        raise ValueError("--frame-stride must be > 0")
    if not (0.0 < args.val_ratio < 1.0):
        raise ValueError("--val-ratio must be in (0,1)")
    if not (0.0 < args.reset_tail_ratio <= 1.0):
        raise ValueError("--reset-tail-ratio must be in (0,1]")
    if not (0.0 < args.terminal_tail_ratio <= 1.0):
        raise ValueError("--terminal-tail-ratio must be in (0,1]")
    if not (0.0 <= args.inprogress_trim_ratio < 0.5):
        raise ValueError("--inprogress-trim-ratio must be in [0,0.5)")

    output_root = Path(args.output_dir)
    ensure_output_dirs(output_root)

    metadata_path = output_root / "metadata.csv"
    class_counter: Counter = Counter()

    with metadata_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "label", "dataset", "episode", "frame", "path", "instruction"])

        for root in args.dataset_roots:
            process_dataset(
                dataset_root=Path(root),
                output_root=output_root,
                frame_stride=args.frame_stride,
                max_frames_per_segment=args.max_frames_per_segment,
                val_ratio=args.val_ratio,
                camera_key=args.camera_key,
                jpeg_quality=args.jpeg_quality,
                reset_tail_ratio=args.reset_tail_ratio,
                terminal_tail_ratio=args.terminal_tail_ratio,
                inprogress_trim_ratio=args.inprogress_trim_ratio,
                metadata_writer=writer,
                class_counter=class_counter,
            )

    stats = {
        "labels": LABELS,
        "frame_stride": args.frame_stride,
        "max_frames_per_segment": args.max_frames_per_segment,
        "val_ratio": args.val_ratio,
        "reset_tail_ratio": args.reset_tail_ratio,
        "terminal_tail_ratio": args.terminal_tail_ratio,
        "inprogress_trim_ratio": args.inprogress_trim_ratio,
        "counts": {
            split: {label: class_counter[(split, label)] for label in LABELS}
            for split in ("train", "val")
        },
    }
    stats_path = output_root / "stats.json"
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(f"Dataset built at: {output_root}")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
