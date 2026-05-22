#!/usr/bin/env python3
"""Build a segment-stage image dataset for sorting prompt switching.

This script is intentionally separate from build_sorting_phase_dataset.py.
It uses meta/info.json instruction_segments directly and creates labels for
segment-specific prompt switching.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    class _SimpleTqdm:
        def __init__(self, iterable, *args, **kwargs):
            self._iterable = iterable

        def __iter__(self):
            return iter(self._iterable)

        @staticmethod
        def write(message: str) -> None:
            print(message)

        def set_postfix(self, **kwargs) -> None:
            return None

    def tqdm(iterable, *args, **kwargs):
        return _SimpleTqdm(iterable)


LABEL_GRAB_TABLE = "grab_table"
LABEL_TURN_TO_SCANNER = "turn_to_scanner"
LABEL_PLACE_ON_SCANNING_TABLE = "place_on_scanning_table"
LABEL_GRAB_FROM_SCANNING_TABLE = "grab_from_scanning_table"
LABEL_ROTATE_TO_BIN = "rotate_to_bin"
LABEL_PLACE_IN_BLUE_BIN = "place_in_blue_bin"
LABEL_RETURN_INITIAL = "return_initial"

LABELS = (
    LABEL_GRAB_TABLE,
    LABEL_TURN_TO_SCANNER,
    LABEL_PLACE_ON_SCANNING_TABLE,
    LABEL_GRAB_FROM_SCANNING_TABLE,
    LABEL_ROTATE_TO_BIN,
    LABEL_PLACE_IN_BLUE_BIN,
    LABEL_RETURN_INITIAL,
)

COLOR_PATTERN = re.compile(r"\b(black|red|yellow|white)\b", re.IGNORECASE)

STAGE_PROMPTS = {
    LABEL_GRAB_TABLE: "Grab the package on the table with right arm <color>.",
    LABEL_TURN_TO_SCANNER: "Turn the waist right to face the barcode scanner.",
    LABEL_PLACE_ON_SCANNING_TABLE: "Place the package on the scanning table with the barcode facing up.",
    LABEL_GRAB_FROM_SCANNING_TABLE: "The right arm grabs the package.",
    LABEL_ROTATE_TO_BIN: "Rotate the waist with the right arm.",
    LABEL_PLACE_IN_BLUE_BIN: "Place the package in the blue bin.",
    LABEL_RETURN_INITIAL: "Both arms coordinate and the waist returns to the initial posture.",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build sorting segment prompt classifier dataset")
    parser.add_argument(
        "--dataset-roots",
        nargs="+",
        default=[
            "/mnt/a/ljz/.tmp/agibot_r2a_lerobot/sorting_packages_part_1",
            "/mnt/a/ljz/.tmp/agibot_r2a_lerobot/sorting_packages_part_2",
            "/mnt/a/ljz/.tmp/agibot_r2a_lerobot/sorting_packages_part_3",
        ],
        help="LeRobot sorting package dataset roots",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/sorting_segment_prompt_dataset",
        help="Output directory with train/val class folders",
    )
    parser.add_argument("--camera-key", default="observation.images.top_head")
    parser.add_argument("--frame-stride", type=int, default=10)
    parser.add_argument("--max-frames-per-segment", type=int, default=40)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--trim-ratio", type=float, default=0.05)
    parser.add_argument(
        "--place-scan-window",
        default="0.30,0.85",
        help=(
            "Normalized segment window for place_on_scanning_table samples. "
            "Use middle/late frames to reduce confusion with grab_from_scanning_table."
        ),
    )
    parser.add_argument(
        "--grab-scan-window",
        default="0.25,0.75",
        help=(
            "Normalized segment window for grab_from_scanning_table samples. "
            "Use middle frames to reduce boundary ambiguity."
        ),
    )
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument(
        "--unknown-policy",
        choices=("error", "skip"),
        default="error",
        help="How to handle instructions that do not map to a segment label",
    )
    return parser.parse_args()


def extract_color(instruction: str) -> str | None:
    match = COLOR_PATTERN.search(instruction)
    if match is None:
        return None
    return match.group(1).lower()


def prompt_for_label(label: str, instruction: str) -> str:
    prompt = STAGE_PROMPTS[label]
    color = extract_color(instruction) or "<color>"
    return prompt.replace("<color>", color)


def segment_label(instruction: str) -> str | None:
    text = " ".join(instruction.strip().lower().split())

    if "package on the table" in text and "grab" in text:
        return LABEL_GRAB_TABLE
    if "barcode scanner" in text and "turn" in text and "waist" in text:
        return LABEL_TURN_TO_SCANNER
    if ("scanning table" in text or "scanning platform" in text) and ("place" in text or "put" in text):
        return LABEL_PLACE_ON_SCANNING_TABLE
    if re.search(r"\bright arm grab(?:s|bed)?\b", text):
        return LABEL_GRAB_FROM_SCANNING_TABLE
    if "rotate" in text and "waist" in text:
        return LABEL_ROTATE_TO_BIN
    if ("blue bin" in text or "blue box" in text) and ("place" in text or "put" in text):
        return LABEL_PLACE_IN_BLUE_BIN
    if "initial posture" in text or ("return" in text and "initial" in text):
        return LABEL_RETURN_INITIAL

    return None


def split_from_episode_key(key: str, val_ratio: float) -> str:
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()
    score = int(digest[:8], 16) / float(16**8 - 1)
    return "val" if score < val_ratio else "train"


def ensure_output_dirs(output_root: Path) -> None:
    for split in ("train", "val"):
        for label in LABELS:
            (output_root / split / label).mkdir(parents=True, exist_ok=True)


def parse_fraction_window(raw: str, name: str) -> tuple[float, float]:
    try:
        start_s, end_s = raw.split(",", maxsplit=1)
        start = float(start_s)
        end = float(end_s)
    except ValueError as exc:
        raise ValueError(f"{name} must be formatted as start,end, got {raw!r}") from exc

    if not (0.0 <= start < end <= 1.0):
        raise ValueError(f"{name} must satisfy 0 <= start < end <= 1, got {raw!r}")
    return start, end


def sampling_window(
    start: int,
    end: int,
    *,
    label: str,
    is_final_segment: bool,
    trim_ratio: float,
    place_scan_window: tuple[float, float],
    grab_scan_window: tuple[float, float],
) -> tuple[int, int]:
    end_exclusive = end + 1 if is_final_segment else end
    if end_exclusive <= start:
        end_exclusive = start + 1

    length = end_exclusive - start
    if label == LABEL_PLACE_ON_SCANNING_TABLE:
        frac_start, frac_end = place_scan_window
        return start + int(round(length * frac_start)), start + max(1, int(round(length * frac_end)))

    if label == LABEL_GRAB_FROM_SCANNING_TABLE:
        frac_start, frac_end = grab_scan_window
        return start + int(round(length * frac_start)), start + max(1, int(round(length * frac_end)))

    if length < 20 or trim_ratio <= 0:
        return start, end_exclusive

    trim = int(round(length * trim_ratio))
    trimmed_start = start + trim
    trimmed_end = end_exclusive - trim
    if trimmed_end <= trimmed_start:
        return start, end_exclusive
    return trimmed_start, trimmed_end


def sample_indices(start: int, end_exclusive: int, stride: int, max_frames: int) -> list[int]:
    if end_exclusive <= start:
        return []

    indices = list(range(start, end_exclusive, stride))
    last = end_exclusive - 1
    if last not in indices:
        indices.append(last)

    if max_frames > 0 and len(indices) > max_frames:
        if max_frames == 1:
            keep = [0]
        else:
            keep = [
                round(i * (len(indices) - 1) / (max_frames - 1))
                for i in range(max_frames)
            ]
        indices = [indices[i] for i in sorted(set(keep))]

    return indices


def iter_selected_video_frames(video_path: Path, selected_indices: set[int]):
    if not selected_indices:
        return

    try:
        import cv2  # type: ignore

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        frame_idx = 0
        try:
            while True:
                ok, frame_bgr = cap.read()
                if not ok:
                    break
                if frame_idx in selected_indices:
                    yield frame_idx, cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                frame_idx += 1
        finally:
            cap.release()
        return
    except ImportError:
        pass

    try:
        import imageio.v3 as iio  # type: ignore
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(
            "Video decoding requires either opencv-python or imageio[ffmpeg]. "
            "Install project dependencies or run inside the OpenPI uv environment."
        ) from exc

    for frame_idx, frame in enumerate(iio.imiter(video_path)):
        if frame_idx in selected_indices:
            frame_rgb = np.asarray(frame)
            if frame_rgb.ndim == 3 and frame_rgb.shape[-1] == 4:
                frame_rgb = frame_rgb[..., :3]
            yield frame_idx, frame_rgb


def save_rgb_jpeg(path: Path, frame_rgb, quality: int) -> None:
    try:
        from PIL import Image

        Image.fromarray(frame_rgb).save(path, format="JPEG", quality=quality)
        return
    except ImportError:
        pass

    try:
        import cv2  # type: ignore

        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        ok = cv2.imwrite(str(path), frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
        if not ok:
            raise RuntimeError(f"cv2.imwrite failed for {path}")
        return
    except ImportError:
        pass

    try:
        import imageio.v3 as iio  # type: ignore

        iio.imwrite(path, frame_rgb, quality=quality)
        return
    except ImportError as exc:
        raise RuntimeError(
            "Saving JPEG frames requires pillow, opencv-python, or imageio. "
            "Install one of them in the environment used to run this script."
        ) from exc


def build_episode_frame_records(
    segments: list[dict],
    *,
    frame_stride: int,
    max_frames_per_segment: int,
    trim_ratio: float,
    place_scan_window: tuple[float, float],
    grab_scan_window: tuple[float, float],
    unknown_policy: str,
) -> tuple[dict[int, dict], Counter]:
    frame_records: dict[int, dict] = {}
    unknown_counter: Counter = Counter()

    for segment_index, segment in enumerate(segments):
        instruction = str(segment.get("instruction", "")).strip()
        label = segment_label(instruction)
        if label is None:
            unknown_counter[instruction] += 1
            if unknown_policy == "error":
                raise ValueError(f"Unknown sorting segment instruction: {instruction!r}")
            continue

        start = int(segment.get("start_frame_index", 0))
        end = int(segment.get("end_frame_index", segment.get("success_frame_index", start)))
        sample_start, sample_end = sampling_window(
            start,
            end,
            label=label,
            is_final_segment=(segment_index == len(segments) - 1),
            trim_ratio=trim_ratio,
            place_scan_window=place_scan_window,
            grab_scan_window=grab_scan_window,
        )

        for frame_idx in sample_indices(sample_start, sample_end, frame_stride, max_frames_per_segment):
            frame_records[frame_idx] = {
                "label": label,
                "segment_index": segment_index,
                "start_frame": start,
                "end_frame": end,
                "sample_start": sample_start,
                "sample_end": sample_end,
                "instruction": instruction,
                "prompt": prompt_for_label(label, instruction),
            }

    return frame_records, unknown_counter


def process_dataset(
    dataset_root: Path,
    output_root: Path,
    *,
    camera_key: str,
    frame_stride: int,
    max_frames_per_segment: int,
    val_ratio: float,
    trim_ratio: float,
    place_scan_window: tuple[float, float],
    grab_scan_window: tuple[float, float],
    jpeg_quality: int,
    unknown_policy: str,
    metadata_writer: csv.writer,
    class_counter: Counter,
    unknown_counter: Counter,
) -> None:
    info_path = dataset_root / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"Missing info.json: {info_path}")

    info = json.loads(info_path.read_text(encoding="utf-8"))
    instruction_segments = info.get("instruction_segments", {})
    dataset_name = dataset_root.name
    episode_keys = sorted(instruction_segments.keys(), key=lambda x: int(x))

    pbar = tqdm(episode_keys, desc=f"Extracting {dataset_name}", dynamic_ncols=True)
    for ep_key in pbar:
        episode_idx = int(ep_key)
        split = split_from_episode_key(f"{dataset_name}:{episode_idx}", val_ratio)
        segments = instruction_segments[ep_key]
        frame_records, episode_unknown = build_episode_frame_records(
            segments,
            frame_stride=frame_stride,
            max_frames_per_segment=max_frames_per_segment,
            trim_ratio=trim_ratio,
            place_scan_window=place_scan_window,
            grab_scan_window=grab_scan_window,
            unknown_policy=unknown_policy,
        )
        unknown_counter.update(episode_unknown)
        if not frame_records:
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

        saved = 0
        selected_indices = set(frame_records)
        try:
            frame_iter = iter_selected_video_frames(video_path, selected_indices)
            for frame_idx, frame_rgb in frame_iter:
                record = frame_records[frame_idx]
                label = record["label"]
                out_name = (
                    f"{dataset_name}_ep{episode_idx:06d}_seg{record['segment_index']:02d}_"
                    f"f{frame_idx:06d}.jpg"
                )
                out_path = output_root / split / label / out_name
                save_rgb_jpeg(out_path, frame_rgb, jpeg_quality)

                class_counter[(split, label)] += 1
                metadata_writer.writerow(
                    [
                        split,
                        label,
                        dataset_name,
                        episode_idx,
                        record["segment_index"],
                        record["start_frame"],
                        record["end_frame"],
                        record["sample_start"],
                        record["sample_end"],
                        frame_idx,
                        str(out_path.relative_to(output_root)),
                        record["instruction"],
                        record["prompt"],
                    ]
                )
                saved += 1
        except RuntimeError as exc:
            pbar.write(f"[WARN] {exc}")
            continue

        pbar.set_postfix(saved=saved)


def main() -> None:
    args = parse_args()
    if args.frame_stride <= 0:
        raise ValueError("--frame-stride must be > 0")
    if args.max_frames_per_segment <= 0:
        raise ValueError("--max-frames-per-segment must be > 0")
    if not (0.0 < args.val_ratio < 1.0):
        raise ValueError("--val-ratio must be in (0,1)")
    if not (0.0 <= args.trim_ratio < 0.5):
        raise ValueError("--trim-ratio must be in [0,0.5)")
    place_scan_window = parse_fraction_window(args.place_scan_window, "--place-scan-window")
    grab_scan_window = parse_fraction_window(args.grab_scan_window, "--grab-scan-window")

    output_root = Path(args.output_dir)
    ensure_output_dirs(output_root)

    metadata_path = output_root / "metadata.csv"
    class_counter: Counter = Counter()
    unknown_counter: Counter = Counter()

    with metadata_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "split",
                "label",
                "dataset",
                "episode",
                "segment_index",
                "start_frame",
                "end_frame",
                "sample_start",
                "sample_end",
                "frame",
                "path",
                "instruction",
                "prompt",
            ]
        )

        for root in args.dataset_roots:
            process_dataset(
                dataset_root=Path(root),
                output_root=output_root,
                camera_key=args.camera_key,
                frame_stride=args.frame_stride,
                max_frames_per_segment=args.max_frames_per_segment,
                val_ratio=args.val_ratio,
                trim_ratio=args.trim_ratio,
                place_scan_window=place_scan_window,
                grab_scan_window=grab_scan_window,
                jpeg_quality=args.jpeg_quality,
                unknown_policy=args.unknown_policy,
                metadata_writer=writer,
                class_counter=class_counter,
                unknown_counter=unknown_counter,
            )

    stats = {
        "labels": LABELS,
        "stage_prompts": STAGE_PROMPTS,
        "camera_key": args.camera_key,
        "frame_stride": args.frame_stride,
        "max_frames_per_segment": args.max_frames_per_segment,
        "val_ratio": args.val_ratio,
        "trim_ratio": args.trim_ratio,
        "place_scan_window": place_scan_window,
        "grab_scan_window": grab_scan_window,
        "counts": {
            split: {label: class_counter[(split, label)] for label in LABELS}
            for split in ("train", "val")
        },
        "unknown_instructions": dict(unknown_counter),
    }

    stats_path = output_root / "stats.json"
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print(f"Dataset built at: {output_root}")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
