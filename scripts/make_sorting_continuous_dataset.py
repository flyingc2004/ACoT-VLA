#!/usr/bin/env python3
"""Build a pseudo-continuous sorting dataset from existing single-package LeRobot episodes.

This script creates a new LeRobot dataset (v2.1 style) by concatenating multiple
existing sorting episodes into one longer episode:
- parquet rows are concatenated and re-indexed
- per-camera mp4 files are concatenated with ffmpeg
- meta files are regenerated (info/tasks/episodes/episodes_stats)

The goal is minimal-cost expansion for "sorting_packages_continuous" training.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


REQUIRED_COLUMNS = [
    "observation.state",
    "action",
    "episode_index",
    "frame_index",
    "index",
    "task_index",
    "timestamp",
]

RESET_KEYWORDS = ("reset", "return", "default")


@dataclass(frozen=True)
class EpisodeRef:
    dataset_dir: Path
    dataset_name: str
    episode_index: int
    length: int
    parquet_path: Path
    video_paths: dict[str, Path]
    instruction_segments: list[dict[str, Any]]


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")


def gather_video_keys(info: dict[str, Any]) -> list[str]:
    video_keys = []
    for key, spec in info.get("features", {}).items():
        if spec.get("dtype") == "video":
            video_keys.append(key)
    if not video_keys:
        raise ValueError("No video features found in info.json; expected sorting dataset with videos.")
    return sorted(video_keys)


def build_episode_refs(source_dirs: list[Path]) -> tuple[list[EpisodeRef], dict[str, Any], dict[str, Any] | None]:
    episode_refs: list[EpisodeRef] = []
    info_template: dict[str, Any] | None = None
    stats_template: dict[str, Any] | None = None

    for source_dir in source_dirs:
        meta_dir = source_dir / "meta"
        info_path = meta_dir / "info.json"
        episodes_path = meta_dir / "episodes.jsonl"
        stats_path = meta_dir / "episodes_stats.jsonl"

        if not info_path.exists() or not episodes_path.exists():
            raise FileNotFoundError(f"Missing meta files in {source_dir}")

        info = read_json(info_path)
        episodes = read_jsonl(episodes_path)
        instruction_map = info.get("instruction_segments", {})

        if info_template is None:
            info_template = copy.deepcopy(info)

        if stats_template is None and stats_path.exists():
            stats_rows = read_jsonl(stats_path)
            if stats_rows:
                stats_template = copy.deepcopy(stats_rows[0].get("stats", {}))

        video_keys = gather_video_keys(info)
        chunks_size = int(info.get("chunks_size", 1000))
        data_tpl = info["data_path"]
        video_tpl = info["video_path"]

        for ep in episodes:
            ep_idx = int(ep["episode_index"])
            length = int(ep["length"])
            ep_chunk = ep_idx // chunks_size

            parquet_rel = data_tpl.format(episode_chunk=ep_chunk, episode_index=ep_idx)
            parquet_path = source_dir / parquet_rel
            if not parquet_path.exists():
                raise FileNotFoundError(f"Parquet not found: {parquet_path}")

            video_paths: dict[str, Path] = {}
            for vid_key in video_keys:
                video_rel = video_tpl.format(
                    episode_chunk=ep_chunk,
                    video_key=vid_key,
                    episode_index=ep_idx,
                )
                vpath = source_dir / video_rel
                if not vpath.exists():
                    raise FileNotFoundError(f"Video not found: {vpath}")
                video_paths[vid_key] = vpath

            segments = copy.deepcopy(instruction_map.get(str(ep_idx), []))
            episode_refs.append(
                EpisodeRef(
                    dataset_dir=source_dir,
                    dataset_name=source_dir.name,
                    episode_index=ep_idx,
                    length=length,
                    parquet_path=parquet_path,
                    video_paths=video_paths,
                    instruction_segments=segments,
                )
            )

    if info_template is None:
        raise RuntimeError("No source info.json found.")

    episode_refs.sort(key=lambda x: (str(x.dataset_dir), x.episode_index))
    return episode_refs, info_template, stats_template


def build_groups(
    refs: list[EpisodeRef],
    episodes_per_group: int,
    group_stride: int,
    shuffle: bool,
    seed: int,
    max_groups: int | None,
) -> list[list[EpisodeRef]]:
    if episodes_per_group <= 0:
        raise ValueError("episodes_per_group must be > 0")
    if group_stride <= 0:
        raise ValueError("group_stride must be > 0")
    if len(refs) < episodes_per_group:
        raise ValueError(
            f"Not enough source episodes: {len(refs)} < episodes_per_group={episodes_per_group}"
        )

    refs_local = refs[:]
    if shuffle:
        random.Random(seed).shuffle(refs_local)

    groups: list[list[EpisodeRef]] = []
    for i in range(0, len(refs_local) - episodes_per_group + 1, group_stride):
        groups.append(refs_local[i : i + episodes_per_group])

    if max_groups is not None:
        groups = groups[:max_groups]

    if not groups:
        raise ValueError("No groups generated; adjust episodes_per_group/group_stride/max_groups.")

    return groups


def _to_numeric_array(series: pd.Series) -> np.ndarray:
    if series.dtype == object:
        values = series.tolist()
        arr = np.asarray(values)
    else:
        arr = series.to_numpy()

    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr


def _jsonable_list(arr: np.ndarray, as_float: bool = True) -> list[Any]:
    if as_float:
        return np.asarray(arr, dtype=np.float64).tolist()
    return np.asarray(arr).tolist()


def compute_episode_stats(
    frame_df: pd.DataFrame,
    stats_template: dict[str, Any] | None,
    video_keys: list[str],
    feature_specs: dict[str, Any],
) -> dict[str, Any]:
    stats: dict[str, Any] = {}

    for key in REQUIRED_COLUMNS:
        arr = _to_numeric_array(frame_df[key])
        # Keep int-ish values in min/max/count, float in mean/std.
        is_int_like = np.issubdtype(arr.dtype, np.integer)

        stats[key] = {
            "min": _jsonable_list(arr.min(axis=0), as_float=not is_int_like),
            "max": _jsonable_list(arr.max(axis=0), as_float=not is_int_like),
            "mean": _jsonable_list(arr.mean(axis=0), as_float=True),
            "std": _jsonable_list(arr.std(axis=0), as_float=True),
            "count": [int(arr.shape[0])],
        }

    for vid_key in video_keys:
        if stats_template is not None and vid_key in stats_template:
            stats[vid_key] = copy.deepcopy(stats_template[vid_key])
            continue

        channels = int(feature_specs[vid_key]["shape"][-1])
        zero_channel = [[[0.0]] for _ in range(channels)]
        stats[vid_key] = {
            "min": zero_channel,
            "max": zero_channel,
            "mean": zero_channel,
            "std": zero_channel,
            "count": [0],
        }

    return stats


def _escape_ffmpeg_path(path: Path) -> str:
    # ffmpeg concat demuxer: single quotes are escaped as '\''
    return str(path).replace("'", "'\\''")


def concat_videos(
    in_paths: list[Path],
    out_path: Path,
    temp_dir: Path,
    reencode_on_fail: bool,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    list_path = temp_dir / f"{out_path.stem}.txt"

    with list_path.open("w", encoding="utf-8") as f:
        for p in in_paths:
            f.write(f"file '{_escape_ffmpeg_path(p)}'\n")

    cmd_copy = [
        "ffmpeg",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(list_path),
        "-c",
        "copy",
        str(out_path),
    ]

    copy_proc = subprocess.run(
        cmd_copy,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    if copy_proc.returncode == 0:
        return

    if not reencode_on_fail:
        raise RuntimeError(
            f"ffmpeg concat(copy) failed for {out_path}. stderr:\n{copy_proc.stderr[:2000]}"
        )

    cmd_reencode = [
        "ffmpeg",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(list_path),
        "-c:v",
        "libx265",
        "-pix_fmt",
        "yuv420p",
        "-an",
        str(out_path),
    ]

    reenc_proc = subprocess.run(
        cmd_reencode,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    if reenc_proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg concat(reencode) failed for {out_path}. stderr:\n{reenc_proc.stderr[:2000]}"
        )


def build_continuous_dataset(
    source_dirs: list[Path],
    output_dir: Path,
    episodes_per_group: int,
    group_stride: int,
    shuffle: bool,
    seed: int,
    max_groups: int | None,
    task_name: str,
    bridge_instruction: str,
    sanitize_reset_instructions: bool,
    reset_instruction_alias: str,
    overwrite: bool,
    reencode_on_fail: bool,
    dry_run: bool,
) -> None:
    if shutil.which("ffmpeg") is None and not dry_run:
        raise RuntimeError("ffmpeg is not available in PATH.")

    refs, info_template, stats_template = build_episode_refs(source_dirs)
    groups = build_groups(
        refs,
        episodes_per_group=episodes_per_group,
        group_stride=group_stride,
        shuffle=shuffle,
        seed=seed,
        max_groups=max_groups,
    )

    fps = float(info_template["fps"])
    chunks_size = int(info_template.get("chunks_size", 1000))
    video_keys = gather_video_keys(info_template)

    print(f"Source episodes: {len(refs)}")
    print(f"Generated groups: {len(groups)} (episodes_per_group={episodes_per_group}, stride={group_stride})")
    print(f"Output dir: {output_dir}")

    if dry_run:
        for i, group in enumerate(groups[:5]):
            desc = ", ".join(f"{g.dataset_name}:{g.episode_index}" for g in group)
            print(f"group[{i}] -> {desc}")
        print("Dry run only, no files written.")
        return

    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Output directory exists: {output_dir}. Use --overwrite to replace.")
        shutil.rmtree(output_dir)

    (output_dir / "meta").mkdir(parents=True, exist_ok=True)
    (output_dir / "data").mkdir(parents=True, exist_ok=True)
    (output_dir / "videos").mkdir(parents=True, exist_ok=True)

    episodes_rows: list[dict[str, Any]] = []
    episodes_stats_rows: list[dict[str, Any]] = []
    instruction_segments: dict[str, list[dict[str, Any]]] = {}
    reset_segments_sanitized = 0

    global_index_offset = 0

    with tempfile.TemporaryDirectory(prefix="sorting_continuous_concat_") as td:
        temp_dir = Path(td)

        for new_ep_idx, group in enumerate(groups):
            merged_parts: list[pd.DataFrame] = []
            merged_segments: list[dict[str, Any]] = []
            frame_offset = 0

            for src_order, src_ep in enumerate(group):
                part_df = pq.read_table(src_ep.parquet_path).to_pandas()

                missing_cols = [c for c in REQUIRED_COLUMNS if c not in part_df.columns]
                if missing_cols:
                    raise KeyError(
                        f"Missing required columns in {src_ep.parquet_path}: {missing_cols}"
                    )

                part_df["episode_index"] = int(new_ep_idx)
                part_df["task_index"] = 0
                part_df["frame_index"] = np.arange(frame_offset, frame_offset + len(part_df), dtype=np.int64)

                merged_parts.append(part_df)

                for seg in src_ep.instruction_segments:
                    seg_new = copy.deepcopy(seg)
                    if sanitize_reset_instructions:
                        text = str(seg_new.get("instruction", ""))
                        lower = text.lower()
                        if any(k in lower for k in RESET_KEYWORDS):
                            seg_new["instruction"] = reset_instruction_alias
                            reset_segments_sanitized += 1
                    for key in ("start_frame_index", "success_frame_index", "end_frame_index"):
                        if key in seg_new and isinstance(seg_new[key], (int, float)):
                            seg_new[key] = int(seg_new[key]) + frame_offset
                    merged_segments.append(seg_new)

                if src_order < len(group) - 1:
                    boundary = frame_offset + len(part_df) - 1
                    bridge_seg: dict[str, Any] = {
                        "instruction": bridge_instruction,
                        "instruction_augmentation": {},
                        "start_frame_index": boundary,
                        "success_frame_index": boundary,
                        "end_frame_index": boundary,
                    }
                    if merged_segments and "track" in merged_segments[-1]:
                        bridge_seg["track"] = merged_segments[-1]["track"]
                    merged_segments.append(bridge_seg)

                frame_offset += len(part_df)

            merged_df = pd.concat(merged_parts, ignore_index=True)
            ep_len = len(merged_df)

            merged_df["index"] = np.arange(
                global_index_offset,
                global_index_offset + ep_len,
                dtype=np.int64,
            )
            merged_df["timestamp"] = np.arange(ep_len, dtype=np.float32) / fps
            global_index_offset += ep_len

            ep_chunk = new_ep_idx // chunks_size
            data_chunk_dir = output_dir / "data" / f"chunk-{ep_chunk:03d}"
            data_chunk_dir.mkdir(parents=True, exist_ok=True)
            out_parquet = data_chunk_dir / f"episode_{new_ep_idx:06d}.parquet"
            pq.write_table(pa.Table.from_pandas(merged_df, preserve_index=False), out_parquet, compression="zstd")

            for vid_key in video_keys:
                in_videos = [s.video_paths[vid_key] for s in group]
                video_chunk_dir = output_dir / "videos" / f"chunk-{ep_chunk:03d}" / vid_key
                out_video = video_chunk_dir / f"episode_{new_ep_idx:06d}.mp4"
                concat_videos(in_videos, out_video, temp_dir, reencode_on_fail=reencode_on_fail)

            episodes_rows.append(
                {
                    "episode_index": int(new_ep_idx),
                    "tasks": [task_name],
                    "length": int(ep_len),
                }
            )
            instruction_segments[str(new_ep_idx)] = merged_segments

            ep_stats = compute_episode_stats(
                merged_df,
                stats_template=stats_template,
                video_keys=video_keys,
                feature_specs=info_template["features"],
            )
            episodes_stats_rows.append({"episode_index": int(new_ep_idx), "stats": ep_stats})

            if (new_ep_idx + 1) % 10 == 0 or (new_ep_idx + 1) == len(groups):
                print(f"Built {new_ep_idx + 1}/{len(groups)} episodes")

    info_out = copy.deepcopy(info_template)
    total_episodes = len(groups)
    info_out["total_episodes"] = total_episodes
    info_out["total_frames"] = int(global_index_offset)
    info_out["total_tasks"] = 1
    info_out["total_videos"] = total_episodes * len(video_keys)
    info_out["total_chunks"] = max(1, math.ceil(total_episodes / chunks_size)) if total_episodes > 0 else 0
    info_out["splits"] = {"train": f"0:{total_episodes}"}
    info_out["instruction_segments"] = instruction_segments

    write_json(output_dir / "meta" / "info.json", info_out)
    write_jsonl(output_dir / "meta" / "tasks.jsonl", [{"task_index": 0, "task": task_name}])
    write_jsonl(output_dir / "meta" / "episodes.jsonl", episodes_rows)
    write_jsonl(output_dir / "meta" / "episodes_stats.jsonl", episodes_stats_rows)

    print("Done.")
    print(f"Episodes: {total_episodes}")
    print(f"Frames:   {global_index_offset}")
    if sanitize_reset_instructions:
        print(f"Reset-like segments sanitized: {reset_segments_sanitized}")
    print(f"Saved to: {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a pseudo-continuous sorting dataset by concatenating existing sorting episodes."
    )
    parser.add_argument(
        "--source-dirs",
        nargs="+",
        required=True,
        help="One or more source sorting dataset dirs (each containing data/meta/videos).",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output dataset directory.",
    )
    parser.add_argument(
        "--episodes-per-group",
        type=int,
        default=2,
        help="How many source episodes to concatenate into one continuous episode.",
    )
    parser.add_argument(
        "--group-stride",
        type=int,
        default=2,
        help="Sliding stride on source episodes when building groups.",
    )
    parser.add_argument(
        "--max-groups",
        type=int,
        default=None,
        help="Cap generated groups for quick experiments.",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle source episodes before grouping.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--task-name",
        default="Sort logistics parcels continuous",
        help="Task string written to meta/tasks.jsonl and episodes.jsonl.",
    )
    parser.add_argument(
        "--bridge-instruction",
        default="Rotate waist to table-side alignment and prepare the next package grasp.",
        help="Instruction inserted at boundaries between stitched episodes.",
    )
    parser.add_argument(
        "--sanitize-reset-instructions",
        action="store_true",
        default=True,
        help=(
            "Rewrite reset/return/default instructions to avoid subtask sampler truncation "
            "(sampler truncates those to 45 frames)."
        ),
    )
    parser.add_argument(
        "--no-sanitize-reset-instructions",
        dest="sanitize_reset_instructions",
        action="store_false",
        help="Keep original reset-like instruction text.",
    )
    parser.add_argument(
        "--reset-instruction-alias",
        default="Coordinate both arms and waist to align for the next package grasp.",
        help="Replacement text for reset/return/default instructions when sanitization is enabled.",
    )
    parser.add_argument(
        "--reencode-on-fail",
        action="store_true",
        help="If ffmpeg concat copy fails, fallback to h265 re-encode.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output directory if it already exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print grouping plan without writing dataset.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_dirs = [Path(p).expanduser().resolve() for p in args.source_dirs]
    output_dir = Path(args.output_dir).expanduser().resolve()

    build_continuous_dataset(
        source_dirs=source_dirs,
        output_dir=output_dir,
        episodes_per_group=args.episodes_per_group,
        group_stride=args.group_stride,
        shuffle=args.shuffle,
        seed=args.seed,
        max_groups=args.max_groups,
        task_name=args.task_name,
        bridge_instruction=args.bridge_instruction,
        sanitize_reset_instructions=args.sanitize_reset_instructions,
        reset_instruction_alias=args.reset_instruction_alias,
        overwrite=args.overwrite,
        reencode_on_fail=args.reencode_on_fail,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
