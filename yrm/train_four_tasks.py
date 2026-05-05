"""
四任务联合微调入口（在 baseline 权重上继续训练）。

用法（在 ACoT-VLA 仓库根目录）:
  uv run python yrm/train_four_tasks.py --exp-name=my_run --overwrite

环境变量可与 scripts/train.sh 一致（如 WANDB_MODE、XLA_PYTHON_CLIENT_MEM_FRACTION）。

数据前提: repo 路径下需为已解压的 LeRobot 数据集（含 meta/、data/ 等），而非仅 *.tar.gz 分片。
"""

from __future__ import annotations

import dataclasses
import importlib.util
import os
import sys
from pathlib import Path


def _ensure_openpi_path() -> Path:
    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    s = str(src)
    if s not in sys.path:
        sys.path.insert(0, s)
    return root

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DATASET_ROOT = Path(
    os.environ.get(
        "R2A_DATASET_ROOT",
        str(_REPO_ROOT / "datasets" / "Reasoning2Action-Sim" / "dataset_without_depth"),
    )
).expanduser()

# Task_3 / Task_4 / Task_6 / Task_9 + open_door
_DEFAULT_REPO_PATHS: tuple[str, ...] = (
    str(_DATASET_ROOT / "pour_workpiece"),
    str(_DATASET_ROOT / "open_door"),
    str(_DATASET_ROOT / "take_wrong_item_shelf"),
    str(_DATASET_ROOT / "scoop_popcorn"),
    str(_DATASET_ROOT / "hold_pot"),
)

# 若 Task_6 需包含第二部分数据，取消下一行注释并并入列表:
_DEFAULT_REPO_PATHS = _DEFAULT_REPO_PATHS + (str(_DATASET_ROOT / "scoop_popcorn_part_2"),)

_BASELINE_PARAMS = os.environ.get(
    "BASELINE_PARAMS",
    str(Path(os.environ.get("ACOT_BASELINE_CHECKPOINT_DIR", "./checkpoints/baseline/30000")).expanduser() / "params"),
)
# baseline 同一步保存的 norm_stats.json，与九任务全量统计一致；仅训四任务时更稳妥做法是另行 compute_norm_stats
_BASELINE_NORM_ASSETS_DIR = os.environ.get(
    "BASELINE_NORM_ASSETS_DIR",
    str(Path(os.environ.get("ACOT_BASELINE_CHECKPOINT_DIR", "./checkpoints/baseline/30000")).expanduser() / "assets"),
)


def _repo_paths_from_env() -> list[str]:
    raw = os.environ.get("FOUR_TASK_REPO_PATHS")
    if raw:
        return [p.strip() for p in raw.split(os.pathsep) if p.strip()]
    return list(_DEFAULT_REPO_PATHS)


def build_config(
    *,
    exp_name: str,
    overwrite: bool = False,
    resume: bool = False,
    repo_paths: list[str] | None = None,
) -> "TrainConfig":
    _ensure_openpi_path()
    from openpi.training import config as cfg
    from openpi.training import weight_loaders

    base = cfg.get_config("acot_icra_simulation_challenge_reasoning_to_action")
    repos = repo_paths if repo_paths is not None else _repo_paths_from_env()

    # 仅保留与四任务相关的 prompt 注入（与 config.py 中 acot_icra 配置一致）
    full_map = base.data.prompt_map_inject_to_training
    four_keys = (
        "Unload workpiece_icra_SIM",
        "Turn the doorknob",
        "Remove misplaced beverages from shelves",
        "Make popcorn",
        "Carry the pot",
    )
    prompt_subset = {k: full_map[k] for k in four_keys if k in full_map}

    new_data = dataclasses.replace(
        base.data,
        repo_id=repos,
        assets=dataclasses.replace(
            base.data.assets,
            assets_dir=_BASELINE_NORM_ASSETS_DIR,
            asset_id=".",
        ),
        prompt_map_inject_to_training=prompt_subset,
    )

    return dataclasses.replace(
        base,
        exp_name=exp_name,
        overwrite=overwrite,
        resume=resume,
        weight_loader=weight_loaders.ACOTCheckpointWeightLoader(_BASELINE_PARAMS),
        data=new_data,
    )


def main() -> None:
    root = _ensure_openpi_path()
    import tyro

    @dataclasses.dataclass
    class Args:
        exp_name: str
        overwrite: bool = False
        resume: bool = False
        batch_size: int = 256
        num_workers: int = 24
        num_train_steps: int = 20_000
        save_interval: int = 5_000
        warmup_steps: int = 0

    args = tyro.cli(Args)
    if args.resume and args.overwrite:
        raise ValueError("不能同时指定 resume 与 overwrite（与 TrainConfig 一致）。")
    config = build_config(
        exp_name=args.exp_name,
        overwrite=args.overwrite,
        resume=args.resume,
    )
    config = dataclasses.replace(
        config,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_train_steps=args.num_train_steps,
        save_interval=args.save_interval,
        lr_schedule=dataclasses.replace(config.lr_schedule, warmup_steps=args.warmup_steps),
    )
    train_path = root / "scripts" / "train.py"
    spec = importlib.util.spec_from_file_location("_yrm_train", train_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载 {train_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.main(config)


if __name__ == "__main__":
    main()
