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


def _repo_paths() -> list[str]:
    raw = os.environ.get("FIVE_TASK_REPO_PATHS")
    if raw:
        return [p.strip() for p in raw.split(os.pathsep) if p.strip()]
    root = Path(__file__).resolve().parents[1]
    base = Path(
        os.environ.get(
            "R2A_DATASET_ROOT",
            str(root / "datasets" / "Reasoning2Action-Sim" / "dataset_without_depth"),
        )
    ).expanduser()
    return [
        str(base / "pour_workpiece"),
        str(base / "open_door"),
        str(base / "take_wrong_item_shelf"),
        str(base / "scoop_popcorn"),
        str(base / "scoop_popcorn_part_2"),
        str(base / "hold_pot"),
    ]


def build_config(*, exp_name: str, overwrite: bool, resume: bool, batch_size: int, num_workers: int):
    _ensure_openpi_path()
    from openpi.training import config as cfg
    from openpi.training import weight_loaders

    base = cfg.get_config("acot_icra_simulation_challenge_reasoning_to_action")
    repos = _repo_paths()

    prompt_subset_keys = (
        "Unload workpiece_icra_SIM",
        "Turn the doorknob",
        "Remove misplaced beverages from shelves",
        "Make popcorn",
        "Carry the pot",
    )
    full_map = base.data.prompt_map_inject_to_training
    prompt_subset = {k: full_map[k] for k in prompt_subset_keys if k in full_map}

    baseline_dir = Path(os.environ.get("ACOT_BASELINE_CHECKPOINT_DIR", "./checkpoints/baseline/30000")).expanduser()
    baseline_params = os.environ.get("BASELINE_PARAMS", str(baseline_dir / "params"))
    baseline_assets = os.environ.get("BASELINE_NORM_ASSETS_DIR", str(baseline_dir / "assets"))

    data = dataclasses.replace(
        base.data,
        repo_id=repos,
        assets=dataclasses.replace(base.data.assets, assets_dir=baseline_assets, asset_id="."),
        prompt_map_inject_to_training=prompt_subset,
    )

    return dataclasses.replace(
        base,
        exp_name=exp_name,
        overwrite=overwrite,
        resume=resume,
        data=data,
        weight_loader=weight_loaders.ACOTCheckpointWeightLoader(baseline_params),
        batch_size=batch_size,
        num_workers=num_workers,
    )


def main() -> None:
    root = _ensure_openpi_path()
    import tyro

    @dataclasses.dataclass
    class Args:
        exp_name: str = "baseline_5tasks"
        overwrite: bool = True
        resume: bool = False
        batch_size: int = 256
        num_workers: int = 24
        num_train_steps: int = 20_000
        save_interval: int = 5_000
        warmup_steps: int = 0

    args = tyro.cli(Args)
    if args.resume and args.overwrite:
        raise ValueError("Cannot set both resume and overwrite.")

    config = build_config(
        exp_name=args.exp_name,
        overwrite=args.overwrite,
        resume=args.resume,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    config = dataclasses.replace(
        config,
        num_train_steps=args.num_train_steps,
        save_interval=args.save_interval,
        lr_schedule=dataclasses.replace(config.lr_schedule, warmup_steps=args.warmup_steps),
    )

    train_path = root / "scripts" / "train.py"
    spec = importlib.util.spec_from_file_location("_yrm_train", train_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load {train_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.main(config)


if __name__ == "__main__":
    main()
