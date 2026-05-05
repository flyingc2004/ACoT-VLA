from __future__ import annotations

import dataclasses
import os
import sys
from pathlib import Path


def _ensure_src_path() -> None:
    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def _build_config(
    *,
    exp_name: str,
    overwrite: bool,
    resume: bool,
    batch_size: int,
    num_workers: int,
    num_train_steps: int,
    save_interval: int,
    warmup_steps: int,
):
    _ensure_src_path()
    from openpi.training import config as cfg
    from openpi.training import weight_loaders

    base = cfg.get_config("acot_icra_simulation_challenge_reasoning_to_action")

    root = Path(__file__).resolve().parents[1]
    dataset_root = Path(
        os.environ.get(
            "R2A_DATASET_ROOT",
            str(root / "datasets" / "Reasoning2Action-Sim" / "dataset_without_depth"),
        )
    ).expanduser()
    default_repos = os.pathsep.join(
        str(dataset_root / name)
        for name in (
            "pour_workpiece",
            "open_door",
            "take_wrong_item_shelf",
            "scoop_popcorn",
            "scoop_popcorn_part_2",
            "hold_pot",
        )
    )
    repo_paths = [p for p in os.environ.get("FIVE_TASK_REPO_PATHS", default_repos).split(os.pathsep) if p]

    baseline_dir = Path(os.environ.get("ACOT_BASELINE_CHECKPOINT_DIR", "./checkpoints/baseline/30000")).expanduser()
    baseline_params = os.environ.get("BASELINE_PARAMS", str(baseline_dir / "params"))
    baseline_assets = os.environ.get("BASELINE_NORM_ASSETS_DIR", str(baseline_dir / "assets"))

    full_map = base.data.prompt_map_inject_to_training
    keep_keys = (
        "Unload workpiece_icra_SIM",
        "Turn the doorknob",
        "Remove misplaced beverages from shelves",
        "Make popcorn",
        "Carry the pot",
    )
    prompt_subset = {k: full_map[k] for k in keep_keys if k in full_map}

    new_data = dataclasses.replace(
        base.data,
        repo_id=repo_paths,
        assets=dataclasses.replace(base.data.assets, assets_dir=baseline_assets, asset_id="."),
        prompt_map_inject_to_training=prompt_subset,
    )

    cfg_out = dataclasses.replace(
        base,
        exp_name=exp_name,
        overwrite=overwrite,
        resume=resume,
        data=new_data,
        weight_loader=weight_loaders.ACOTCheckpointWeightLoader(baseline_params),
        batch_size=batch_size,
        num_workers=num_workers,
        num_train_steps=num_train_steps,
        save_interval=save_interval,
        lr_schedule=dataclasses.replace(base.lr_schedule, warmup_steps=warmup_steps),
    )
    return cfg_out


def main() -> None:
    _ensure_src_path()
    import tyro
    import importlib.util

    @dataclasses.dataclass
    class Args:
        exp_name: str = "baseline_5tasks_original_style"
        overwrite: bool = True
        resume: bool = False
        batch_size: int = 16
        num_workers: int = 8
        num_train_steps: int = 20_000
        save_interval: int = 5_000
        warmup_steps: int = 0

    args = tyro.cli(Args)
    if args.resume and args.overwrite:
        raise ValueError("Cannot set both resume and overwrite.")

    config = _build_config(
        exp_name=args.exp_name,
        overwrite=args.overwrite,
        resume=args.resume,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_train_steps=args.num_train_steps,
        save_interval=args.save_interval,
        warmup_steps=args.warmup_steps,
    )
    root = Path(__file__).resolve().parents[1]
    train_path = root / "scripts" / "train.py"
    spec = importlib.util.spec_from_file_location("_original_train", train_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load {train_path}")
    original_train = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original_train)
    original_train.main(config)


if __name__ == "__main__":
    main()
