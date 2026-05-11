"""Compute normalization statistics for a config.

This script is used to compute the normalization statistics for a given config. It
will compute the mean and standard deviation of the data in the dataset and save it
to the config assets directory.
"""

import numpy as np
import pathlib
import tqdm
import tyro
import random
import dataclasses
import openpi.models.model as _model
import openpi.shared.normalize as normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as transforms

class RemoveStrings(transforms.DataTransformFn):
    def __call__(self, x: dict) -> dict:
        return {k: v for k, v in x.items() if not np.issubdtype(np.asarray(v).dtype, np.str_)}


def _disable_visual_feature_loading(dataset) -> None:
    """Avoid decoding videos/images when norm stats only need numeric features."""
    if hasattr(dataset, "meta") and hasattr(dataset.meta, "info"):
        features = dataset.meta.info.get("features", {})
        for key, feature in list(features.items()):
            if feature.get("dtype") in ("image", "video"):
                del features[key]

    if hasattr(dataset, "_dataset"):
        _disable_visual_feature_loading(dataset._dataset)
    if hasattr(dataset, "_datasets"):
        for child in dataset._datasets:
            _disable_visual_feature_loading(child)


def _drop_visual_repack_fields(structure):
    if isinstance(structure, dict):
        result = {}
        for key, value in structure.items():
            if key in ("image", "images", "image_mask"):
                continue
            pruned = _drop_visual_repack_fields(value)
            if pruned not in ({}, None):
                result[key] = pruned
        return result
    return structure


def _norm_stat_transform(transform: transforms.DataTransformFn) -> transforms.DataTransformFn:
    if isinstance(transform, transforms.RepackTransform):
        return transforms.RepackTransform(_drop_visual_repack_fields(transform.structure))
    if dataclasses.is_dataclass(transform) and hasattr(transform, "require_images"):
        return dataclasses.replace(transform, require_images=False)
    return transform


def create_torch_dataloader(
    data_config: _config.DataConfig,
    batch_size: int,
    model_config: _model.BaseModelConfig,
    max_frames: int | None = None,
    num_workers: int = 0,
) -> tuple[_data_loader.Dataset, int]:
    if data_config.repo_id is None:
        raise ValueError("Data config must have a repo_id")
    dataset = _data_loader.create_torch_dataset(data_config, model_config)
    _disable_visual_feature_loading(dataset)
    dataset = _data_loader.TransformedDataset(
        dataset,
        [
            *[_norm_stat_transform(transform) for transform in data_config.repack_transforms.inputs],
            *[_norm_stat_transform(transform) for transform in data_config.data_transforms.inputs],
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
    )

    sampler = None
    available_frames = len(dataset)
    shuffle = False
    if data_config.dataloader_sampler:
        from openpi.training.sampler import FrameSampler

        sampler = FrameSampler(
            dataset,
            data_config.dataloader_sampler,
            reset_truncation_mode=data_config.subtask_reset_truncation_mode,
        )
        available_frames = len(sampler)
    elif max_frames is not None and max_frames < len(dataset):
        shuffle = True

    requested_frames = min(max_frames, available_frames) if max_frames is not None else available_frames
    if 0 < requested_frames < batch_size <= available_frames:
        num_batches = 1
    else:
        num_batches = requested_frames // batch_size

    dataset = _data_loader.SafeDataset(dataset)
    
    data_loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        num_batches=num_batches,
        sampler=sampler,
    )
    return data_loader, num_batches


def create_rlds_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    dataset = _data_loader.create_rlds_dataset(data_config, action_horizon, batch_size, shuffle=False)
    dataset = _data_loader.IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
        is_batched=True,
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
    else:
        num_batches = len(dataset) // batch_size
    data_loader = _data_loader.RLDSDataLoader(
        dataset,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def main(config_name: str, max_frames: int | None = None, num_workers: int = 0):
    config = _config.get_config(config_name)
    data_config = config.data.create(config.assets_dirs, config.model)

    if data_config.rlds_data_dir is not None:
        data_loader, num_batches = create_rlds_dataloader(
            data_config, config.model.action_horizon, config.batch_size, max_frames
        )
    else:
        data_loader, num_batches = create_torch_dataloader(
            data_config, config.batch_size, config.model, max_frames, num_workers
        )

    candidate_keys = ["state", "actions", "coarse_actions"]
    stats = {key: normalize.RunningStats() for key in candidate_keys}
    active_keys = set()

    sample_ratio = 0.1
    if num_batches <= 0:
        raise RuntimeError("No batches available while computing normalization stats.")
    max_batches = max(1, int(num_batches * sample_ratio))

    data_iter = iter(data_loader)
    pbar = tqdm.tqdm(total=max_batches, desc="Computing stats")
    valid_batches = 0
    skipped_batches = 0
    last_error = None
    while valid_batches < max_batches:
        try:
            batch = next(data_iter)
        except StopIteration:
            break
        except Exception as e:
            skipped_batches += 1
            last_error = e
            print(f"\n[Warning] Skipped a bad batch due to error: {e}")
            continue

        updated = False
        for key in candidate_keys:
            if key not in batch:
                continue
            values = np.asarray(batch[key])
            if values.size == 0:
                continue
            stats[key].update(values.reshape(-1, values.shape[-1]))
            active_keys.add(key)
            updated = True

        if not updated:
            skipped_batches += 1
            print(f"\n[Warning] Skipped a batch without any of {candidate_keys}.")
            continue

        pbar.update(1)
        valid_batches += 1

    pbar.close()

    if valid_batches == 0:
        detail = f" Last batch error: {last_error}" if last_error is not None else ""
        raise RuntimeError(
            "No valid batches were collected while computing normalization stats. "
            f"Skipped batches: {skipped_batches}. Please check dataset paths and data integrity."
            f"{detail}"
        )

    norm_stats = {key: stats[key].get_statistics() for key in candidate_keys if key in active_keys}

    assets = getattr(config.data, "assets", None)
    asset_id = None
    if hasattr(config, "data") and hasattr(config.data, "assets"):
        asset_id = getattr(config.data.assets, "asset_id", None)

    assets_dir = pathlib.Path(getattr(assets, "assets_dir", None) or config.assets_dirs).expanduser()
    output_dir = assets_dir if asset_id in (None, ".") else assets_dir / asset_id
    print(f"Writing stats to: {output_dir}")
    normalize.save(output_dir, norm_stats)

if __name__ == "__main__":
    tyro.cli(main)
