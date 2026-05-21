
import torch
import lerobot.datasets.lerobot_dataset as lerobot_dataset


RESET_KEYWORDS = ("reset", "return", "default")
CONTINUOUS_KEYWORDS = ("packages", "sort")
RESET_TRUNCATE_THRESHOLD = 90
RESET_TRUNCATE_TO = 45
RESET_TRUNCATION_MODES = ("auto", "always", "never")


def _instruction_weight(instruction: str, keyword_multipliers: dict[str, int] | None) -> int:
    if not keyword_multipliers:
        return 1

    weight = 1
    instruction = instruction.lower()
    for keyword, multiplier in keyword_multipliers.items():
        if keyword.lower() in instruction:
            weight = max(weight, int(multiplier))
    return max(1, weight)


def get_base_dataset(ds):
    if hasattr(ds, "_dataset"):
        return get_base_dataset(ds._dataset)
    return ds


def _iter_string_values(obj):
    if obj is None:
        return
    if isinstance(obj, str):
        yield obj
        return
    if isinstance(obj, dict):
        for value in obj.values():
            yield from _iter_string_values(value)
        return
    if isinstance(obj, (list, tuple, set)):
        for value in obj:
            yield from _iter_string_values(value)

def sample_subtask(dataset, *, keyword_multipliers: dict[str, int] | None = None):
    valid_intervals = []
    base_ds = get_base_dataset(dataset)
    
    sub_datasets = []
    
    if isinstance(base_ds, lerobot_dataset.MultiLeRobotDataset):
        for sub_ds in base_ds._datasets:
            sub_datasets.append(sub_ds)
    else:
        sub_datasets.append(dataset)

    current_global_offset = 0
    total_episodes_processed = 0

    print(f"Processing {len(sub_datasets)} sub-datasets...")

    for sub_ds in sub_datasets:
        inner_ds = get_base_dataset(sub_ds)
        instruction_segment = inner_ds.meta.info.get('instruction_segments', {})
        episode_data_index = inner_ds.episode_data_index
        num_episodes = len(episode_data_index['from'])
        
        for ep_idx in range(num_episodes):
            local_episode_start = episode_data_index['from'][ep_idx].item()
            disable_reset_truncation = True
            if str(ep_idx) not in instruction_segment:
                continue

            tasks = instruction_segment[str(ep_idx)]
            for subtask in tasks:
                local_start = subtask["start_frame_index"] + local_episode_start
                local_end = subtask["success_frame_index"] + local_episode_start
                
                instruction = subtask["instruction"].lower()
                weight = _instruction_weight(instruction, keyword_multipliers)
                if any(k in instruction for k in CONTINUOUS_KEYWORDS):
                    disable_reset_truncation = False
                is_reset = any(k in instruction for k in RESET_KEYWORDS)
                
                if is_reset and not disable_reset_truncation:
                    if local_end - local_start > RESET_TRUNCATE_THRESHOLD:
                        local_end = local_start + RESET_TRUNCATE_TO
                
                global_start = local_start + current_global_offset
                global_end = local_end + current_global_offset
                
                valid_intervals.append((global_start, global_end, weight))
        
        current_global_offset += len(sub_ds)
        total_episodes_processed += num_episodes

    print(f"Total {len(valid_intervals)} valid intervals from {total_episodes_processed} episodes.")
    return valid_intervals


class FrameSampler(torch.utils.data.Sampler):
    """
    Custom sampler that only samples data indices falling within specified intervals
    """
    def __init__(
        self,
        dataset,
        sampler_type,
        *,
        reset_truncation_mode="auto",
        keyword_multipliers: dict[str, int] | None = None,
    ):
        if reset_truncation_mode not in RESET_TRUNCATION_MODES:
            raise ValueError(
                f"Invalid reset truncation mode: {reset_truncation_mode}. "
                f"Expected one of {RESET_TRUNCATION_MODES}."
            )
        self.reset_truncation_mode = reset_truncation_mode
        self.keyword_multipliers = keyword_multipliers or {}
        valid_intervals = self.parse_dataset(dataset, sampler_type)
        self.sample_frames(valid_intervals, len(dataset))

    def parse_dataset(self, dataset, sampler_type):
        """
        Args:
            intervals: List of (start_index, end_index) tuples
        """
        if sampler_type == 'subtask':
            return sample_subtask(dataset, keyword_multipliers=self.keyword_multipliers)
        else:
            raise ValueError(f"Invalid sampler type: {sampler_type}")

    def sample_frames(self, intervals, dataset_size):
        """
        Args:
            intervals: List of (start_index, end_index) tuples
            dataset_size: Total size of the dataset
        """
        self.intervals = intervals
        self.dataset_size = dataset_size
        
        # Pre-compute all valid indices
        self.valid_indices = []
        weighted_intervals = 0
        for interval in intervals:
            if len(interval) == 2:
                start_idx, end_idx = interval
                weight = 1
            else:
                start_idx, end_idx, weight = interval
            weight = max(1, int(weight))
            if weight > 1:
                weighted_intervals += 1
            # Ensure indices are within dataset bounds
            start_idx = max(0, start_idx)
            end_idx = min(dataset_size - 1, end_idx)
            
            # Add all indices within the interval
            for _ in range(weight):
                self.valid_indices.extend(range(start_idx, end_idx + 1))
        
        if not weighted_intervals:
            self.valid_indices = sorted(list(set(self.valid_indices)))
        print(
            f"Total {len(self.valid_indices)} valid indices,",
            'original:',
            dataset_size,
            'weighted intervals:',
            weighted_intervals,
        )

        import random
        random.shuffle(self.valid_indices)
    
    def __iter__(self):
        import random
        indices = list(self.valid_indices)
        random.shuffle(indices)
        return iter(indices)
    
    def __len__(self):
        return len(self.valid_indices)
