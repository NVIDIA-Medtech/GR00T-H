import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gr00t.data.interfaces import ShardedDataset
from gr00t.data.step_filtering import compute_valid_step_indices_parallel
from gr00t.data.types import EmbodimentTag, MessageType, ModalityConfig, VLAStepData

from .lerobot_episode_loader import LeRobotEpisodeLoader


# Maximum number of workers for parallel filtering
MAX_FILTER_WORKERS = 128


# Constants for percentile stats file handling
PERCENTILE_STATS_FILENAME = "meta/temporal_stats.json"


def extract_step_data(
    episode_data: pd.DataFrame,
    step_index: int,
    modality_configs: dict[str, ModalityConfig],
    embodiment_tag: EmbodimentTag,
    allow_padding: bool = False,
) -> VLAStepData:
    """Extract a single training sample from episode data at a given step index.

    Handles both dense DataFrames (from full episode loading, with a standard
    RangeIndex 0..N-1) and sparse DataFrames (from get_sparse_episode, where
    the index contains the original frame indices). For sparse DataFrames, an
    index map translates original frame indices to positional iloc offsets.

    Args:
        episode_data: Episode DataFrame, either dense (full episode) or sparse
            (subset of frames with original indices preserved in df.index).
        step_index: The anchor timestep to extract data for.
        modality_configs: Per-modality configuration (delta_indices, keys).
        embodiment_tag: Embodiment identifier for the dataset.
        allow_padding: If True, clamp out-of-range indices to the valid range
            instead of raising an error.

    Returns:
        VLAStepData containing video, state, action, and language data for the
        requested step_index with all configured delta offsets applied.
    """
    step_data = {}

    # Detect sparse DataFrames from get_sparse_episode.
    # Sparse DataFrames have a non-standard index (original frame indices)
    # instead of the default RangeIndex(start=0, step=1).
    is_sparse = not (
        isinstance(episode_data.index, pd.RangeIndex)
        and episode_data.index.start == 0
        and episode_data.index.step == 1
    )

    # Build index map for sparse DataFrames: {original_frame_idx: iloc_position}
    # This lets us convert step_index + delta -> iloc position for data access.
    # For dense DataFrames, index_map is None and positions == indices_to_load.
    index_map: dict[int, int] | None = None
    if is_sparse:
        index_map = {int(idx): pos for pos, idx in enumerate(episode_data.index)}

    # Extract data for each configured modality
    for modality, config in modality_configs.items():
        step_data[modality] = {}
        # Sample timesteps according to delta indices configuration
        indices_to_load = [step_index + delta_index for delta_index in config.delta_indices]

        if allow_padding:
            if is_sparse:
                # For sparse DataFrames, clamp to the range of available indices
                min_idx = int(episode_data.index.min())
                max_idx = int(episode_data.index.max())
                indices_to_load = [max(min_idx, min(idx, max_idx)) for idx in indices_to_load]
            else:
                indices_to_load = [
                    max(0, min(idx, len(episode_data) - 1)) for idx in indices_to_load
                ]

        # Convert original frame indices to iloc positions for sparse DataFrames.
        # When allow_padding=True, clamp to the sparse DF's index range so
        # out-of-range indices map to boundary frames (same as dense-path padding).
        # When allow_padding=False, do a direct lookup — a missing index raises
        # KeyError, consistent with the dense path's IndexError on out-of-bounds iloc.
        if is_sparse and index_map is not None:
            if allow_padding:
                min_idx = int(episode_data.index.min())
                max_idx = int(episode_data.index.max())
                clamped_indices = [max(min_idx, min(idx, max_idx)) for idx in indices_to_load]
                positions = [index_map[idx] for idx in clamped_indices]
            else:
                positions = [index_map[idx] for idx in indices_to_load]
        else:
            positions = indices_to_load

        for key in config.modality_keys:
            if f"{modality}.{key}" in episode_data.columns:
                modality_data = episode_data[f"{modality}.{key}"].iloc[positions]
            else:
                raise KeyError(
                    f"{modality}.{key} not found in episode data, available keys: {episode_data.columns}"
                )
            if modality in ["state", "action"]:
                # Stack arrays for numerical modalities
                step_data[modality][key] = np.vstack(
                    [
                        np.array(modality_data.iloc[i]).astype(np.float32)
                        for i in range(len(modality_data))
                    ]
                )
            else:
                # Keep as lists for other modalities (video, language)
                step_data[modality][key] = modality_data.tolist()

    # Parse extracted data into VLAStepData structure
    video_data = step_data.get("video", {})
    mask_data = step_data.get("mask", {})
    state_data = step_data.get("state", {})
    action_data = step_data.get("action", {})
    language_data = step_data.get("language", {})
    assert len(language_data) == 1, f"Expected 1 language, got {len(language_data)}"
    text = language_data[list(language_data.keys())[0]][0]

    vla_step_data = VLAStepData(
        images=video_data,
        masks=mask_data if mask_data else None,
        states=state_data,
        actions=action_data,
        text=text,
        embodiment=embodiment_tag,
    )
    return vla_step_data


class ShardedSingleStepDataset(ShardedDataset):
    """
    Single-step dataset that creates shards from individual timesteps across episodes.

    This dataset implementation provides step-level data access for VLA training by:
    1. Loading episodes using LeRobotEpisodeLoader
    2. Splitting episodes into individual timesteps
    3. Organizing timesteps into balanced shards for efficient loading
    4. Supporting episode subsampling for data efficiency

    The sharding strategy ensures balanced shard sizes while maintaining randomization
    across episodes and timesteps within episodes. Each shard contains a mix of
    timesteps from different episodes to improve training diversity.

    Key features:
    - Step-level data access (vs episode-level)
    - Balanced sharding for consistent batch sizes
    - Episode subsampling via sampling rate
    - Integration with LeRobot data format
    - Support for multi-modal data (video, state, action, language)

    Args:
        dataset_path: Path to LeRobot format dataset directory
        embodiment_tag: Embodiment identifier for cross-embodiment training
        modality_configs: Configuration for each modality (sampling, keys)
        video_backend: Video decoding backend ('torchcodec', 'decord', etc.)
        video_backend_kwargs: Additional arguments for video backend
        shard_size: Target number of timesteps per shard
        episode_sampling_rate: Fraction of episode timesteps to use (for efficiency)
        seed: Random seed for reproducible sharding and sampling
        allow_padding: Whether to allow padding of indices to valid range [0, max_length - 1]
        episode_indices: Optional subset of episode indices to use. If provided, only these
            episodes will be included in the dataset. This enables train/val splitting at
            the episode level to prevent data leakage. If None, all episodes are used.

    Example:
        >>> dataset = ShardedSingleStepDataset(
        ...     dataset_path="/path/to/lerobot_dataset",
        ...     embodiment_tag=EmbodimentTag.FRANKA,
        ...     modality_configs={
        ...         "video": ModalityConfig(delta_indices=[0], modality_keys=["front_cam"]),
        ...         "state": ModalityConfig(delta_indices=[0], modality_keys=["joint_positions"]),
        ...         "action": ModalityConfig(
        ...             delta_indices=list(range(8)), modality_keys=["joint_velocities"]
        ...         ),
        ...     },
        ...     shard_size=1024,
        ...     episode_sampling_rate=0.1,
        ... )
        >>> shard_data = dataset.get_shard(0)  # Get first shard of processed timesteps
    """

    def __init__(
        self,
        dataset_path: str | Path,
        embodiment_tag: EmbodimentTag,
        modality_configs: dict[str, ModalityConfig],
        video_backend: str = "torchcodec",
        video_backend_kwargs: dict[str, Any] | None = None,
        shard_size: int = 2**10,  # 1024 steps
        episode_sampling_rate: float = 0.1,
        seed: int = 42,
        allow_padding: bool = False,
        episode_indices: np.ndarray | None = None,
    ):
        """Initialize single-step dataset with sharding configuration.

        Args:
            dataset_path: Path to LeRobot format dataset directory
            embodiment_tag: Embodiment identifier for cross-embodiment training
            modality_configs: Configuration for each modality (sampling, keys)
            video_backend: Video decoding backend ('torchcodec', 'decord', etc.)
            video_backend_kwargs: Additional arguments for video backend
            shard_size: Target number of timesteps per shard
            episode_sampling_rate: Fraction of episode timesteps to use (for efficiency)
            seed: Random seed for reproducible sharding and sampling
            allow_padding: Whether to allow padding of indices to valid range
            episode_indices: Optional subset of episode indices to use for train/val split.
                If None, all episodes are used.
        """
        super().__init__(dataset_path)
        self.embodiment_tag = embodiment_tag
        self.modality_configs = modality_configs
        self.video_backend = video_backend
        self.video_backend_kwargs = video_backend_kwargs
        self.shard_size = shard_size
        self.episode_sampling_rate = episode_sampling_rate
        self.seed = seed
        self.allow_padding = allow_padding
        self.episode_indices = episode_indices  # Store for shard_dataset()
        self.processor = None
        self.rng = np.random.default_rng(seed)
        action_delta_indices = modality_configs["action"].delta_indices

        # BUG: The below assumed contiguous delta indices, but fails for [0,2,4,...,32]
        # self.action_horizon = max(action_delta_indices) - min(action_delta_indices) + 1
        # action_horizon must account for the maximum delta index, not just the range
        # This ensures step_index + max(delta_indices) < episode_length
        self.action_horizon = max(action_delta_indices) + 1

        self.episode_loader = LeRobotEpisodeLoader(
            dataset_path=dataset_path,
            modality_configs=modality_configs,
            video_backend=video_backend,
            video_backend_kwargs=video_backend_kwargs,
        )

        # Create balanced shards from episode timesteps
        self.shard_dataset()

    def shard_dataset(self):
        """
        Create balanced shards by distributing episode timesteps across shards.

        The sharding process:
        1. Run parallel clutch-aware filtering (PyArrow + ProcessPoolExecutor)
        2. Shuffle episode order for randomization
        3. Split each episode into multiple sub-sequences based on sampling rate
        4. Distribute sub-sequences across shards to balance shard sizes
        5. Use greedy assignment to minimize shard size variance

        This approach ensures:
        - Balanced shard sizes for consistent training batches
        - Diversity within shards (mix of episodes and timesteps)
        - Reproducible sharding based on seed
        - Fast startup with parallel filtering (~15-30 seconds for 4792 episodes)
        """
        # Run parallel filtering once at startup (auto-detects CMR data)
        # Uses PyArrow for fast column selection + ProcessPoolExecutor for parallelism
        # Worker count auto-detects available CPUs, capped at MAX_FILTER_WORKERS (128)
        filter_results = self._filter_all_episodes_parallel()

        # Use provided episode_indices subset or all episodes
        # This enables train/val splitting at the episode level
        if self.episode_indices is not None:
            all_episode_indices = self.episode_indices
        else:
            all_episode_indices = np.arange(len(self.episode_loader.episode_lengths))

        shuffled_episode_indices = self.rng.permutation(all_episode_indices)
        num_splits = int(1 / self.episode_sampling_rate)

        assert len(shuffled_episode_indices) > 0, (
            f"No valid trajectories found for dataset {self.dataset_path}"
        )

        # Calculate total timesteps accounting for clutch filtering when applied
        # Use filtered step counts for CMR data, raw episode lengths otherwise
        if filter_results is not None:
            # CMR data: count actual valid steps after filtering
            total_steps = 0
            for idx in shuffled_episode_indices:
                if idx in filter_results:
                    total_steps += len(filter_results[idx])
                # Episodes not in filter_results had 0 valid indices, skip them
        else:
            # Non-CMR data: use raw episode lengths
            total_steps = int(
                np.sum([self.get_effective_episode_length(idx) for idx in shuffled_episode_indices])
            )

        # Ensure at least 1 shard, handle edge case of all data filtered out
        assert total_steps > 0, (
            f"No valid timesteps after filtering for dataset {self.dataset_path}. "
            f"All {len(shuffled_episode_indices)} episodes were filtered out by clutch-aware filtering. "
            f"filter_results is None: {filter_results is None}, "
            f"filter_results keys: {list(filter_results.keys()) if filter_results else 'N/A'}"
        )

        # Count episodes that will actually contribute data
        if filter_results is not None:
            num_episodes_with_data = sum(
                1 for idx in shuffled_episode_indices if idx in filter_results
            )
        else:
            num_episodes_with_data = len(shuffled_episode_indices)

        # Calculate shards needed, but cap at episodes * num_splits since each episode
        # contributes to at most num_splits shards in the distribution loop
        num_shards_by_steps = max(1, int(np.ceil(total_steps / self.shard_size)))
        max_shards_by_episodes = num_episodes_with_data * num_splits
        num_shards = min(num_shards_by_steps, max_shards_by_episodes)

        print(
            f"Shard plan: total_steps={total_steps}, num_shards={num_shards}, shard_size={self.shard_size}, "
            f"num_episodes_with_data={num_episodes_with_data}, num_splits={num_splits}"
        )

        # Initialize shard containers
        sharded_episodes = [[] for _ in range(num_shards)]
        shard_lengths = np.zeros(num_shards, dtype=int)

        # Distribute episode sub-sequences across shards
        for ep_idx in shuffled_episode_indices:
            # Get step indices - either from pre-computed filter results or full range
            if filter_results is not None:
                # CMR data: only use episodes that passed filtering
                if ep_idx in filter_results:
                    step_indices = filter_results[ep_idx].copy()
                else:
                    # Episode had 0 valid indices after filtering, skip it
                    continue
            else:
                # Non-CMR data: use full range
                step_indices = np.arange(0, self.get_effective_episode_length(ep_idx))

            if len(step_indices) == 0:
                continue  # Skip episodes with no valid indices

            self.rng.shuffle(step_indices)
            for i in range(num_splits):
                split_step_indices = step_indices[i::num_splits]
                if len(split_step_indices) == 0:
                    continue  # Skip empty splits
                # Assign to shard with minimum current length (greedy balancing)
                shard_index = np.argmin(shard_lengths)
                sharded_episodes[shard_index].append((ep_idx, split_step_indices))
                shard_lengths[shard_index] += len(split_step_indices)

        # Validate shard creation
        empty_shards = [i for i in range(num_shards) if shard_lengths[i] == 0]
        assert len(empty_shards) == 0, (
            f"All shards must have length > 0. Empty shards: {empty_shards}, "
            f"shard_lengths: {shard_lengths.tolist()}, total distributed: {shard_lengths.sum()}, "
            f"expected total_steps: {total_steps}"
        )

        print(f"Generated {num_shards} shards for dataset {self.dataset_path}")
        print(
            f"Total steps: {total_steps}, average shard length: {total_steps / num_shards}, shard length std: {np.std(shard_lengths)}"
        )
        self.sharded_episodes = sharded_episodes
        self.shard_lengths = shard_lengths

    def get_effective_episode_length(self, episode_index: int) -> int:
        """Get the effective episode length accounting for action horizon."""
        original_length = self.episode_loader.get_episode_length(episode_index)
        return max(0, original_length - self.action_horizon + 1)

    def _filter_all_episodes_parallel(
        self, num_workers: int | None = None
    ) -> dict[int, np.ndarray] | None:
        """Parallel filtering of all episodes using PyArrow column selection.

        This function applies one of two dataset-specific filters:
        1) CMR clutch-aware filtering (uses observation.state keys)
        2) Terminal-step filtering for datasets with `next.done` padding

        Uses ProcessPoolExecutor to distribute filtering across multiple CPUs.
        Each episode's parquet file is read with PyArrow (only required columns),
        avoiding video decoding and full DataFrame loads.

        Args:
            num_workers: Number of parallel workers. If None, uses min(cpu_count, MAX_FILTER_WORKERS).

        Returns:
            None if no filtering is required for this dataset.
            Dict mapping episode_idx to valid step indices if filtering is applied.
            Empty dict {} means all episodes were filtered out.
        """
        if self.episode_indices is not None:
            episode_indices = list(self.episode_indices)
        else:
            episode_indices = list(range(len(self.episode_loader)))
        effective_lengths = [self.get_effective_episode_length(i) for i in episode_indices]
        return compute_valid_step_indices_parallel(
            dataset_path=self.dataset_path,
            embodiment_tag=self.embodiment_tag,
            chunk_size=self.episode_loader.chunk_size,
            data_path_pattern=self.episode_loader.data_path_pattern,
            action_delta_indices=self.modality_configs["action"].delta_indices,
            episode_indices=episode_indices,
            effective_lengths=effective_lengths,
            num_workers=num_workers,
            max_filter_workers=MAX_FILTER_WORKERS,
            show_progress=True,
        )

    def __len__(self):
        """Return the number of shards in the dataset."""
        return len(self.shard_lengths)

    def get_datapoint(self, episode_data: pd.DataFrame, step_index: int) -> dict:
        """
        Extract and process a single timestep from episode data.

        Converts raw episode data into a VLAStepData structure and applies
        the configured processor to create model-ready inputs.

        Args:
            episode_data: Complete episode DataFrame from LeRobotEpisodeLoader
            step_index: Timestep index within the episode to extract

        Returns:
            Processed datapoint ready for model training

        Raises:
            AssertionError: If processor is not set before calling this method
            KeyError: If the expected stats key is not present in processor normalization params
        """
        assert self.processor is not None, "Processor must be set before getting datapoints"
        vla_step_data = extract_step_data(
            episode_data,
            step_index,
            self.modality_configs,
            self.embodiment_tag,
            self.allow_padding,
        )
        # Apply processor to convert to model inputs
        messages = [{"type": MessageType.EPISODE_STEP.value, "content": vla_step_data}]
        return self.processor(messages)

    def get_shard_length(self, idx: int) -> int:
        """Get the number of timesteps in a specific shard."""
        return self.shard_lengths[idx]

    def get_shard(self, idx: int) -> list:
        """Load and process all timesteps in a specific shard using sparse loading.

        Uses sparse episode loading to decode only the video frames and parquet
        rows actually needed for the timesteps assigned to this shard, rather than
        loading entire episodes. This reduces video decode work by 5-11x for
        typical configurations where video delta_indices=[0] but action horizons
        span 16-50 steps.

        For each episode referenced in the shard:
        1. Compute the union of all frame indices needed across all modalities
        2. Compute the (smaller) union of frame indices needed for video only
        3. Call get_sparse_episode to load just those frames
        4. Extract individual timestep datapoints from the sparse DataFrame

        Args:
            idx: Shard index to load

        Returns:
            List of processed timesteps ready for model training
        """
        episodes = self.sharded_episodes[idx]
        datapoints = []
        for ep_idx, step_indices in episodes:
            # Compute the minimal set of frame indices needed for this episode's
            # step_indices across all modality delta offsets
            required_indices = self._compute_required_indices(ep_idx, step_indices)
            required_video_indices = self._compute_required_video_indices(ep_idx, step_indices)

            # Load only the needed frames (sparse parquet subset + targeted video decode)
            episode_data = self.episode_loader.get_sparse_episode(
                ep_idx,
                frame_indices=required_indices,
                video_indices=required_video_indices,
                allow_padding=self.allow_padding,
            )
            for step_index in step_indices:
                datapoints.append(self.get_datapoint(episode_data, step_index))
        return datapoints

    def _compute_required_indices(
        self,
        ep_idx: int,
        step_indices: np.ndarray,
    ) -> np.ndarray:
        """Compute the union of all frame indices needed for a set of step indices.

        For sparse parquet loading, we need to know exactly which timesteps are
        required across all step_indices and all modality delta_indices. This method
        computes that union, handling padding by clamping to valid episode bounds.

        Video frames are included here too (the sparse DataFrame must contain rows
        for both state/action AND video indices), but the actual video decode uses
        the smaller set from _compute_required_video_indices.

        Args:
            ep_idx: Episode index (for getting episode length bounds).
            step_indices: Array of step indices assigned to this shard for this episode.

        Returns:
            Sorted array of unique frame indices needed across all modalities.
        """
        episode_length = self.episode_loader.get_episode_length(ep_idx)
        required_indices = set()

        # Collect delta_indices from ALL modalities (state, action, video, language)
        all_deltas = set()
        for config in self.modality_configs.values():
            all_deltas.update(config.delta_indices)

        # Compute the union of all needed frame indices
        for step_idx in step_indices:
            for delta in all_deltas:
                idx = step_idx + delta
                if self.allow_padding:
                    # Clamp to valid range when padding is allowed
                    idx = max(0, min(idx, episode_length - 1))
                if 0 <= idx < episode_length:
                    required_indices.add(idx)

        return np.array(sorted(required_indices), dtype=np.int64)

    def _compute_required_video_indices(
        self,
        ep_idx: int,
        step_indices: np.ndarray,
    ) -> np.ndarray:
        """Compute the union of video frame indices needed for a set of step indices.

        This deliberately uses ONLY the video modality's delta_indices, which are
        typically just [0] (the current frame). This is the core of the sparse
        loading optimization: by separating video indices from state/action indices,
        we avoid decoding frames that are only needed for action labels.

        For example, with video delta_indices=[0] and action delta_indices=range(16),
        a step_index of 100 needs video frame 100 but action frames 100-115. Without
        this separation, we'd decode 16 video frames instead of 1.

        Args:
            ep_idx: Episode index (for getting episode length bounds).
            step_indices: Array of step indices assigned to this shard for this episode.

        Returns:
            Sorted array of unique video frame indices to decode.
        """
        if "video" not in self.modality_configs:
            return np.array([], dtype=np.int64)

        episode_length = self.episode_loader.get_episode_length(ep_idx)
        required_indices = set()
        video_deltas = self.modality_configs["video"].delta_indices

        for step_idx in step_indices:
            for delta in video_deltas:
                idx = step_idx + delta
                if self.allow_padding:
                    idx = max(0, min(idx, episode_length - 1))
                if 0 <= idx < episode_length:
                    required_indices.add(idx)

        return np.array(sorted(required_indices), dtype=np.int64)

    def get_dataset_statistics(self) -> dict:
        """Get dataset statistics from the underlying episode loader."""
        return self.episode_loader.get_dataset_statistics()

    def get_percentile_statistics(
        self,
        consolidated_stats_path: str | Path | None = None,
    ) -> dict | None:
        """
        Get percentile statistics for this dataset from consolidated or per-dataset file.

        Percentile statistics contain q01, q02, q98, q99, mean, and std for both
        state and action modalities. Action statistics are temporal-aware with
        shape (horizon, dim), while state statistics have shape (dim,).

        Args:
            consolidated_stats_path: Optional path to a consolidated stats JSON file
                                    keyed by repo_id. If not provided, looks for
                                    per-dataset stats in meta/temporal_stats.json.

        Returns:
            Dictionary with structure:
            {
                "state": {key: {stat_type: values}},
                "action": {key: {stat_type: values}}
            }
            Returns None if no stats file is found.

        Raises:
            FileNotFoundError: If require_stats=True and no stats file is found
            KeyError: If using consolidated stats but this dataset's repo_id is missing
        """
        # Try consolidated stats file first (if provided)
        if consolidated_stats_path is not None:
            consolidated_stats_path = Path(consolidated_stats_path)
            if consolidated_stats_path.exists():
                with open(consolidated_stats_path, "r") as f:
                    all_stats = json.load(f)
                repo_id = Path(self.dataset_path).name
                if repo_id not in all_stats:
                    raise KeyError(
                        f"Dataset '{repo_id}' not found in consolidated stats file "
                        f"{consolidated_stats_path}. Available datasets: {list(all_stats.keys())}"
                    )
                return all_stats[repo_id]

        # Fall back to per-dataset stats file
        per_dataset_stats_path = Path(self.dataset_path) / PERCENTILE_STATS_FILENAME
        if per_dataset_stats_path.exists():
            with open(per_dataset_stats_path, "r") as f:
                return json.load(f)

        return None

    def get_initial_actions(self):
        """Get initial actions from the underlying episode loader."""
        return self.episode_loader.get_initial_actions()
