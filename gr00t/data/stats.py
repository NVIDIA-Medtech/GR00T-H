#!/usr/bin/env python
"""
Calculate dataset statistics for LeRobot datasets.

This module provides functions for computing normalization statistics including:
- Standard statistics (mean, std, min, max, q01, q99) for backward compatibility
- Temporal-aware percentile statistics for action normalization with shape (horizon, dim)
- Non-temporal percentile statistics for state normalization with shape (dim,)

The temporal statistics are designed for percentile-based normalization using 2nd/98th
percentiles, which is more robust to outliers than min-max scaling.

Note: Please update the `gr00t/configs/data/embodiment_configs.py` file with the correct
modality configurations for the dataset you are using before running this script.

Usage:
    python gr00t/data/stats.py <dataset_path> <embodiment_tag>

Args:
    dataset_path: Path to the dataset.
    embodiment_tag: Embodiment tag to use to load modality configurations from
                   `gr00t/configs/data/embodiment_configs.py`.
"""

from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
from typing import Any

import numpy as np
import open_h.embodiments  # noqa: F401 - registers Open-H embodiment configs
import pandas as pd
from tqdm import tqdm

from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.data.split_utils import load_info_json, resolve_episode_indices
from gr00t.data.state_action.action_chunking import EndEffectorActionChunk, JointActionChunk
from gr00t.data.state_action.pose import (
    EndEffectorPose,
    JointPose,
    apply_motion_scaling_to_rel_xyz_rot6d,
    convert_to_rel_xyz_rot6d,
)
from gr00t.data.step_filtering import compute_valid_step_indices_parallel
from gr00t.data.types import (
    ActionFormat,
    ActionRepresentation,
    ActionType,
    EmbodimentTag,
    ModalityConfig,
)
from gr00t.data.utils import to_json_serializable


LE_ROBOT_DATA_FILENAME = "data/*/*.parquet"
LE_ROBOT_INFO_FILENAME = "meta/info.json"
LE_ROBOT_STATS_FILENAME = "meta/stats.json"
LE_ROBOT_REL_STATS_FILENAME = "meta/relative_stats.json"
LE_ROBOT_TEMPORAL_STATS_FILENAME = "meta/temporal_stats.json"


def _resolve_episode_ids(
    total_episodes: int,
    episode_indices: np.ndarray | list[int] | None = None,
    max_episodes: int = -1,
    context: str = "percentile stats",
) -> list[int]:
    """Resolve episode IDs from optional indices and max episode cap."""
    if episode_indices is None:
        episode_ids = list(range(total_episodes))
    else:
        indices = np.unique(np.asarray(episode_indices, dtype=int))
        invalid = [int(i) for i in indices if i < 0 or i >= total_episodes]
        if invalid:
            raise ValueError(
                f"Invalid episode indices for {context}: {invalid}. "
                f"Valid range: 0-{total_episodes - 1}"
            )
        episode_ids = indices.tolist()

    if max_episodes != -1:
        episode_ids = episode_ids[:max_episodes]

    return episode_ids


def calculate_dataset_statistics(
    parquet_paths: list[Path], features: list[str] | None = None
) -> dict[str, dict[str, float]]:
    """Calculate the dataset statistics of all columns for a list of parquet files.

    Args:
        parquet_paths (list[Path]): List of paths to parquet files to process.
        features (list[str] | None): List of feature names to compute statistics for.
            If None, computes statistics for all columns in the data.

    Returns:
        dict[str, DatasetStatisticalValues]: Dictionary mapping feature names to their
            statistical values (mean, std, min, max, q01, q99).
    """
    # Dataset statistics
    all_low_dim_data_list = []
    # Collect all the data
    for parquet_path in tqdm(
        sorted(list(parquet_paths)),
        desc="Collecting all parquet files...",
    ):
        # Load the parquet file
        parquet_data = pd.read_parquet(parquet_path)
        parquet_data = parquet_data
        all_low_dim_data_list.append(parquet_data)
    all_low_dim_data = pd.concat(all_low_dim_data_list, axis=0)
    # Compute dataset statistics
    dataset_statistics = {}
    if features is None:
        features = list(all_low_dim_data.columns)
    else:
        # Some datasets list float features in info.json that never appear in parquet.
        # Skip those to avoid KeyError during stats calculation.
        missing_features = [feature for feature in features if feature not in all_low_dim_data]
        if missing_features:
            print(
                "WARNING: skipping missing features during stats generation: "
                f"{sorted(missing_features)}"
            )
            features = [feature for feature in features if feature not in missing_features]
    for le_modality in features:
        print(f"Computing statistics for {le_modality}...")
        np_data = np.vstack(
            [np.asarray(x, dtype=np.float32) for x in all_low_dim_data[le_modality]]
        )
        dataset_statistics[le_modality] = dict(
            mean=np.mean(np_data, axis=0).tolist(),
            std=np.std(np_data, axis=0).tolist(),
            min=np.min(np_data, axis=0).tolist(),
            max=np.max(np_data, axis=0).tolist(),
            q01=np.quantile(np_data, 0.01, axis=0).tolist(),
            q99=np.quantile(np_data, 0.99, axis=0).tolist(),
        )
    return dataset_statistics


def check_stats_validity(dataset_path: Path | str, features: list[str]):
    stats_path = Path(dataset_path) / LE_ROBOT_STATS_FILENAME
    if not stats_path.exists():
        return False
    with open(stats_path, "r") as f:
        stats = json.load(f)
    for feature in features:
        if feature not in stats:
            return False
        if not isinstance(stats[feature], dict):
            return False
        for stat in ["mean", "std", "min", "max", "q01", "q99"]:
            if stat not in stats[feature]:
                return False
    return True


def generate_stats(dataset_path: Path | str, episode_indices: np.ndarray | list[int] | None = None):
    """Generate stats.json for low-dimensional float features.

    Args:
        dataset_path: Path to the LeRobot dataset root.
        episode_indices: Optional list/array of episode indices to include. If None,
            compute stats over the full dataset.
    """
    dataset_path = Path(dataset_path)
    print(f"Generating stats for {str(dataset_path)}")
    lowdim_features = []
    with open(dataset_path / LE_ROBOT_INFO_FILENAME, "r") as f:
        le_features = json.load(f)["features"]
    for feature in le_features:
        if "float" in le_features[feature]["dtype"]:
            lowdim_features.append(feature)

    if episode_indices is None:
        parquet_files = list(dataset_path.glob(LE_ROBOT_DATA_FILENAME))
    else:
        # Use loader to validate indices and resolve chunked parquet paths
        loader = LeRobotEpisodeLoader(
            dataset_path,
            modality_configs={},
            skip_video=True,
            require_stats=False,
        )
        total_episodes = len(loader)
        indices = np.unique(np.asarray(episode_indices, dtype=int))
        invalid = [int(i) for i in indices if i < 0 or i >= total_episodes]
        if invalid:
            raise ValueError(
                f"Invalid episode indices for stats: {invalid}. Valid range: 0-{total_episodes - 1}"
            )

        parquet_files = [
            dataset_path
            / loader.data_path_pattern.format(
                episode_chunk=int(ep_idx) // loader.chunk_size,
                episode_index=int(ep_idx),
            )
            for ep_idx in indices
        ]

    stats = calculate_dataset_statistics(parquet_files, lowdim_features)
    stats_path = dataset_path / LE_ROBOT_STATS_FILENAME
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=4)


class RelativeActionLoader:
    def __init__(
        self,
        dataset_path: Path | str,
        embodiment_tag: EmbodimentTag,
        action_key: str,
        episode_indices: np.ndarray | list[int] | None = None,
    ):
        """Load episodes for RELATIVE action stats computation.

        Args:
            dataset_path: Path to dataset root directory.
            embodiment_tag: Embodiment tag for modality config lookup.
            action_key: Action key to compute relative stats for.
            episode_indices: Optional subset of episodes to include.
        """
        self.dataset_path = Path(dataset_path)
        self.modality_configs: dict[str, ModalityConfig] = {}
        self.action_key = action_key
        # Check action config
        assert action_key in MODALITY_CONFIGS[embodiment_tag.value]["action"].modality_keys
        idx = MODALITY_CONFIGS[embodiment_tag.value]["action"].modality_keys.index(action_key)
        action_configs = MODALITY_CONFIGS[embodiment_tag.value]["action"].action_configs
        assert action_configs is not None, MODALITY_CONFIGS[embodiment_tag.value]["action"]
        self.action_config = action_configs[idx]
        self.modality_configs["action"] = ModalityConfig(
            delta_indices=MODALITY_CONFIGS[embodiment_tag.value]["action"].delta_indices,
            modality_keys=[action_key],
        )
        # Check state config
        state_key = self.action_config.state_key or action_key
        assert state_key in MODALITY_CONFIGS[embodiment_tag.value]["state"].modality_keys
        self.modality_configs["state"] = ModalityConfig(
            delta_indices=MODALITY_CONFIGS[embodiment_tag.value]["state"].delta_indices,
            modality_keys=[state_key],
        )
        # Check state-action consistency:
        # Allow action horizons that start AFTER the reference state (e.g., [1..H])
        # so we can compute relative deltas to future targets.
        state_delta = self.modality_configs["state"].delta_indices[-1]
        action_start = self.modality_configs["action"].delta_indices[0]
        assert state_delta <= action_start, (
            "State reference index must be <= first action delta index. "
            f"Got state_delta={state_delta}, action_start={action_start}."
        )
        self.loader = LeRobotEpisodeLoader(dataset_path, self.modality_configs)
        self.episode_indices = None
        if episode_indices is not None:
            indices = np.unique(np.asarray(episode_indices, dtype=int))
            invalid = [int(i) for i in indices if i < 0 or i >= len(self.loader)]
            if invalid:
                raise ValueError(
                    f"Invalid episode indices for relative stats: {invalid}. "
                    f"Valid range: 0-{len(self.loader) - 1}"
                )
            self.episode_indices = indices

    def _resolve_episode_index(self, local_index: int) -> int:
        """Map a local episode index to a dataset episode index."""
        if self.episode_indices is None:
            return local_index
        return int(self.episode_indices[local_index])

    def load_relative_actions(self, trajectory_id: int) -> list[np.ndarray]:
        episode_index = self._resolve_episode_index(trajectory_id)
        df = self.loader[episode_index]

        # OPTIMIZATION: Extract columns once and convert to numpy arrays
        # This eliminates repeated DataFrame.__getitem__ and Series.__getitem__ calls
        if self.action_config.state_key is not None:
            state_key = f"state.{self.action_config.state_key}"
        else:
            state_key = f"state.{self.action_key}"
        action_key = f"action.{self.action_key}"

        # Convert to numpy arrays once - this is much faster than repeated pandas access
        state_data = df[state_key].values  # Shape: (episode_length, joint_dim)
        action_data = df[action_key].values  # Shape: (episode_length, joint_dim)
        trajectories = []
        usable_length = len(df) - self.modality_configs["action"].delta_indices[-1]
        action_delta_indices = np.array(self.modality_configs["action"].delta_indices)
        for i in range(usable_length):
            state_ind = self.modality_configs["state"].delta_indices[-1] + i
            action_inds = action_delta_indices + i
            last_state = state_data[state_ind]
            actions = action_data[action_inds]
            if self.action_config.type == ActionType.EEF:
                action_format = self.action_config.format
                reference_frame = EndEffectorPose.from_action_format(last_state, action_format)
                traj = EndEffectorActionChunk.from_array(actions, action_format).relative_chunking(
                    reference_frame=reference_frame
                )
                trajectories.append(traj.to(action_format).astype(np.float32))
            elif self.action_config.type == ActionType.NON_EEF:
                reference_frame = JointPose(last_state)
                traj = JointActionChunk([JointPose(m) for m in actions]).relative_chunking(
                    reference_frame=reference_frame
                )
                trajectories.append(np.stack([p.joints for p in traj.poses], dtype=np.float32))
            else:
                raise ValueError(f"Unknown ActionType: {self.action_config.type}")
        return trajectories

    def __len__(self) -> int:
        if self.episode_indices is None:
            return len(self.loader)
        return len(self.episode_indices)


def calculate_stats_for_key(
    dataset_path: Path | str,
    embodiment_tag: EmbodimentTag,
    group_key: str,
    max_episodes: int = -1,
    episode_indices: np.ndarray | list[int] | None = None,
) -> dict:
    """Compute stats for a single RELATIVE action key.

    Args:
        dataset_path: Path to dataset root directory.
        embodiment_tag: Embodiment tag for modality config lookup.
        group_key: Action key to compute stats for.
        max_episodes: Optional cap on episodes to process (-1 means all).
        episode_indices: Optional subset of episode indices to include.
    """
    loader = RelativeActionLoader(
        dataset_path, embodiment_tag, group_key, episode_indices=episode_indices
    )
    trajectories = []
    for episode_id in tqdm(range(len(loader)), desc=f"Loading trajectories for key {group_key}"):
        if max_episodes != -1 and episode_id >= max_episodes:
            break
        trajectories.extend(loader.load_relative_actions(episode_id))
    return {
        "max": np.max(trajectories, axis=0),
        "min": np.min(trajectories, axis=0),
        "q01": np.quantile(trajectories, 0.01, axis=0),
        "q99": np.quantile(trajectories, 0.99, axis=0),
        "mean": np.mean(trajectories, axis=0),
        "std": np.std(trajectories, axis=0),
    }


def generate_rel_stats(
    dataset_path: Path | str,
    embodiment_tag: EmbodimentTag,
    episode_indices: np.ndarray | list[int] | None = None,
) -> None:
    """Generate relative_stats.json for RELATIVE action representations.

    Args:
        dataset_path: Path to the LeRobot dataset root.
        embodiment_tag: Embodiment tag for action config lookup.
        episode_indices: Optional list/array of episode indices to include. If None,
            compute stats over the full dataset.
    """
    dataset_path = Path(dataset_path)
    action_config = MODALITY_CONFIGS[embodiment_tag.value]["action"]
    if action_config.action_configs is None:
        return
    action_keys = [
        key
        for key, action_config in zip(action_config.modality_keys, action_config.action_configs)
        if action_config.rep == ActionRepresentation.RELATIVE
    ]
    stats_path = Path(dataset_path) / LE_ROBOT_REL_STATS_FILENAME
    if stats_path.exists() and episode_indices is None:
        with open(stats_path, "r") as f:
            stats = json.load(f)
    else:
        stats = {}
    for action_key in sorted(action_keys):
        if action_key in stats and episode_indices is None:
            continue
        print(f"Generating relative stats for {dataset_path} {embodiment_tag} {action_key}")
        stats[action_key] = calculate_stats_for_key(
            dataset_path, embodiment_tag, action_key, episode_indices=episode_indices
        )
    with open(stats_path, "w") as f:
        json.dump(to_json_serializable(dict(stats)), f, indent=4)


def calculate_temporal_percentile_stats(
    dataset_path: Path | str,
    modality_configs: dict[str, ModalityConfig],
    skip_video: bool = True,
    max_episodes: int = -1,
    episode_indices: np.ndarray | list[int] | None = None,
    embodiment_tag: EmbodimentTag | None = None,
) -> dict[str, dict[str, Any]]:
    """
    Calculate temporal-aware percentile statistics for actions and non-temporal stats for states.

    This function computes statistics needed for percentile-based normalization:
    - For actions: temporal-aware stats with shape (horizon, dim) where each timestep
      in the action horizon has its own statistics
    - For states: non-temporal stats with shape (dim,) using single timestamp

    Statistics computed:
    - q01, q02: 1st and 2nd percentiles (lower bounds for normalization)
    - q98, q99: 98th and 99th percentiles (upper bounds for normalization)
    - mean, std: Mean and standard deviation (for meanstd normalization)
    - min, max: Absolute min/max values (for minmax normalization)

    Args:
        dataset_path: Path to the LeRobot dataset directory
        modality_configs: Dictionary mapping modality names ('state', 'action') to
                         ModalityConfig objects specifying keys and delta indices
        skip_video: If True, skip loading video data for faster iteration. Default True.
        max_episodes: Maximum number of episodes to process. -1 for all episodes.
        episode_indices: Optional list/array of episode indices to include. If None,
            uses all episodes in the dataset.
        embodiment_tag: Optional embodiment tag used to gate dataset-specific behaviors
            (e.g., skipping terminal `next.done` rows for UCSD).

    Returns:
        Dictionary with structure:
        {
            "state": {
                "<state_key>": {
                    "q01": [dim], "q02": [dim], "q98": [dim], "q99": [dim],
                    "mean": [dim], "std": [dim], "min": [dim], "max": [dim]
                }
            },
            "action": {
                "<action_key>": {
                    "q01": [horizon, dim], "q02": [horizon, dim],
                    "q98": [horizon, dim], "q99": [horizon, dim],
                    "mean": [horizon, dim], "std": [horizon, dim],
                    "min": [horizon, dim], "max": [horizon, dim]
                }
            }
        }

    Example:
        >>> modality_configs = {
        ...     "state": ModalityConfig(delta_indices=[0], modality_keys=["arm_joints"]),
        ...     "action": ModalityConfig(
        ...         delta_indices=list(range(16)), modality_keys=["arm_joints"]
        ...     ),
        ... }
        >>> stats = calculate_temporal_percentile_stats("/path/to/dataset", modality_configs)
        >>> print(
        ...     stats["action"]["arm_joints"]["q02"].shape
        ... )  # (16, 6) for 16-step horizon, 6 joints
    """
    dataset_path = Path(dataset_path)
    loader = LeRobotEpisodeLoader(
        dataset_path,
        modality_configs,
        skip_video=skip_video,
        require_stats=False,  # Stats don't exist yet - we're calculating them
    )

    # Determine action horizon from delta indices
    action_delta_indices = np.array(modality_configs["action"].delta_indices)
    max_delta = int(np.max(action_delta_indices))
    action_horizon = len(action_delta_indices)

    # Collect data for each modality key, organized by timestep for actions
    # state_data[key] = list of (dim,) arrays
    # action_data[key][timestep] = list of (dim,) arrays
    state_data: dict[str, list[np.ndarray]] = {
        key: [] for key in modality_configs.get("state", ModalityConfig([], [])).modality_keys
    }
    action_data: dict[str, dict[int, list[np.ndarray]]] = {
        key: {t: [] for t in range(action_horizon)}
        for key in modality_configs["action"].modality_keys
    }

    # Collect relative action data for RELATIVE representation (joint-angle deltas)
    # This is separate from action_data because we need both absolute and relative stats
    # relative_action_data[key][timestep] = list of (dim,) arrays
    action_configs = modality_configs["action"].action_configs
    relative_action_keys = []
    if action_configs:
        for key_idx, key in enumerate(modality_configs["action"].modality_keys):
            if key_idx < len(action_configs):
                ac = action_configs[key_idx]
                if ac.rep == ActionRepresentation.RELATIVE:
                    relative_action_keys.append(key)

    relative_action_data: dict[str, dict[int, list[np.ndarray]]] = {
        key: {t: [] for t in range(action_horizon)} for key in relative_action_keys
    }

    episode_ids = _resolve_episode_ids(
        total_episodes=len(loader),
        episode_indices=episode_indices,
        max_episodes=max_episodes,
    )

    # Compute per-episode effective lengths and train-time-valid step indices
    effective_length_by_episode = {
        ep_id: max(0, loader.get_episode_length(ep_id) - max_delta) for ep_id in episode_ids
    }
    valid_step_indices_by_episode = compute_valid_step_indices_parallel(
        dataset_path=dataset_path,
        embodiment_tag=embodiment_tag,
        chunk_size=loader.chunk_size,
        data_path_pattern=loader.data_path_pattern,
        action_delta_indices=modality_configs["action"].delta_indices,
        episode_indices=episode_ids,
        effective_lengths=[effective_length_by_episode[ep_id] for ep_id in episode_ids],
        show_progress=False,
    )

    for episode_id in tqdm(episode_ids, desc="Collecting data for percentile stats"):
        effective_length = effective_length_by_episode[episode_id]
        if effective_length <= 0:
            continue

        if valid_step_indices_by_episode is None:
            step_indices = np.arange(effective_length, dtype=np.int32)
        else:
            step_indices = valid_step_indices_by_episode.get(episode_id)
            if step_indices is None or len(step_indices) == 0:
                continue

        df = loader[episode_id]

        # Collect state data (non-temporal, single timestamp per sample)
        if "state" in modality_configs:
            state_delta_idx = modality_configs["state"].delta_indices[-1]
            for state_key in modality_configs["state"].modality_keys:
                col_name = f"state.{state_key}"
                if col_name not in df.columns:
                    continue
                state_col = df[col_name].values
                for step_idx in step_indices:
                    state_idx = state_delta_idx + int(step_idx)
                    state_data[state_key].append(np.asarray(state_col[state_idx], dtype=np.float32))

        # Collect action data (temporal, separate by timestep in horizon)
        # Apply action transformations (rel-xyz-rot6d) if configured
        action_configs = modality_configs["action"].action_configs
        state_delta_idx = (
            modality_configs["state"].delta_indices[-1] if "state" in modality_configs else 0
        )

        for key_idx, action_key in enumerate(modality_configs["action"].modality_keys):
            col_name = f"action.{action_key}"
            if col_name not in df.columns:
                continue
            action_col = df[col_name].values

            # Check if this action needs rel-xyz-rot6d conversion
            action_config = (
                action_configs[key_idx]
                if action_configs and key_idx < len(action_configs)
                else None
            )
            needs_rel_xyz_rot6d = (
                action_config is not None
                and action_config.rep == ActionRepresentation.REL_XYZ_ROT6D
            )

            # Check if this action needs RELATIVE stats (joint-angle deltas)
            needs_relative = (
                action_config is not None and action_config.rep == ActionRepresentation.RELATIVE
            )

            # Get state column for reference pose if needed (for REL_XYZ_ROT6D or RELATIVE)
            if needs_rel_xyz_rot6d or needs_relative:
                state_key = action_config.state_key or action_key
                state_col_name = f"state.{state_key}"
                if state_col_name not in df.columns:
                    raise ValueError(
                        f"State key '{state_key}' not found in data for REL_XYZ_ROT6D conversion. "
                        f"Available columns: {df.columns.tolist()}"
                    )
                state_col = df[state_col_name].values

            # Check if motion scaling is configured (CMR Versius specific)
            has_motion_scaling = action_config is not None and (
                action_config.translation_scaling_key or action_config.rotation_scaling_key
            )
            trans_scale_col = None
            rot_scale_col = None

            if has_motion_scaling:
                if action_config.translation_scaling_key:
                    trans_col_name = f"state.{action_config.translation_scaling_key}"
                    if trans_col_name in df.columns:
                        trans_scale_col = df[trans_col_name].values
                if action_config.rotation_scaling_key:
                    rot_col_name = f"state.{action_config.rotation_scaling_key}"
                    if rot_col_name in df.columns:
                        rot_scale_col = df[rot_col_name].values

            for step_idx in step_indices:
                step_idx = int(step_idx)
                # Get reference state for rel-xyz-rot6d conversion
                if needs_rel_xyz_rot6d:
                    ref_state_idx = state_delta_idx + step_idx
                    eef_pose = np.asarray(state_col[ref_state_idx], dtype=np.float32)

                for t, delta in enumerate(action_delta_indices):
                    action_idx = step_idx + int(delta)
                    action_value = np.asarray(action_col[action_idx], dtype=np.float32)

                    # Apply rel-xyz-rot6d conversion if configured
                    if needs_rel_xyz_rot6d:
                        action_value = convert_to_rel_xyz_rot6d(
                            action_data=action_value[np.newaxis, :],  # Add timestep dim
                            eef_pose=eef_pose,
                            input_rotation_format=action_config.input_rotation_format,
                            reference_rotation_format=action_config.reference_rotation_format,
                            input_quat_order=action_config.input_quat_order,
                            reference_quat_order=action_config.reference_quat_order,
                        )[0]  # Remove timestep dim

                        # Apply motion scaling if configured (CMR Versius specific)
                        # This converts from hand-controller-space to instrument-space
                        if has_motion_scaling:
                            trans_scale = 1.0
                            rot_scale = 1.0
                            if trans_scale_col is not None:
                                trans_scale = float(trans_scale_col[ref_state_idx])
                            if rot_scale_col is not None:
                                rot_scale = float(rot_scale_col[ref_state_idx])
                            action_value = apply_motion_scaling_to_rel_xyz_rot6d(
                                action_value[np.newaxis, :], trans_scale, rot_scale
                            )[0]

                        # For EEF XYZ_ROT6D, only keep xyz (first 3 dims) for stats
                        # rot6d is already bounded [-1, 1] and doesn't need normalization
                        if (
                            action_config.type == ActionType.EEF
                            and action_config.format == ActionFormat.XYZ_ROT6D
                        ):
                            action_value = action_value[:3]  # Only xyz

                    action_data[action_key][t].append(action_value)

                    # Compute relative action (delta from reference state) for RELATIVE representation
                    # This is separate from REL_XYZ_ROT6D which handles EEF poses with rotation math
                    if needs_relative:
                        ref_state_idx = state_delta_idx + step_idx
                        ref_state = np.asarray(state_col[ref_state_idx], dtype=np.float32)
                        # Simple subtraction for joint angles: relative = action - reference_state
                        relative_value = action_value - ref_state
                        relative_action_data[action_key][t].append(relative_value)

    # Calculate statistics
    result: dict[str, dict[str, Any]] = {"state": {}, "action": {}}

    # State statistics: non-temporal, shape (dim,)
    for state_key, data_list in state_data.items():
        if not data_list:
            continue
        stacked = np.stack(data_list, axis=0)  # (num_samples, dim)
        result["state"][state_key] = {
            "q01": np.percentile(stacked, 1, axis=0),
            "q02": np.percentile(stacked, 2, axis=0),
            "q98": np.percentile(stacked, 98, axis=0),
            "q99": np.percentile(stacked, 99, axis=0),
            "mean": np.mean(stacked, axis=0),
            "std": np.std(stacked, axis=0),
            "min": np.min(stacked, axis=0),
            "max": np.max(stacked, axis=0),
        }

    # Action statistics: temporal-aware, shape (horizon, dim)
    for action_key, timestep_data in action_data.items():
        if not timestep_data[0]:  # Check if any data was collected
            continue

        # Build stats for each timestep
        horizon_stats = {
            "q01": [],
            "q02": [],
            "q98": [],
            "q99": [],
            "mean": [],
            "std": [],
            "min": [],
            "max": [],
        }

        for t in range(action_horizon):
            stacked = np.stack(timestep_data[t], axis=0)  # (num_samples, dim)
            horizon_stats["q01"].append(np.percentile(stacked, 1, axis=0))
            horizon_stats["q02"].append(np.percentile(stacked, 2, axis=0))
            horizon_stats["q98"].append(np.percentile(stacked, 98, axis=0))
            horizon_stats["q99"].append(np.percentile(stacked, 99, axis=0))
            horizon_stats["mean"].append(np.mean(stacked, axis=0))
            horizon_stats["std"].append(np.std(stacked, axis=0))
            horizon_stats["min"].append(np.min(stacked, axis=0))
            horizon_stats["max"].append(np.max(stacked, axis=0))

        # Stack to get shape (horizon, dim)
        result["action"][action_key] = {
            stat_name: np.stack(stat_values, axis=0)
            for stat_name, stat_values in horizon_stats.items()
        }

    # Relative action statistics: temporal-aware, shape (horizon, dim)
    # For RELATIVE representation (joint-angle deltas), stored separately from absolute action stats
    if relative_action_data:
        result["relative_action"] = {}
        for action_key, timestep_data in relative_action_data.items():
            if not timestep_data[0]:  # Check if any data was collected
                continue

            # Build stats for each timestep (same structure as action stats)
            horizon_stats = {
                "q01": [],
                "q02": [],
                "q98": [],
                "q99": [],
                "mean": [],
                "std": [],
                "min": [],
                "max": [],
            }

            for t in range(action_horizon):
                stacked = np.stack(timestep_data[t], axis=0)  # (num_samples, dim)
                horizon_stats["q01"].append(np.percentile(stacked, 1, axis=0))
                horizon_stats["q02"].append(np.percentile(stacked, 2, axis=0))
                horizon_stats["q98"].append(np.percentile(stacked, 98, axis=0))
                horizon_stats["q99"].append(np.percentile(stacked, 99, axis=0))
                horizon_stats["mean"].append(np.mean(stacked, axis=0))
                horizon_stats["std"].append(np.std(stacked, axis=0))
                horizon_stats["min"].append(np.min(stacked, axis=0))
                horizon_stats["max"].append(np.max(stacked, axis=0))

            # Stack to get shape (horizon, dim)
            result["relative_action"][action_key] = {
                stat_name: np.stack(stat_values, axis=0)
                for stat_name, stat_values in horizon_stats.items()
            }

    return result


def _collect_episodes_worker(args: tuple) -> tuple[dict, dict, dict]:
    """
    Worker function for parallel episode processing in stats calculation.

    This function is designed to run in a separate process via ProcessPoolExecutor.
    Each worker processes a chunk of episodes independently and returns collected
    data for later aggregation.

    Architecture:
        - Each worker creates its own LeRobotEpisodeLoader (required because loaders
          can't be pickled across process boundaries)
        - Workers process episodes in parallel, with no communication between them
        - Results are merged in the main process after all workers complete

    Memory Optimization (Episode-at-a-Time Stacking):
        Instead of appending individual frames one-by-one:
            for i in range(usable_length):
                state_data[key].append(np.asarray(state_col[i], dtype=np.float32))

        We stack all frames for an episode into a single contiguous array:
            stacked = np.stack(df[col_name].values, axis=0).astype(np.float32)
            state_data[key].append(stacked[start:end])

        This reduces numpy object overhead significantly:
        - Before: ~3600 small arrays per episode × 4792 episodes = ~17M array objects
        - After: ~1 array per episode × 4792 episodes = ~5K array objects
        - Each numpy array has ~96 bytes of header overhead, so this saves ~1.6GB

        The total DATA volume is identical - only the number of Python objects differs.
        This is 100% statistically equivalent to frame-by-frame collection.

    Data Structures Returned:
        state_data: {state_key: [array(frames, dim), array(frames, dim), ...]}
            - Each list item is a 2D array containing all frames from one episode
            - Shape per item: (usable_frames_in_episode, state_dim)

        action_data: {action_key: {timestep: [array(frames, dim), ...]}}
            - Organized by action key, then by timestep in the action horizon
            - Each list item is a 2D array for one episode at one timestep
            - Shape per item: (usable_frames_in_episode, action_dim)

    Args:
        args: Tuple of
            (dataset_path, modality_configs, skip_video, episode_ids, worker_id,
             valid_step_indices_by_episode, target_keys)
            - dataset_path: Path to LeRobot dataset (as string for pickling)
            - modality_configs: Dict of ModalityConfig objects
            - skip_video: Whether to skip video loading
            - episode_ids: List of episode indices this worker should process
            - worker_id: Integer ID for progress bar positioning
            - valid_step_indices_by_episode: Optional map episode_idx -> valid step indices
            - target_keys: Optional dict with "state" and/or "action" lists. If provided,
              only those keys are collected to reduce memory. Missing entries default
              to all keys for that modality.

    Returns:
        Tuple of (state_data, action_data) dictionaries containing collected data
        for all episodes processed by this worker.
    """
    if len(args) == 5:
        dataset_path, modality_configs, skip_video, episode_ids, worker_id = args
        valid_step_indices_by_episode = None
        target_keys = None
    elif len(args) == 6:
        dataset_path, modality_configs, skip_video, episode_ids, worker_id, maybe_valid = args
        valid_step_indices_by_episode = maybe_valid if isinstance(maybe_valid, dict) else None
        target_keys = None
    else:
        (
            dataset_path,
            modality_configs,
            skip_video,
            episode_ids,
            worker_id,
            valid_step_indices_by_episode,
            target_keys,
        ) = args
    dataset_path = Path(dataset_path)

    # Each worker creates its own loader
    loader = LeRobotEpisodeLoader(
        dataset_path,
        modality_configs,
        skip_video=skip_video,
        require_stats=False,
    )

    # Determine action horizon from delta indices
    action_delta_indices = np.array(modality_configs["action"].delta_indices)
    max_delta = int(np.max(action_delta_indices))
    action_horizon = len(action_delta_indices)

    # Initialize data structures (optionally filtered by target_keys)
    all_state_keys = modality_configs.get("state", ModalityConfig([], [])).modality_keys
    if target_keys is None or "state" not in target_keys:
        state_keys_to_collect = list(all_state_keys)
    else:
        target_state_keys = set(target_keys.get("state", []))
        state_keys_to_collect = [key for key in all_state_keys if key in target_state_keys]

    all_action_keys = modality_configs["action"].modality_keys
    if target_keys is None or "action" not in target_keys:
        action_keys_to_collect = list(all_action_keys)
    else:
        target_action_keys = set(target_keys.get("action", []))
        action_keys_to_collect = [key for key in all_action_keys if key in target_action_keys]
    action_keys_to_collect_set = set(action_keys_to_collect)

    state_data: dict[str, list[np.ndarray]] = {key: [] for key in state_keys_to_collect}
    action_data: dict[str, dict[int, list[np.ndarray]]] = {
        key: {t: [] for t in range(action_horizon)} for key in action_keys_to_collect
    }

    # Collect relative action data for RELATIVE representation (joint-angle deltas)
    action_configs = modality_configs["action"].action_configs
    relative_action_keys = []
    if action_configs:
        for key_idx, key in enumerate(all_action_keys):
            if key not in action_keys_to_collect_set:
                continue
            if key_idx < len(action_configs):
                ac = action_configs[key_idx]
                if ac.rep == ActionRepresentation.RELATIVE:
                    relative_action_keys.append(key)

    relative_action_data: dict[str, dict[int, list[np.ndarray]]] = {
        key: {t: [] for t in range(action_horizon)} for key in relative_action_keys
    }

    state_delta_idx = (
        modality_configs["state"].delta_indices[-1] if "state" in modality_configs else 0
    )

    # Progress bar for this worker
    pbar = tqdm(
        episode_ids,
        desc=f"Worker {worker_id:2d}",
        position=worker_id,
        leave=False,
        ncols=80,
    )

    for episode_id in pbar:
        df = loader[episode_id]

        # Determine valid step indices for this episode.
        if valid_step_indices_by_episode is None:
            usable_length = max(0, len(df) - max_delta)
            if usable_length <= 0:
                continue
            step_indices = np.arange(usable_length, dtype=np.int64)
        else:
            step_indices = valid_step_indices_by_episode.get(episode_id)
            if step_indices is None or len(step_indices) == 0:
                continue
            step_indices = np.asarray(step_indices, dtype=np.int64)

        # -----------------------------------------------------------------
        # Collect state data using VECTORIZED approach (episode-at-a-time)
        # -----------------------------------------------------------------
        # This is the key memory optimization: instead of appending 3600
        # individual (dim,) arrays, we create ONE (3600, dim) array per episode.
        # Same data, far fewer Python objects.
        if "state" in modality_configs:
            for state_key in state_keys_to_collect:
                col_name = f"state.{state_key}"
                if col_name not in df.columns:
                    continue
                # Stack all frames: df[col].values is list of arrays → (frames, dim)
                stacked = np.stack(df[col_name].values, axis=0).astype(np.float32)
                # Gather only valid step indices (accounting for state delta index)
                state_indices = state_delta_idx + step_indices
                sliced = stacked[state_indices]
                # Append single 2D array instead of per-step 1D arrays
                state_data[state_key].append(sliced)

        # -----------------------------------------------------------------
        # Collect action data using VECTORIZED approach (per timestep)
        # -----------------------------------------------------------------
        # Actions are organized by timestep in the horizon because each
        # timestep has its own normalization statistics (temporal-aware).
        for key_idx, action_key in enumerate(all_action_keys):
            if action_key not in action_keys_to_collect_set:
                continue
            col_name = f"action.{action_key}"
            if col_name not in df.columns:
                continue

            # Stack all action frames for this episode: (total_frames, action_dim)
            action_stacked = np.stack(df[col_name].values, axis=0).astype(np.float32)

            # Check if this action key requires rel-xyz-rot6d transformation
            # (converts absolute poses to relative-to-reference-frame)
            action_config = (
                action_configs[key_idx]
                if action_configs and key_idx < len(action_configs)
                else None
            )
            needs_rel_xyz_rot6d = (
                action_config is not None
                and action_config.rep == ActionRepresentation.REL_XYZ_ROT6D
            )

            # Check if this action needs RELATIVE stats (joint-angle deltas)
            needs_relative = (
                action_config is not None and action_config.rep == ActionRepresentation.RELATIVE
            )

            # Load reference state data for rel-xyz-rot6d or relative conversion
            if needs_rel_xyz_rot6d or needs_relative:
                state_key = action_config.state_key or action_key
                state_col_name = f"state.{state_key}"
                if state_col_name not in df.columns:
                    raise ValueError(f"State key '{state_key}' not found for relative conversion.")
                ref_state_stacked = np.stack(df[state_col_name].values, axis=0).astype(np.float32)

            # Check if motion scaling is configured (CMR Versius specific)
            has_motion_scaling = action_config is not None and (
                action_config.translation_scaling_key or action_config.rotation_scaling_key
            )
            trans_scale_stacked = None
            rot_scale_stacked = None

            if has_motion_scaling:
                if action_config.translation_scaling_key:
                    trans_col_name = f"state.{action_config.translation_scaling_key}"
                    if trans_col_name in df.columns:
                        trans_scale_stacked = np.stack(df[trans_col_name].values, axis=0).astype(
                            np.float32
                        )
                if action_config.rotation_scaling_key:
                    rot_col_name = f"state.{action_config.rotation_scaling_key}"
                    if rot_col_name in df.columns:
                        rot_scale_stacked = np.stack(df[rot_col_name].values, axis=0).astype(
                            np.float32
                        )

            # Process each timestep in the action horizon separately
            # This is necessary because stats are computed per-timestep
            for t, delta in enumerate(action_delta_indices):
                # Gather actions for this timestep across valid frames only.
                action_indices = step_indices + int(delta)
                action_slice = action_stacked[action_indices]  # (num_valid_steps, dim)

                if needs_rel_xyz_rot6d:
                    # Get reference states (current pose) for all valid frames.
                    ref_states = ref_state_stacked[state_delta_idx + step_indices]

                    # NOTE: REL_XYZ_ROT6D requires frame-by-frame processing due to
                    # rotation math (can't be easily vectorized). This is the main
                    # computational bottleneck for EEF actions.
                    converted = []
                    for i in range(len(step_indices)):
                        action_value = convert_to_rel_xyz_rot6d(
                            action_data=action_slice[i : i + 1],
                            eef_pose=ref_states[i],
                            input_rotation_format=action_config.input_rotation_format,
                            reference_rotation_format=action_config.reference_rotation_format,
                            input_quat_order=action_config.input_quat_order,
                            reference_quat_order=action_config.reference_quat_order,
                        )[0]

                        # Apply motion scaling if configured (CMR Versius specific)
                        # This converts from hand-controller-space to instrument-space
                        if has_motion_scaling:
                            trans_scale = 1.0
                            rot_scale = 1.0
                            ref_idx = state_delta_idx + int(step_indices[i])
                            if trans_scale_stacked is not None:
                                trans_scale = float(trans_scale_stacked[ref_idx])
                            if rot_scale_stacked is not None:
                                rot_scale = float(rot_scale_stacked[ref_idx])
                            action_value = apply_motion_scaling_to_rel_xyz_rot6d(
                                action_value[np.newaxis, :], trans_scale, rot_scale
                            )[0]

                        # For EEF XYZ_ROT6D format, only keep xyz translation for stats
                        # rot6d is already bounded [-1, 1] and doesn't need normalization
                        if (
                            action_config.type == ActionType.EEF
                            and action_config.format == ActionFormat.XYZ_ROT6D
                        ):
                            action_value = action_value[:3]

                        converted.append(action_value)
                    action_slice = np.stack(converted, axis=0)

                # Append single 2D array: (num_valid_steps, dim) for this timestep.
                action_data[action_key][t].append(action_slice)

                # Compute relative action (delta from reference state) for RELATIVE representation
                # This is vectorized for efficiency since it's just subtraction (no rotation math)
                if needs_relative:
                    ref_states = ref_state_stacked[state_delta_idx + step_indices]
                    # Simple subtraction for joint angles: relative = action - reference_state
                    relative_slice = action_slice - ref_states
                    relative_action_data[action_key][t].append(relative_slice)

    pbar.close()
    return state_data, action_data, relative_action_data


def calculate_temporal_percentile_stats_parallel(
    dataset_path: Path | str,
    modality_configs: dict[str, ModalityConfig],
    skip_video: bool = True,
    max_episodes: int = -1,
    num_workers: int | None = None,
    episode_indices: np.ndarray | list[int] | None = None,
    embodiment_tag: EmbodimentTag | None = None,
) -> dict[str, dict[str, Any]]:
    """
    Parallel version of calculate_temporal_percentile_stats with episode-level parallelism.

    This implementation processes one modality key at a time (state or action) to
    reduce peak memory while preserving exact numerical results.

    This function distributes episode processing across multiple CPU cores for faster
    statistics computation on large datasets.

    Architecture Overview:
        1. SPLIT: Episodes are divided into chunks, one per worker
        2. KEY-BY-KEY: For each modality key, workers collect ONLY that key's data
        3. MERGE: Results are merged for that key and stats are computed immediately
        4. REPEAT: Memory is freed before moving to the next key

    Data Flow:
        Episodes → [Worker 1] → state_data_1, action_data_1 ─┐
        Episodes → [Worker 2] → state_data_2, action_data_2 ─┼→ Merge → Compute Stats
        Episodes → [Worker N] → state_data_N, action_data_N ─┘

    Statistical Equivalence:
        This parallel implementation produces IDENTICAL results to the sequential
        version. All data is collected before computing percentiles - no approximation
        or streaming algorithms are used. The only difference is the order in which
        episodes are processed, which doesn't affect percentile computation.

        Note: True streaming/incremental percentile algorithms (like t-digest) would
        NOT be statistically equivalent. We explicitly avoid such approaches.

    Memory Considerations:
        - Each worker holds data for its chunk of episodes in memory
        - Peak memory is reduced because only ONE key is merged at a time
        - For memory-constrained systems, reduce num_workers (e.g., --stats-num-workers 8)
        - The episode-at-a-time stacking optimization reduces numpy object overhead
          but does not reduce total data volume
        - Trade-off: the dataset is scanned once per key (higher runtime)

    Progress Monitoring:
        - Each worker displays its own tqdm progress bar showing episodes processed
        - A main "Merging results" progress bar shows workers completing
        - Progress bars use position parameter for clean multi-line display

    Args:
        dataset_path: Path to the LeRobot dataset directory
        modality_configs: Dictionary mapping modality names ('state', 'action') to
                         ModalityConfig objects specifying keys, delta indices, and
                         action configs (for rel-xyz-rot6d conversion)
        skip_video: If True, skip loading video data for faster iteration. Default True.
                   Video data is not needed for state/action statistics.
        max_episodes: Maximum number of episodes to process. -1 for all episodes.
                     Useful for quick testing on a subset of data.
        num_workers: Number of parallel workers. None uses os.cpu_count().
                    Set to 1 to disable parallelism (falls back to sequential in
                    launch_finetune.py). Recommended: 8-16 for large datasets.
        episode_indices: Optional list/array of episode indices to include. If None,
            uses all episodes in the dataset.
        embodiment_tag: Optional embodiment tag used to gate dataset-specific behaviors
            (e.g., skipping terminal `next.done` rows for UCSD).

    Returns:
        Dictionary with structure identical to calculate_temporal_percentile_stats:
        {
            "state": {
                "<state_key>": {
                    "q01": [dim], "q02": [dim], "q98": [dim], "q99": [dim],
                    "mean": [dim], "std": [dim], "min": [dim], "max": [dim]
                }
            },
            "action": {
                "<action_key>": {
                    "q01": [horizon, dim], "q02": [horizon, dim], ...
                }
            }
        }

    Example:
        >>> # Calculate stats with 8 workers
        >>> stats = calculate_temporal_percentile_stats_parallel(
        ...     dataset_path="/path/to/large_dataset",
        ...     modality_configs=modality_configs,
        ...     num_workers=8,
        ... )

    See Also:
        - calculate_temporal_percentile_stats: Sequential version
        - _collect_episodes_worker: Worker function with memory optimization details
    """
    import gc
    import os

    dataset_path = Path(dataset_path)
    action_delta_indices = np.array(modality_configs["action"].delta_indices)
    max_delta = int(np.max(action_delta_indices))

    # Get total episodes by creating a temporary loader
    temp_loader = LeRobotEpisodeLoader(
        dataset_path,
        modality_configs,
        skip_video=True,
        require_stats=False,
    )
    total_episodes = len(temp_loader)
    episode_ids = _resolve_episode_ids(
        total_episodes=total_episodes,
        episode_indices=episode_indices,
        max_episodes=max_episodes,
    )
    effective_lengths = [
        max(0, temp_loader.get_episode_length(ep_id) - max_delta) for ep_id in episode_ids
    ]
    valid_step_indices_by_episode = compute_valid_step_indices_parallel(
        dataset_path=dataset_path,
        embodiment_tag=embodiment_tag,
        chunk_size=temp_loader.chunk_size,
        data_path_pattern=temp_loader.data_path_pattern,
        action_delta_indices=modality_configs["action"].delta_indices,
        episode_indices=episode_ids,
        effective_lengths=effective_lengths,
        show_progress=False,
    )
    num_episodes = len(episode_ids)
    del temp_loader

    # Determine number of workers
    if num_workers is None:
        num_workers = min(os.cpu_count() or 4, num_episodes)
    num_workers = min(num_workers, num_episodes)

    if num_episodes == 0:
        return {"state": {}, "action": {}}

    print(f"Processing {num_episodes} episodes with {num_workers} workers (key-by-key)")

    # =========================================================================
    # PHASE 1: SPLIT - Divide episodes into chunks for parallel processing
    # =========================================================================
    chunk_size = (num_episodes + num_workers - 1) // num_workers  # Ceiling division
    episode_chunks = [episode_ids[i : i + chunk_size] for i in range(0, num_episodes, chunk_size)]

    # =========================================================================
    # PHASE 2: PROCESS - Run workers in parallel (key-by-key)
    # =========================================================================
    action_horizon = len(action_delta_indices)
    state_keys = modality_configs.get("state", ModalityConfig([], [])).modality_keys
    action_keys = modality_configs["action"].modality_keys

    # Identify action keys that need RELATIVE stats
    action_configs = modality_configs["action"].action_configs
    relative_action_keys = []
    if action_configs:
        for key_idx, key in enumerate(action_keys):
            if key_idx < len(action_configs):
                ac = action_configs[key_idx]
                if ac.rep == ActionRepresentation.RELATIVE:
                    relative_action_keys.append(key)

    # Calculate statistics from merged data
    result: dict[str, dict[str, Any]] = {"state": {}, "action": {}}
    relative_action_result: dict[str, dict[str, Any]] = {}

    def _compute_stats_from_concatenated(concatenated: np.ndarray) -> dict[str, np.ndarray]:
        """Compute scalar stats for a single (num_samples, dim) array."""
        return {
            "q01": np.percentile(concatenated, 1, axis=0),
            "q02": np.percentile(concatenated, 2, axis=0),
            "q98": np.percentile(concatenated, 98, axis=0),
            "q99": np.percentile(concatenated, 99, axis=0),
            "mean": np.mean(concatenated, axis=0),
            "std": np.std(concatenated, axis=0),
            "min": np.min(concatenated, axis=0),
            "max": np.max(concatenated, axis=0),
        }

    def _compute_temporal_stats(
        timestep_data: dict[int, list[np.ndarray]],
    ) -> dict[str, np.ndarray]:
        """Compute temporal stats for per-timestep lists of arrays."""
        if not timestep_data or not timestep_data[0]:
            return {}

        horizon_stats = {
            "q01": [],
            "q02": [],
            "q98": [],
            "q99": [],
            "mean": [],
            "std": [],
            "min": [],
            "max": [],
        }

        for t in range(action_horizon):
            if not timestep_data[t]:
                return {}
            concatenated = np.concatenate(timestep_data[t], axis=0)
            horizon_stats["q01"].append(np.percentile(concatenated, 1, axis=0))
            horizon_stats["q02"].append(np.percentile(concatenated, 2, axis=0))
            horizon_stats["q98"].append(np.percentile(concatenated, 98, axis=0))
            horizon_stats["q99"].append(np.percentile(concatenated, 99, axis=0))
            horizon_stats["mean"].append(np.mean(concatenated, axis=0))
            horizon_stats["std"].append(np.std(concatenated, axis=0))
            horizon_stats["min"].append(np.min(concatenated, axis=0))
            horizon_stats["max"].append(np.max(concatenated, axis=0))
            del concatenated

        return {
            stat_name: np.stack(stat_values, axis=0)
            for stat_name, stat_values in horizon_stats.items()
        }

    from concurrent.futures import as_completed

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # ---------------------------------------------------------------------
        # State statistics: non-temporal, processed key-by-key
        # ---------------------------------------------------------------------
        for state_key in state_keys:
            target_keys = {"state": [state_key], "action": []}
            args_list = [
                (
                    str(dataset_path),
                    modality_configs,
                    skip_video,
                    chunk,
                    worker_id,
                    (
                        None
                        if valid_step_indices_by_episode is None
                        else {
                            ep_id: valid_step_indices_by_episode[ep_id]
                            for ep_id in chunk
                            if ep_id in valid_step_indices_by_episode
                        }
                    ),
                    target_keys,
                )
                for worker_id, chunk in enumerate(episode_chunks)
            ]

            merged_state_data: list[np.ndarray] = []

            # Print newlines to make room for worker progress bars
            print("\n" * num_workers)
            futures = {
                executor.submit(_collect_episodes_worker, args): i
                for i, args in enumerate(args_list)
            }
            main_pbar = tqdm(
                total=len(futures),
                desc=f"Merging state:{state_key}",
                position=num_workers,
                leave=True,
                ncols=80,
            )

            for future in as_completed(futures):
                state_data, _, _ = future.result()
                merged_state_data.extend(state_data.get(state_key, []))
                main_pbar.update(1)

            main_pbar.close()
            print("\n")

            if merged_state_data:
                concatenated = np.concatenate(merged_state_data, axis=0)
                result["state"][state_key] = _compute_stats_from_concatenated(concatenated)
                del concatenated

            del merged_state_data
            gc.collect()

        # ---------------------------------------------------------------------
        # Action statistics: temporal-aware, processed key-by-key
        # ---------------------------------------------------------------------
        for action_key in action_keys:
            target_keys = {"state": [], "action": [action_key]}
            args_list = [
                (
                    str(dataset_path),
                    modality_configs,
                    skip_video,
                    chunk,
                    worker_id,
                    (
                        None
                        if valid_step_indices_by_episode is None
                        else {
                            ep_id: valid_step_indices_by_episode[ep_id]
                            for ep_id in chunk
                            if ep_id in valid_step_indices_by_episode
                        }
                    ),
                    target_keys,
                )
                for worker_id, chunk in enumerate(episode_chunks)
            ]

            merged_action_data: dict[int, list[np.ndarray]] = {t: [] for t in range(action_horizon)}
            merged_relative_action_data: dict[int, list[np.ndarray]] | None = None
            if action_key in relative_action_keys:
                merged_relative_action_data = {t: [] for t in range(action_horizon)}

            print("\n" * num_workers)
            futures = {
                executor.submit(_collect_episodes_worker, args): i
                for i, args in enumerate(args_list)
            }
            main_pbar = tqdm(
                total=len(futures),
                desc=f"Merging action:{action_key}",
                position=num_workers,
                leave=True,
                ncols=80,
            )

            for future in as_completed(futures):
                _, action_data, relative_action_data = future.result()

                for t in range(action_horizon):
                    merged_action_data[t].extend(action_data.get(action_key, {}).get(t, []))

                if merged_relative_action_data is not None:
                    for t in range(action_horizon):
                        merged_relative_action_data[t].extend(
                            relative_action_data.get(action_key, {}).get(t, [])
                        )

                main_pbar.update(1)

            main_pbar.close()
            print("\n")

            action_stats = _compute_temporal_stats(merged_action_data)
            if action_stats:
                result["action"][action_key] = action_stats

            if merged_relative_action_data is not None:
                relative_stats = _compute_temporal_stats(merged_relative_action_data)
                if relative_stats:
                    relative_action_result[action_key] = relative_stats

            del merged_action_data
            del merged_relative_action_data
            gc.collect()

    if relative_action_result:
        result["relative_action"] = relative_action_result

    return result


def main(
    dataset_path: Path | str,
    embodiment_tag: EmbodimentTag,
    include_splits: list[str] | None = None,
    exclude_splits: list[str] | None = None,
):
    """CLI entrypoint for stats generation with optional split filtering."""
    episode_indices = None
    if include_splits or exclude_splits:
        info = load_info_json(dataset_path)
        total_episodes = info.get("total_episodes")
        episode_indices = resolve_episode_indices(
            info,
            include_splits=include_splits,
            exclude_splits=exclude_splits,
            total_episodes=total_episodes,
        )
    generate_stats(dataset_path, episode_indices=episode_indices)
    generate_rel_stats(dataset_path, embodiment_tag, episode_indices=episode_indices)


if __name__ == "__main__":
    import tyro

    tyro.cli(main)
