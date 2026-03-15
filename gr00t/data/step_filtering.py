"""
Shared step-index filtering utilities for dataset loading and stats generation.

This module centralizes dataset-specific filtering rules so train-time sampling
and offline percentile-stat generation use the same valid step indices.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import json
import os
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from tqdm import tqdm

from gr00t.data.types import (
    CMR_ENGAGED_LEFT_KEY,
    CMR_ENGAGED_RIGHT_KEY,
    CMR_RAW_INDEX_ARM_LINKED_LEFT,
    CMR_RAW_INDEX_ARM_LINKED_RIGHT,
    CMR_RAW_INDEX_HAPTIC_ENGAGED_LEFT,
    CMR_RAW_INDEX_HAPTIC_ENGAGED_RIGHT,
    EMBODIMENTS_SKIP_NEXT_DONE,
    EmbodimentTag,
)


MAX_FILTER_WORKERS = 128


def _filter_episode_fast(
    episode_idx: int,
    dataset_path: Path,
    chunk_size: int,
    data_path_pattern: str,
    action_delta_indices: list[int],
    effective_length: int,
) -> tuple[int, np.ndarray]:
    """Filter one episode using CMR clutch-aware rules.

    Reads only `observation.state` via PyArrow and applies the same constraints
    used at train time:
    1. Arm linkage cannot change within the action horizon.
    2. At least one side must be engaged within the action horizon.

    Args:
        episode_idx: Episode index to process.
        dataset_path: Dataset root directory.
        chunk_size: Episodes per parquet chunk.
        data_path_pattern: Parquet path pattern from LeRobotEpisodeLoader.
        action_delta_indices: Action horizon offsets.
        effective_length: Candidate anchor-step count for this episode.

    Returns:
        Tuple of `(episode_idx, valid_step_indices)`.
    """
    chunk_idx = episode_idx // chunk_size
    parquet_path = dataset_path / data_path_pattern.format(
        episode_chunk=chunk_idx, episode_index=episode_idx
    )

    table = pq.read_table(parquet_path, columns=["observation.state"])
    state_data = table.column("observation.state").to_pylist()

    eng_left = np.array([s[CMR_RAW_INDEX_HAPTIC_ENGAGED_LEFT] for s in state_data], dtype=bool)
    eng_right = np.array([s[CMR_RAW_INDEX_HAPTIC_ENGAGED_RIGHT] for s in state_data], dtype=bool)
    al_left = np.array([s[CMR_RAW_INDEX_ARM_LINKED_LEFT] for s in state_data], dtype=np.float32)
    al_right = np.array([s[CMR_RAW_INDEX_ARM_LINKED_RIGHT] for s in state_data], dtype=np.float32)

    valid_indices: list[int] = []
    for step_idx in range(effective_length):
        horizon_indices = np.array([step_idx + d for d in action_delta_indices])
        if horizon_indices[-1] >= len(state_data):
            continue
        if len(np.unique(al_left[horizon_indices])) > 1:
            continue
        if len(np.unique(al_right[horizon_indices])) > 1:
            continue
        if not eng_left[horizon_indices].any() and not eng_right[horizon_indices].any():
            continue
        valid_indices.append(step_idx)

    return episode_idx, np.array(valid_indices, dtype=np.int32)


def _check_is_cmr_data(dataset_path: Path) -> bool:
    """Check whether dataset metadata indicates CMR clutch-aware fields.

    Args:
        dataset_path: Dataset root directory.

    Returns:
        True if `meta/modality.json` contains both CMR engagement keys.
    """
    try:
        modality_path = dataset_path / "meta" / "modality.json"
        if not modality_path.exists():
            return False
        with open(modality_path, "r") as f:
            modality_config = json.load(f)
        state_keys = set(modality_config.get("state", {}).keys())
        action_keys = set(modality_config.get("action", {}).keys())
        all_keys = state_keys | action_keys
        return CMR_ENGAGED_LEFT_KEY in all_keys and CMR_ENGAGED_RIGHT_KEY in all_keys
    except Exception:
        return False


def _check_has_next_done(dataset_path: Path) -> bool:
    """Check whether dataset metadata declares a `next.done` feature.

    Args:
        dataset_path: Dataset root directory.

    Returns:
        True if `meta/info.json` lists `next.done` in features.
    """
    info_path = dataset_path / "meta" / "info.json"
    if not info_path.exists():
        return False
    try:
        with open(info_path, "r") as f:
            info = json.load(f)
        return "next.done" in info.get("features", {})
    except Exception:
        return False


def _should_skip_next_done(embodiment_tag: EmbodimentTag | None, dataset_path: Path) -> bool:
    """Determine whether terminal `next.done` rows should be skipped.

    Args:
        embodiment_tag: Dataset embodiment tag.
        dataset_path: Dataset root directory.

    Returns:
        True when the embodiment is allowlisted and dataset metadata includes
        `next.done`.
    """
    if embodiment_tag is None:
        return False
    if embodiment_tag not in EMBODIMENTS_SKIP_NEXT_DONE:
        return False
    return _check_has_next_done(dataset_path)


def _filter_episode_done_fast(
    episode_idx: int,
    dataset_path: Path,
    chunk_size: int,
    data_path_pattern: str,
    action_delta_indices: list[int],
    effective_length: int,
) -> tuple[int, np.ndarray]:
    """Filter one episode to exclude anchor steps crossing terminal `next.done`.

    For allowlisted embodiments with terminal padding, this trims candidate
    anchor steps so the action horizon never includes rows at or after the first
    `next.done=True`.

    Args:
        episode_idx: Episode index to process.
        dataset_path: Dataset root directory.
        chunk_size: Episodes per parquet chunk.
        data_path_pattern: Parquet path pattern from LeRobotEpisodeLoader.
        action_delta_indices: Action horizon offsets.
        effective_length: Candidate anchor-step count for this episode.

    Returns:
        Tuple of `(episode_idx, valid_step_indices)`.
    """
    chunk_idx = episode_idx // chunk_size
    parquet_path = dataset_path / data_path_pattern.format(
        episode_chunk=chunk_idx, episode_index=episode_idx
    )

    table = pq.read_table(parquet_path, columns=["next.done"])
    done = np.array(table.column("next.done").to_pylist(), dtype=bool)

    max_delta = int(max(action_delta_indices))
    usable_length = effective_length
    if done.any():
        first_done = int(np.argmax(done))
        usable_length = min(usable_length, max(0, first_done - max_delta))

    if usable_length <= 0:
        return episode_idx, np.array([], dtype=np.int32)
    return episode_idx, np.arange(usable_length, dtype=np.int32)


def compute_valid_step_indices_parallel(
    dataset_path: Path | str,
    embodiment_tag: EmbodimentTag | None,
    chunk_size: int,
    data_path_pattern: str,
    action_delta_indices: list[int] | np.ndarray,
    episode_indices: list[int] | np.ndarray,
    effective_lengths: list[int] | np.ndarray,
    num_workers: int | None = None,
    max_filter_workers: int = MAX_FILTER_WORKERS,
    show_progress: bool = True,
) -> dict[int, np.ndarray] | None:
    """Compute valid step indices per episode using shared filtering semantics.

    Filtering mode is selected in this order:
    1. CMR clutch-aware filtering (if CMR keys are present in modality metadata).
    2. Terminal-step filtering using `next.done` (if embodiment is allowlisted).
    3. No special filtering (returns `None`).

    Args:
        dataset_path: Dataset root directory.
        embodiment_tag: Embodiment tag used for allowlisted behavior gates.
        chunk_size: Episodes per parquet chunk.
        data_path_pattern: Parquet path pattern from LeRobotEpisodeLoader.
        action_delta_indices: Action horizon offsets.
        episode_indices: Episode indices to process.
        effective_lengths: Candidate anchor-step counts aligned to `episode_indices`.
        num_workers: Optional worker count for ProcessPoolExecutor.
        max_filter_workers: Upper bound on worker count.
        show_progress: Whether to print filtering progress and summary.

    Returns:
        - `None` when no special filtering is required.
        - `dict[episode_idx, np.ndarray]` when filtering is applied.
        - Empty dict when filtering is applied but no valid indices remain.

    Raises:
        ValueError: If `episode_indices` and `effective_lengths` lengths differ.
    """
    episode_indices = [int(i) for i in np.asarray(episode_indices, dtype=int).tolist()]
    effective_lengths = [int(max(0, e)) for e in np.asarray(effective_lengths, dtype=int).tolist()]
    if len(episode_indices) != len(effective_lengths):
        raise ValueError(
            "episode_indices and effective_lengths must have same length. "
            f"Got {len(episode_indices)} and {len(effective_lengths)}."
        )
    if not episode_indices:
        return {}

    dataset_path = Path(dataset_path)
    action_delta_indices = [int(d) for d in np.asarray(action_delta_indices, dtype=int).tolist()]

    if _check_is_cmr_data(dataset_path):
        filter_fn = _filter_episode_fast
        filter_desc = "Clutch-aware filtering"
    elif _should_skip_next_done(embodiment_tag, dataset_path):
        filter_fn = _filter_episode_done_fast
        filter_desc = "Terminal-step filtering (next.done)"
    else:
        return None

    if num_workers is None:
        num_workers = min(os.cpu_count() or 32, max_filter_workers)
    num_workers = max(1, min(num_workers, max_filter_workers, len(episode_indices)))

    filter_fn = partial(
        filter_fn,
        dataset_path=dataset_path,
        chunk_size=chunk_size,
        data_path_pattern=data_path_pattern,
        action_delta_indices=action_delta_indices,
    )

    if show_progress:
        print(f"{filter_desc}: {len(episode_indices)} episodes with {num_workers} workers...")

    results: dict[int, np.ndarray] = {}
    total_original = 0
    total_valid = 0

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(filter_fn, ep_idx, effective_length=eff_len): (ep_idx, eff_len)
            for ep_idx, eff_len in zip(episode_indices, effective_lengths)
        }

        iter_futures = as_completed(futures)
        if show_progress:
            iter_futures = tqdm(iter_futures, total=len(futures), desc="Filtering episodes")

        for future in iter_futures:
            ep_idx, valid_indices = future.result()
            _, eff_len = futures[future]
            total_original += eff_len
            total_valid += len(valid_indices)
            if len(valid_indices) > 0:
                results[ep_idx] = valid_indices

    total_filtered = total_original - total_valid
    if show_progress and total_filtered > 0 and total_original > 0:
        print(
            f"{filter_desc} complete: {total_filtered}/{total_original} indices filtered "
            f"({100 * total_filtered / total_original:.1f}%)"
        )

    return results
