from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def load_info_json(dataset_path: Path | str) -> dict[str, Any]:
    """
    Load a LeRobot dataset's `meta/info.json`.

    Args:
        dataset_path: Path to the dataset root directory.

    Returns:
        Parsed info.json contents as a dictionary.

    Raises:
        FileNotFoundError: If info.json does not exist.
        json.JSONDecodeError: If info.json is not valid JSON.
    """
    dataset_path = Path(dataset_path)
    info_path = dataset_path / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"Missing info.json at {info_path}")
    with open(info_path, "r") as f:
        return json.load(f)


def parse_split_ranges(split_spec: str | list[str]) -> list[tuple[int, int]]:
    """
    Parse split range specifications into inclusive (start, end) tuples.

    Supported formats:
        - "0:499" (inclusive end)
        - "0-499" (inclusive end)
        - "42" (single episode index)
        - Comma/whitespace-separated lists, e.g. "0:99, 200:299"

    Args:
        split_spec: Split range specification string or list of strings.

    Returns:
        List of inclusive (start, end) tuples.

    Raises:
        ValueError: If a range is malformed or has end < start.
    """
    if isinstance(split_spec, list):
        raw_specs = split_spec
    else:
        raw_specs = [split_spec]

    ranges: list[tuple[int, int]] = []
    for raw in raw_specs:
        parts = [p for p in raw.replace(",", " ").split() if p]
        for part in parts:
            if ":" in part:
                start_str, end_str = part.split(":", 1)
            elif "-" in part:
                start_str, end_str = part.split("-", 1)
            else:
                start_str, end_str = part, part

            start = int(start_str)
            end = int(end_str)
            if end < start:
                raise ValueError(f"Invalid split range '{part}': end < start")
            ranges.append((start, end))

    return ranges


def build_split_index_map(info: dict[str, Any]) -> dict[str, np.ndarray]:
    """
    Build a mapping from split name to sorted episode indices.

    Args:
        info: Parsed info.json dictionary containing a `splits` field.

    Returns:
        Dict mapping lowercased split names to sorted numpy arrays of episode indices.
    """
    splits = info.get("splits", {}) or {}
    split_indices: dict[str, np.ndarray] = {}
    for split_name, split_spec in splits.items():
        ranges = parse_split_ranges(split_spec)
        indices: list[int] = []
        for start, end in ranges:
            indices.extend(range(start, end + 1))
        split_indices[split_name.lower()] = np.array(sorted(set(indices)), dtype=int)
    return split_indices


def resolve_episode_indices(
    info: dict[str, Any],
    include_splits: list[str] | None = None,
    exclude_splits: list[str] | None = None,
    *,
    total_episodes: int | None = None,
) -> np.ndarray | None:
    """
    Resolve an allowed episode index set from split allow/deny lists.

    Args:
        info: Parsed info.json dictionary.
        include_splits: Optional allowlist of split names to include. If provided,
            only these splits are used as the base set.
        exclude_splits: Optional denylist of split names to exclude.
        total_episodes: Total episode count fallback if not present in info.json.

    Returns:
        Sorted numpy array of allowed episode indices, or None if no filtering
        is requested.

    Raises:
        ValueError: If include_splits contains a split name not in info.json.
        ValueError: If filtering is requested but no splits are defined.
    """
    if not include_splits and not exclude_splits:
        return None

    split_map = build_split_index_map(info)
    if not split_map:
        raise ValueError("Split filtering requested but info.json has no splits.")

    include_norm = [s.lower() for s in include_splits] if include_splits else []
    exclude_norm = [s.lower() for s in exclude_splits] if exclude_splits else []

    if include_norm:
        missing = [s for s in include_norm if s not in split_map]
        if missing:
            raise ValueError(f"Requested include_splits not found in info.json: {missing}")
        allowed = np.concatenate([split_map[s] for s in include_norm], axis=0)
        allowed_set = set(allowed.tolist())
    else:
        total = info.get("total_episodes", total_episodes)
        if total is None:
            raise ValueError("total_episodes missing from info.json and not provided.")
        allowed_set = set(range(int(total)))

    for split_name in exclude_norm:
        if split_name not in split_map:
            continue
        allowed_set -= set(split_map[split_name].tolist())

    if total_episodes is not None:
        total = int(total_episodes)
        allowed_set = {i for i in allowed_set if 0 <= i < total}

    if not allowed_set:
        raise ValueError("Split filtering removed all episodes.")

    return np.array(sorted(allowed_set), dtype=int)
