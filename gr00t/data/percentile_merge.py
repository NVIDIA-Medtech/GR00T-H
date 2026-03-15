"""Utilities for merging percentile statistics with piecewise-linear CDFs."""

from __future__ import annotations

from typing import Callable

import numpy as np


QUANTILE_KEY_ORDER = ("min", "q01", "q02", "q98", "q99", "max")
QUANTILE_PROBS = np.array([0.0, 0.01, 0.02, 0.98, 0.99, 1.0])
TARGET_QUANTILE_KEYS = ("q01", "q02", "q98", "q99")
TARGET_QUANTILE_PROBS = np.array([0.01, 0.02, 0.98, 0.99])
QUANTILE_PROB_BY_KEY = {
    "min": 0.0,
    "q01": 0.01,
    "q02": 0.02,
    "q98": 0.98,
    "q99": 0.99,
    "max": 1.0,
}


def _normalize_weights(weights: list[float] | np.ndarray) -> np.ndarray:
    """Normalize dataset weights to sum to 1.

    Args:
        weights: Raw dataset sampling weights.

    Returns:
        Normalized weights as a numpy array that sums to 1.

    Raises:
        ValueError: If weights sum to zero or contain negative values.
    """
    weights_array = np.array(weights, dtype=np.float64)
    if np.any(weights_array < 0):
        raise ValueError("Dataset sampling weights must be non-negative.")
    total = weights_array.sum()
    if total <= 0:
        raise ValueError("Dataset sampling weights must sum to a positive value.")
    return weights_array / total


def _flatten_quantiles(
    quantiles: dict[str, np.ndarray],
    key_order: tuple[str, ...] = QUANTILE_KEY_ORDER,
) -> tuple[np.ndarray, tuple[int, ...]]:
    """Flatten quantile arrays into a (num_points, num_features) matrix.

    Args:
        quantiles: Dictionary mapping quantile keys to numpy arrays.
        key_order: Ordered quantile keys to stack along the first dimension.

    Returns:
        A tuple containing:
        - stacked quantiles with shape (num_points, num_features)
        - original feature shape (used for reshaping later)
    """
    first = np.asarray(quantiles[key_order[0]], dtype=np.float64)
    feature_shape = first.shape
    stacked = np.stack(
        [np.asarray(quantiles[key], dtype=np.float64).reshape(-1) for key in key_order],
        axis=0,
    )
    return stacked, feature_shape


def _validate_no_nan_or_none(values: np.ndarray, context: str) -> None:
    """Validate that an array does not contain NaN/None-derived values.

    None values become NaN when cast to float arrays, so a NaN check catches both
    invalid cases requested by callers.

    Args:
        values: Numeric numpy array to validate.
        context: Context string for error reporting.

    Raises:
        ValueError: If the array contains NaN or None-derived values.
    """
    if np.any(np.isnan(values)):
        raise ValueError(
            "Quantile merge inputs cannot contain NaN or None values. "
            f"Invalid value detected in {context}."
        )


def _validate_quantile_monotonicity(
    quantile_values: np.ndarray,
    context: str,
) -> None:
    """Validate that quantile values are non-decreasing per feature.

    Args:
        quantile_values: Array of shape (num_points, num_features) containing
            ordered quantile values per feature.
        context: Context string for error reporting.

    Raises:
        ValueError: If any feature has decreasing quantile values.
    """
    _validate_no_nan_or_none(quantile_values, context=context)
    diffs = np.diff(quantile_values, axis=0)
    if np.any(diffs < 0):
        raise ValueError(
            "Quantile values must be non-decreasing for piecewise CDF merge. "
            f"Invalid ordering detected in {context}."
        )


def _dedupe_piecewise_knots(
    quantile_values: np.ndarray,
    quantile_probs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Collapse duplicate x-knots and keep monotonic probabilities.

    `np.interp` behaves ambiguously when x-knots are duplicated. This helper
    collapses repeated quantile values to unique x locations and uses the
    maximum probability seen at each location, then applies cumulative-max to
    preserve non-decreasing CDF behavior.

    Args:
        quantile_values: Non-decreasing quantile values, shape (num_points,).
        quantile_probs: Corresponding quantile probabilities, shape (num_points,).

    Returns:
        Tuple of (unique_quantile_values, monotonic_quantile_probs).
    """
    unique_values, inverse_indices = np.unique(quantile_values, return_inverse=True)
    dedup_probs = np.full(unique_values.shape, -np.inf, dtype=np.float64)
    np.maximum.at(dedup_probs, inverse_indices, quantile_probs)
    dedup_probs = np.maximum.accumulate(dedup_probs)
    return unique_values, dedup_probs


def build_piecewise_cdf(
    quantile_values: np.ndarray,
    quantile_probs: np.ndarray,
) -> Callable[[np.ndarray], np.ndarray]:
    """Create a piecewise-linear CDF for a single feature.

    Args:
        quantile_values: Array of shape (num_points,) containing quantile values.
        quantile_probs: Array of shape (num_points,) containing corresponding CDF
            probabilities in ascending order.

    Returns:
        Callable that maps input values to CDF probabilities.
    """

    quantile_values = np.asarray(quantile_values, dtype=np.float64)
    quantile_probs = np.asarray(quantile_probs, dtype=np.float64)
    _validate_no_nan_or_none(quantile_values, context="piecewise CDF quantile values")
    _validate_no_nan_or_none(quantile_probs, context="piecewise CDF quantile probabilities")
    if quantile_values.ndim != 1 or quantile_probs.ndim != 1:
        raise ValueError(
            "Piecewise CDF inputs must be 1D arrays. "
            f"Got shapes values={quantile_values.shape}, probs={quantile_probs.shape}."
        )
    if quantile_values.size == 0:
        raise ValueError("Piecewise CDF requires at least one quantile knot.")
    if quantile_values.size != quantile_probs.size:
        raise ValueError(
            "Piecewise CDF requires matching lengths for quantile values and probabilities. "
            f"Got {quantile_values.size} values and {quantile_probs.size} probabilities."
        )
    if np.any(np.diff(quantile_values) < 0):
        raise ValueError("Piecewise CDF quantile values must be non-decreasing.")
    if np.any(np.diff(quantile_probs) < 0):
        raise ValueError("Piecewise CDF probabilities must be non-decreasing.")
    if np.any((quantile_probs < 0.0) | (quantile_probs > 1.0)):
        raise ValueError("Piecewise CDF probabilities must be within [0, 1].")

    unique_values, unique_probs = _dedupe_piecewise_knots(quantile_values, quantile_probs)

    def _cdf(x: np.ndarray) -> np.ndarray:
        return np.interp(x, unique_values, unique_probs, left=0.0, right=1.0)

    return _cdf


def merge_cdfs(
    cdfs: list[Callable[[np.ndarray], np.ndarray]],
    weights: list[float] | np.ndarray,
) -> Callable[[np.ndarray], np.ndarray]:
    """Create a weighted mixture of multiple CDFs.

    Args:
        cdfs: List of CDF callables, each mapping values to probabilities.
        weights: Sampling weights for each CDF.

    Returns:
        Callable representing the weighted mixture CDF.
    """
    if len(cdfs) != len(weights):
        raise ValueError(
            "CDF merge requires the same number of CDFs and weights. "
            f"Got {len(cdfs)} CDFs and {len(weights)} weights."
        )
    normalized_weights = _normalize_weights(weights)

    def _merged_cdf(x: np.ndarray) -> np.ndarray:
        cdf_values = np.zeros_like(x, dtype=np.float64)
        for cdf, weight in zip(cdfs, normalized_weights):
            cdf_values += weight * cdf(x)
        return cdf_values

    return _merged_cdf


def invert_cdf(
    cdf: Callable[[np.ndarray], np.ndarray],
    x_grid: np.ndarray,
    target_probs: np.ndarray,
) -> np.ndarray:
    """Invert a CDF over a predefined grid to obtain quantile values.

    Args:
        cdf: Callable CDF to invert.
        x_grid: Monotonic grid of x values to evaluate the CDF on.
        target_probs: Target probabilities to invert (e.g., [0.01, 0.02, 0.98, 0.99]).

    Returns:
        Array of quantile values corresponding to target_probs.
    """
    if x_grid.size == 1:
        return np.full_like(target_probs, x_grid[0], dtype=np.float64)

    cdf_values = np.maximum.accumulate(cdf(x_grid))
    cdf_values[0] = 0.0
    cdf_values[-1] = 1.0

    quantiles = np.zeros_like(target_probs, dtype=np.float64)
    for idx, prob in enumerate(target_probs):
        if prob <= cdf_values[0]:
            quantiles[idx] = x_grid[0]
            continue
        if prob >= cdf_values[-1]:
            quantiles[idx] = x_grid[-1]
            continue
        right = np.searchsorted(cdf_values, prob, side="left")
        left = max(right - 1, 0)
        if cdf_values[right] == cdf_values[left]:
            quantiles[idx] = x_grid[right]
        else:
            ratio = (prob - cdf_values[left]) / (cdf_values[right] - cdf_values[left])
            quantiles[idx] = x_grid[left] + ratio * (x_grid[right] - x_grid[left])
    return quantiles


def merge_piecewise_linear_quantiles(
    per_dataset_quantiles: list[dict[str, np.ndarray]],
    weights: list[float] | np.ndarray,
    key_order: tuple[str, ...] = QUANTILE_KEY_ORDER,
    target_keys: tuple[str, ...] = TARGET_QUANTILE_KEYS,
    target_probs: np.ndarray = TARGET_QUANTILE_PROBS,
) -> dict[str, np.ndarray]:
    """Merge per-dataset percentile statistics using piecewise-linear CDFs.

    Args:
        per_dataset_quantiles: List of quantile dicts, each containing keys in key_order
            and arrays of identical shape (e.g., (dim,) or (horizon, dim)).
        weights: Sampling weights for each dataset.
        key_order: Ordered quantile keys defining the CDF breakpoints.
        target_keys: Output quantile keys to return.
        target_probs: Target probabilities corresponding to target_keys.

    Returns:
        Dictionary mapping target_keys to merged quantile arrays with the same shape
        as the input quantile arrays.
    """
    if not per_dataset_quantiles:
        raise ValueError("Quantile merge requires at least one dataset quantile payload.")
    if len(per_dataset_quantiles) != len(weights):
        raise ValueError(
            "Quantile merge requires the same number of quantile payloads and weights. "
            f"Got {len(per_dataset_quantiles)} payloads and {len(weights)} weights."
        )
    if len(key_order) == 0:
        raise ValueError("Quantile merge requires a non-empty key_order.")
    if len(set(key_order)) != len(key_order):
        raise ValueError(f"Quantile merge key_order contains duplicate keys: {key_order}.")
    if len(target_keys) == 0:
        raise ValueError("Quantile merge requires at least one target quantile key.")
    if len(set(target_keys)) != len(target_keys):
        raise ValueError(f"Quantile merge target_keys contains duplicate keys: {target_keys}.")

    target_probs = np.asarray(target_probs, dtype=np.float64)
    _validate_no_nan_or_none(target_probs, context="quantile merge target probabilities")
    if target_probs.ndim != 1:
        raise ValueError(
            f"Quantile merge target_probs must be a 1D array. Got shape {target_probs.shape}."
        )
    if len(target_keys) != target_probs.size:
        raise ValueError(
            "Quantile merge requires target_keys and target_probs to have equal lengths. "
            f"Got {len(target_keys)} target keys and {target_probs.size} target probabilities."
        )
    if np.any(np.diff(target_probs) < 0):
        raise ValueError(f"Quantile merge target_probs must be non-decreasing. Got {target_probs}.")
    if np.any((target_probs < 0.0) | (target_probs > 1.0)):
        raise ValueError(f"Quantile merge target_probs must be within [0, 1]. Got {target_probs}.")

    try:
        quantile_probs = np.array(
            [QUANTILE_PROB_BY_KEY[key] for key in key_order], dtype=np.float64
        )
    except KeyError as exc:
        raise ValueError(f"Unsupported quantile key in key_order: {exc}.") from exc
    if np.any(np.diff(quantile_probs) < 0):
        raise ValueError(
            "Quantile probabilities must be non-decreasing. "
            f"Got {quantile_probs} for key_order={key_order}."
        )
    stacked_quantiles = []
    feature_shape = None

    for idx, quantiles in enumerate(per_dataset_quantiles):
        missing = [key for key in key_order if key not in quantiles]
        if missing:
            raise ValueError(
                f"Missing required quantiles for piecewise merge: {missing} in dataset index {idx}."
            )
        stacked, shape = _flatten_quantiles(quantiles, key_order=key_order)
        if feature_shape is None:
            feature_shape = shape
        elif feature_shape != shape:
            raise ValueError(
                "Quantile shapes must match across datasets. "
                f"Expected {feature_shape}, got {shape} in dataset index {idx}."
            )
        _validate_quantile_monotonicity(
            stacked,
            context=f"dataset index {idx}",
        )
        stacked_quantiles.append(stacked)

    _, num_features = stacked_quantiles[0].shape
    merged_quantiles = np.zeros((len(target_probs), num_features), dtype=np.float64)

    for feature_idx in range(num_features):
        feature_quantiles = [dataset[:, feature_idx] for dataset in stacked_quantiles]
        x_grid = np.unique(np.concatenate(feature_quantiles))
        if x_grid.size == 0:
            raise ValueError(
                f"Quantile merge failed: empty grid encountered for feature index {feature_idx}."
            )
        cdfs = [build_piecewise_cdf(values, quantile_probs) for values in feature_quantiles]
        merged_cdf = merge_cdfs(cdfs, weights)
        merged_quantiles[:, feature_idx] = invert_cdf(merged_cdf, x_grid, target_probs)

    reshaped = {
        key: merged_quantiles[idx].reshape(feature_shape) for idx, key in enumerate(target_keys)
    }
    return reshaped
