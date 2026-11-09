"""Proper scoring rules for sample-based distributional forecasts."""

from __future__ import annotations

import numpy as np

__all__ = ["crps_ensemble", "pit_values", "interval_coverage"]


def crps_ensemble(samples: np.ndarray, observations: np.ndarray | float) -> np.ndarray:
    """CRPS of an ensemble (sample) forecast, in O(n log n) time and O(n) memory.

    Uses ``CRPS = E|X - y| - 0.5 E|X - X'|`` with the sorted-sample identity
    ``E|X - X'| = (2 / n^2) * sum_i (2i - n - 1) x_(i)``, which is exactly the
    pairwise (V-statistic) estimator without materialising an n x n matrix.

    Args:
        samples: Shape ``(n_members,)`` for one forecast, or
            ``(n_obs, n_members)`` for a batch.
        observations: Scalar, or shape ``(n_obs,)`` matching the batch.

    Returns:
        CRPS per observation (0-d array for a single forecast). Lower is better.
    """
    x = np.sort(np.asarray(samples, dtype=float), axis=-1)
    y = np.asarray(observations, dtype=float)
    n = x.shape[-1]
    mae = np.mean(np.abs(x - y[..., np.newaxis]), axis=-1)
    weights = (2.0 * np.arange(1, n + 1) - n - 1) / n**2
    half_spread = x @ weights
    return np.asarray(mae - half_spread)


def pit_values(samples: np.ndarray, observations: np.ndarray | float) -> np.ndarray:
    """Probability integral transform: the forecast CDF evaluated at the outcome.

    A calibrated forecast gives PIT values that are Uniform(0, 1).

    Args:
        samples: Shape ``(n_members,)`` or ``(n_obs, n_members)``.
        observations: Scalar or shape ``(n_obs,)``.

    Returns:
        PIT value(s) in [0, 1].
    """
    x = np.asarray(samples, dtype=float)
    y = np.asarray(observations, dtype=float)
    return np.asarray(np.mean(x <= y[..., np.newaxis], axis=-1))


def interval_coverage(
    samples: np.ndarray, observations: np.ndarray | float, level: float
) -> tuple[np.ndarray, np.ndarray]:
    """Whether outcomes fall inside the central ``level`` prediction interval.

    Args:
        samples: Shape ``(n_members,)`` or ``(n_obs, n_members)``.
        observations: Scalar or shape ``(n_obs,)``.
        level: Central interval probability, e.g. 0.9.

    Returns:
        Tuple ``(covered, width)``: boolean coverage indicator(s) and interval width(s).
    """
    x = np.asarray(samples, dtype=float)
    y = np.asarray(observations, dtype=float)
    tail = (1.0 - level) / 2.0
    lo = np.quantile(x, tail, axis=-1)
    hi = np.quantile(x, 1.0 - tail, axis=-1)
    return np.asarray((y >= lo) & (y <= hi)), np.asarray(hi - lo)
