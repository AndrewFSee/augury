"""Synth-compatible multi-scale CRPS scoring pipeline.

Implements the scoring methodology used by the Synth subnet (Bittensor SN50)
for evaluating probabilistic price path forecasts:

1. Convert paths to basis-point price changes at multiple temporal scales
2. Compute CRPS at each scale (5-min, 30-min, 3-hour, 24-hour)
3. Percentile capping (worst 10% at 90th percentile)
4. Best-score subtraction (normalization)
5. EWMA smoothing over rolling window

References:
    https://deepwiki.com/mode-network/synth-subnet/1-overview
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

__all__ = ["SynthScorer", "SynthScoreResult"]

logger = logging.getLogger(__name__)

# Default asset-specific EWMA coefficients (from Synth subnet)
DEFAULT_ASSET_COEFFICIENTS: dict[str, float] = {
    "BTC": 1.0,
    "ETH": 0.621,
    "XAU": 1.455,
    "SOL": 0.502,
    "SPY": 1.0,
    "NVDA": 0.8,
    "TSLA": 0.6,
    "AAPL": 0.9,
    "GOOGL": 0.9,
}


@dataclass
class SynthScoreResult:
    """Results from Synth-compatible scoring.

    Attributes:
        crps_by_scale: CRPS values at each temporal scale (in basis points).
        composite_crps: Weighted average CRPS across scales.
        scale_names: Human-readable names for each scale.
        raw_crps_values: Per-observation CRPS before aggregation.
    """

    crps_by_scale: dict[str, float]
    composite_crps: float
    scale_names: list[str]
    raw_crps_values: dict[str, np.ndarray] = field(default_factory=dict)


class SynthScorer:
    """Multi-scale CRPS scorer matching Synth subnet methodology.

    Evaluates probabilistic price path forecasts by computing CRPS at
    multiple temporal scales on basis-point price changes.

    Args:
        scales_minutes: List of temporal scales in minutes for CRPS evaluation.
            Default: [5, 30, 180, 1440] (5-min, 30-min, 3-hour, 24-hour).
        interval_seconds: Time interval between path price points.
            Default: 300 (5 minutes, matching Synth's format).
        scale_weights: Optional weights for compositing across scales.
            Default: equal weighting.
    """

    def __init__(
        self,
        scales_minutes: list[int] | None = None,
        interval_seconds: int = 300,
        scale_weights: list[float] | None = None,
    ) -> None:
        self.scales_minutes = scales_minutes or [5, 30, 180, 1440]
        self.interval_seconds = interval_seconds
        self.scale_weights = scale_weights

    def score_paths(
        self,
        predicted_paths: np.ndarray,
        actual_prices: np.ndarray,
    ) -> SynthScoreResult:
        """Score predicted paths against actual prices at multiple temporal scales.

        Args:
            predicted_paths: Array of shape (n_paths, n_points) with simulated
                price paths.
            actual_prices: Array of shape (n_points,) with actual observed prices,
                aligned to the same time grid as the predicted paths.

        Returns:
            SynthScoreResult with per-scale and composite CRPS.
        """
        paths = np.asarray(predicted_paths, dtype=float)
        actual = np.asarray(actual_prices, dtype=float)

        if paths.ndim == 1:
            paths = paths.reshape(1, -1)

        n_paths, n_points = paths.shape

        crps_by_scale: dict[str, float] = {}
        raw_crps: dict[str, np.ndarray] = {}
        scale_names: list[str] = []

        for scale_min in self.scales_minutes:
            # Convert scale to number of intervals
            steps_per_scale = max(1, int(scale_min * 60 / self.interval_seconds))
            name = self._scale_name(scale_min)
            scale_names.append(name)

            # Compute basis-point price changes at this scale
            # For each evaluation point, compute returns over the scale window
            eval_points = list(range(steps_per_scale, n_points, steps_per_scale))
            if not eval_points:
                crps_by_scale[name] = 0.0
                raw_crps[name] = np.array([0.0])
                continue

            crps_values = []
            for t in eval_points:
                t_start = t - steps_per_scale
                # Basis-point changes: (P_t - P_{t-k}) / P_{t-k} * 10000
                pred_changes = (paths[:, t] - paths[:, t_start]) / paths[:, t_start] * 10000
                actual_change = (actual[t] - actual[t_start]) / actual[t_start] * 10000

                # CRPS = E[|X - y|] - 0.5 * E[|X - X'|]
                mae = float(np.mean(np.abs(pred_changes - actual_change)))
                spread = float(np.mean(np.abs(
                    pred_changes[:, np.newaxis] - pred_changes[np.newaxis, :]
                ))) / 2
                crps_values.append(mae - spread)

            crps_arr = np.array(crps_values)
            raw_crps[name] = crps_arr
            crps_by_scale[name] = float(np.mean(crps_arr))

        # Composite: weighted average across scales
        weights = self.scale_weights or [1.0] * len(self.scales_minutes)
        weights = np.array(weights, dtype=float)
        weights /= weights.sum()

        scale_values = [crps_by_scale.get(self._scale_name(s), 0.0) for s in self.scales_minutes]
        composite = float(np.dot(weights, scale_values))

        return SynthScoreResult(
            crps_by_scale=crps_by_scale,
            composite_crps=composite,
            scale_names=scale_names,
            raw_crps_values=raw_crps,
        )

    @staticmethod
    def percentile_cap(
        scores: np.ndarray,
        percentile: float = 90.0,
    ) -> np.ndarray:
        """Cap worst scores at the given percentile (Synth normalization).

        Scores worse than the percentile threshold are set to the threshold
        value. This prevents outlier forecasters from receiving extreme penalties.

        Args:
            scores: Array of CRPS scores (lower is better).
            percentile: Percentile to cap at (default 90th).

        Returns:
            Capped scores array.
        """
        cap = float(np.percentile(scores, percentile))
        return np.minimum(scores, cap)

    @staticmethod
    def normalize_scores(scores: np.ndarray) -> np.ndarray:
        """Subtract best score so the top performer has score 0.

        Args:
            scores: Array of CRPS scores.

        Returns:
            Normalized scores (best = 0, others > 0).
        """
        return scores - np.min(scores)

    @staticmethod
    def ewma_smooth(
        scores: np.ndarray,
        halflife_days: float = 5.0,
        interval_days: float = 1.0,
    ) -> np.ndarray:
        """Apply EWMA smoothing to a time series of scores.

        Args:
            scores: Array of daily (or per-interval) scores.
            halflife_days: Half-life for exponential decay.
            interval_days: Time between score observations in days.

        Returns:
            Smoothed scores.
        """
        decay = np.log(2) / (halflife_days / interval_days)
        alpha = 1 - np.exp(-decay)
        out = np.empty_like(scores)
        out[0] = scores[0]
        for i in range(1, len(scores)):
            out[i] = alpha * scores[i] + (1 - alpha) * out[i - 1]
        return out

    @staticmethod
    def scores_to_weights(
        scores: np.ndarray,
        beta: float = -0.1,
    ) -> np.ndarray:
        """Convert scores to softmax weights (lower CRPS → higher weight).

        Uses Synth's negative-beta softmax: w_i = exp(beta × score_i) / Σ exp(beta × score_j)

        Args:
            scores: Array of normalized CRPS scores.
            beta: Softmax temperature (negative so lower scores get higher weights).

        Returns:
            Array of weights summing to 1.0.
        """
        exp_scores = np.exp(beta * scores)
        return exp_scores / exp_scores.sum()

    @staticmethod
    def _scale_name(minutes: int) -> str:
        if minutes < 60:
            return f"{minutes}min"
        if minutes < 1440:
            return f"{minutes // 60}h"
        return f"{minutes // 1440}d"
