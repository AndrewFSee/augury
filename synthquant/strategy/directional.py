"""Directional signal generators.

Volatility forecasts describe *how wide* the future return distribution
is, not *which way* it will move. For a signal that can actually beat a
long-only benchmark, you need a directional input. This module provides
classic directional signals well-suited for ETFs:

* :class:`TimeSeriesMomentum` — 12-1 month time-series momentum, one of
  the most robust retail-accessible edges. Signal at time *t* is
  ``sign(return[t - lookback_long : t - skip])``. The 1-month skip avoids
  short-term reversal contamination.
* :class:`CrossSectionalMomentum` — rank-based momentum across an asset
  universe. Long top-N, flat (or short) bottom-N.

These are *inputs* to
:class:`~synthquant.strategy.forecast_signals.ForecastSignal.generate_decomposed`
which provides the size from a forecast's uncertainty properties.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["TimeSeriesMomentum", "CrossSectionalMomentum", "DirectionalSignal"]


@dataclass(frozen=True)
class DirectionalSignal:
    """Output of a directional signal generator.

    Attributes:
        direction: ±1 or 0 indicating long / short / flat.
        strength: Underlying strength in [-1, +1] (pre-thresholding).
    """

    direction: int
    strength: float


class TimeSeriesMomentum:
    """Time-series momentum with a one-period skip (classic 12-1 momentum).

    At each evaluation point, computes cumulative log returns over the
    window ``[t - lookback_long : t - skip]`` and signs it. The ``skip``
    parameter removes the most-recent ``skip`` periods from the window,
    avoiding the short-term reversal effect that can contaminate pure
    12-month momentum.

    Args:
        lookback_long: Total lookback in periods (default 252 ≈ 12 months).
        skip: Periods to skip at the end of the window (default 21 ≈ 1 month).
        allow_short: If False, negative momentum returns direction=0
            instead of -1.
        threshold: Absolute cumulative return below which direction is 0.
            Defaults to 0 (sign only).
    """

    def __init__(
        self,
        lookback_long: int = 252,
        skip: int = 21,
        allow_short: bool = False,
        threshold: float = 0.0,
    ) -> None:
        if skip >= lookback_long:
            raise ValueError("skip must be strictly less than lookback_long")
        self.lookback_long = int(lookback_long)
        self.skip = int(skip)
        self.allow_short = bool(allow_short)
        self.threshold = float(threshold)

    def generate(self, returns: np.ndarray) -> DirectionalSignal:
        """Produce a directional signal from a return series.

        Args:
            returns: 1-D array of daily log returns, most recent last.

        Returns:
            DirectionalSignal with direction in {-1, 0, +1}. If there is
            not enough history, direction is 0.
        """
        r = np.asarray(returns, dtype=float)
        if len(r) < self.lookback_long:
            return DirectionalSignal(direction=0, strength=0.0)

        end = len(r) - self.skip if self.skip > 0 else len(r)
        window = r[-self.lookback_long : end]
        cum = float(np.sum(window))

        if cum > self.threshold:
            direction = 1
        elif cum < -self.threshold and self.allow_short:
            direction = -1
        else:
            direction = 0

        return DirectionalSignal(direction=direction, strength=cum)


class CrossSectionalMomentum:
    """Cross-sectional rank-based momentum across an asset universe.

    Ranks assets by cumulative return over ``[t - lookback : t - skip]``
    and assigns long exposure to the top ``top_n``, flat to the middle,
    and (optionally) short to the bottom ``bottom_n``.

    Args:
        lookback: Total lookback in periods (default 252).
        skip: Periods to skip at the end of the window (default 21).
        top_n: Number of top-ranked assets to go long.
        bottom_n: Number of bottom-ranked assets to go short (0 = long-only).
    """

    def __init__(
        self,
        lookback: int = 252,
        skip: int = 21,
        top_n: int = 2,
        bottom_n: int = 0,
    ) -> None:
        if skip >= lookback:
            raise ValueError("skip must be strictly less than lookback")
        self.lookback = int(lookback)
        self.skip = int(skip)
        self.top_n = int(top_n)
        self.bottom_n = int(bottom_n)

    def generate(self, returns_matrix: np.ndarray) -> np.ndarray:
        """Produce a direction vector across assets.

        Args:
            returns_matrix: Array of shape (n_periods, n_assets). Most
                recent period last.

        Returns:
            Direction vector of length ``n_assets`` with entries in
            {-1, 0, +1}. If history is insufficient, all entries are 0.
        """
        r = np.asarray(returns_matrix, dtype=float)
        if r.ndim != 2:
            raise ValueError("returns_matrix must be 2-D (periods × assets)")
        n_periods, n_assets = r.shape
        directions = np.zeros(n_assets, dtype=int)

        if n_periods < self.lookback:
            return directions

        end = n_periods - self.skip if self.skip > 0 else n_periods
        start = end - self.lookback
        window = r[start:end]
        cum = window.sum(axis=0)

        order = np.argsort(cum)
        if self.bottom_n > 0:
            directions[order[: self.bottom_n]] = -1
        if self.top_n > 0:
            directions[order[-self.top_n :]] = 1
        return directions
