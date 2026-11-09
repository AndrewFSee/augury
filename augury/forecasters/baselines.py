"""Simple distributional forecasters: the bar every complex model must clear."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
from scipy import stats
from scipy.signal import lfilter

from augury.forecasters.base import (
    ReturnForecast,
    ReturnForecaster,
    quantile_grid,
    validate_horizons,
)

__all__ = ["HistoricalSimulation", "GaussianForecaster", "LaplaceForecaster"]


def _window(history: np.ndarray, window: int | None) -> np.ndarray:
    r = np.asarray(history, dtype=float).ravel()
    r = r[np.isfinite(r)]
    if window is not None:
        r = r[-window:]
    if r.size < 2:
        raise ValueError("Need at least 2 finite returns in history")
    return r


def _sqrt_time_samples(
    ppf_grid: np.ndarray, scale: float, loc: float, horizons: tuple[int, ...]
) -> np.ndarray:
    """Per-horizon samples ``loc*h + scale*sqrt(h)*z`` from a standard quantile grid."""
    h = np.asarray(horizons, dtype=float)
    return loc * h[np.newaxis, :] + scale * np.sqrt(h)[np.newaxis, :] * ppf_grid[:, np.newaxis]


class HistoricalSimulation(ReturnForecaster):
    """Bootstrap of trailing returns (iid), summed to each horizon.

    Args:
        window: Number of trailing returns to resample (None = all history).
    """

    name = "historical"

    def __init__(self, window: int | None = 504) -> None:
        self.window = window

    def predict(
        self,
        history: np.ndarray,
        horizons: Sequence[int],
        n_samples: int = 2000,
        random_state: int | np.random.Generator | None = None,
    ) -> ReturnForecast:
        hs = validate_horizons(horizons)
        r = _window(history, self.window)
        rng = np.random.default_rng(random_state)
        draws = r[rng.integers(0, r.size, size=(n_samples, hs[-1]))]
        cum = np.cumsum(draws, axis=1)
        return ReturnForecast(hs, cum[:, [h - 1 for h in hs]])


class GaussianForecaster(ReturnForecaster):
    """Normal returns with rolling or EWMA (RiskMetrics) volatility, sqrt-time scaled.

    Args:
        window: Trailing window for the volatility estimate (None = all history).
        ewma_lambda: If set, use an exponentially weighted variance with this
            decay (0.94 is the RiskMetrics daily value) instead of an equally
            weighted one.
        mean: ``"zero"`` (default; trailing means are mostly noise) or
            ``"sample"`` to use the window's mean return.
    """

    def __init__(
        self,
        window: int | None = 504,
        ewma_lambda: float | None = None,
        mean: Literal["zero", "sample"] = "zero",
    ) -> None:
        if ewma_lambda is not None and not 0.0 < ewma_lambda < 1.0:
            raise ValueError("ewma_lambda must be in (0, 1)")
        self.window = window
        self.ewma_lambda = ewma_lambda
        self.mean = mean
        self.name = "ewma" if ewma_lambda is not None else "gaussian"

    def volatility(self, history: np.ndarray) -> float:
        """Current one-period volatility estimate."""
        r = _window(history, self.window)
        if self.ewma_lambda is None:
            return float(np.std(r, ddof=1))
        lam = self.ewma_lambda
        seed = min(r.size, 20)
        s0 = float(np.mean(r[:seed] ** 2))
        # s_t = lam * s_{t-1} + (1 - lam) * r_t^2
        s = lfilter([1.0 - lam], [1.0, -lam], r**2, zi=[lam * s0])[0]
        return float(np.sqrt(s[-1]))

    def predict(
        self,
        history: np.ndarray,
        horizons: Sequence[int],
        n_samples: int = 2000,
        random_state: int | np.random.Generator | None = None,
    ) -> ReturnForecast:
        hs = validate_horizons(horizons)
        sigma = self.volatility(history)
        mu = float(np.mean(_window(history, self.window))) if self.mean == "sample" else 0.0
        z = stats.norm.ppf(quantile_grid(n_samples))
        return ReturnForecast(hs, _sqrt_time_samples(z, sigma, mu, hs))


class LaplaceForecaster(ReturnForecaster):
    """MLE Laplace distribution of trailing returns, sqrt-time scaled.

    The model that ranked 1st of 190 on all nine assets in the CrunchDAO Synth
    24h backtest. The MLE of the Laplace scale is the mean absolute deviation
    about the location, so the fit is closed-form and robust to outliers.

    The scale has no volatility dynamics, so the window length *is* the model:
    the competition version used one week of data. On daily SPY 2012-2026 a
    21-day window cut CRPS by ~2.4 points of skill versus 504 days, but VaR
    violations still cluster; a GJR-GARCH model remains clearly better.

    Args:
        window: Trailing window (None = all history). Default 21 trading days.
        loc: ``"zero"`` (default, as in the competition model) or ``"median"``
            (the location MLE).
    """

    name = "laplace"

    def __init__(self, window: int | None = 21, loc: Literal["zero", "median"] = "zero") -> None:
        self.window = window
        self.loc = loc

    def fit_params(self, history: np.ndarray) -> tuple[float, float]:
        """Return the MLE ``(loc, scale)`` for the trailing window."""
        r = _window(history, self.window)
        loc = float(np.median(r)) if self.loc == "median" else 0.0
        scale = max(float(np.mean(np.abs(r - loc))), 1e-12)
        return loc, scale

    def predict(
        self,
        history: np.ndarray,
        horizons: Sequence[int],
        n_samples: int = 2000,
        random_state: int | np.random.Generator | None = None,
    ) -> ReturnForecast:
        hs = validate_horizons(horizons)
        loc, scale = self.fit_params(history)
        z = stats.laplace.ppf(quantile_grid(n_samples))
        return ReturnForecast(hs, _sqrt_time_samples(z, scale, loc, hs))
