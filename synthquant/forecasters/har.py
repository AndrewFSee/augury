"""HAR volatility forecaster with empirically calibrated return quantiles."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from synthquant.forecasters.base import (
    ReturnForecast,
    ReturnForecaster,
    quantile_grid,
    validate_horizons,
)

__all__ = ["HARForecaster"]


class HARForecaster(ReturnForecaster):
    """Corsi (2009) HAR variance forecast with an empirical standardized distribution.

    For each horizon ``h`` a direct HAR regression predicts the average daily
    variance over the next ``h`` days from the daily, weekly (5d) and monthly
    (22d) averages of squared returns. The distribution of the h-day return
    divided by its predicted standard deviation is taken from the training
    window, so the forecast is calibrated per horizon without assuming a
    parametric shape.

    With only daily closes, squared returns are a noisy proxy for realized
    variance; feeding intraday realized variance would sharpen the forecasts.

    Args:
        weekly: Window of the weekly component.
        monthly: Window of the monthly component.
        floor_frac: Predicted variance is floored at this fraction of the
            training window's mean squared return.
        center: Subtract the mean of the standardized returns (drift) so the
            forecast is centred on zero, consistent with the other models.
    """

    name = "har"

    def __init__(
        self,
        weekly: int = 5,
        monthly: int = 22,
        floor_frac: float = 0.05,
        center: bool = True,
    ) -> None:
        if not 1 < weekly < monthly:
            raise ValueError("Require 1 < weekly < monthly")
        self.weekly = weekly
        self.monthly = monthly
        self.floor_frac = floor_frac
        self.center = center
        self._train: np.ndarray | None = None
        self._by_h: dict[int, tuple[np.ndarray, np.ndarray, float]] = {}

    def fit(self, returns: np.ndarray) -> HARForecaster:
        r = np.asarray(returns, dtype=float).ravel()
        self._train = r[np.isfinite(r)]
        self._by_h = {}
        return self

    def _features(self, rv: np.ndarray) -> np.ndarray:
        """HAR regressors for every origin t >= monthly - 1: [1, d, w, m]."""
        c = np.concatenate([[0.0], np.cumsum(rv)])
        t = np.arange(self.monthly - 1, rv.size)
        d = rv[t]
        w = (c[t + 1] - c[t + 1 - self.weekly]) / self.weekly
        m = (c[t + 1] - c[t + 1 - self.monthly]) / self.monthly
        return np.column_stack([np.ones_like(d), d, w, m])

    def _fit_horizon(self, h: int) -> tuple[np.ndarray, np.ndarray, float]:
        if h in self._by_h:
            return self._by_h[h]
        if self._train is None:
            raise RuntimeError("HARForecaster must be fit() before predict()")
        r = self._train
        rv = r**2
        n_origins = r.size - self.monthly - h + 1
        if n_origins < 50:
            raise ValueError(f"Training window too short for horizon {h}")
        X = self._features(rv)[:n_origins]
        t = np.arange(self.monthly - 1, self.monthly - 1 + n_origins)
        crv = np.concatenate([[0.0], np.cumsum(rv)])
        cr = np.concatenate([[0.0], np.cumsum(r)])
        y = (crv[t + 1 + h] - crv[t + 1]) / h
        realized = cr[t + 1 + h] - cr[t + 1]
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        floor = self.floor_frac * float(np.mean(rv))
        pred = np.maximum(X @ beta, floor)
        z = realized / np.sqrt(h * pred)
        if self.center:
            z = z - z.mean()
        fitted = (beta, np.sort(z), floor)
        self._by_h[h] = fitted
        return fitted

    def variance_forecast(self, history: np.ndarray, horizon: int) -> float:
        """Predicted average daily variance over the next ``horizon`` periods."""
        beta, _, floor = self._fit_horizon(horizon)
        r = np.asarray(history, dtype=float).ravel()
        r = r[np.isfinite(r)]
        if r.size < self.monthly:
            raise ValueError(f"history must contain at least {self.monthly} returns")
        x = self._features(r[-self.monthly :] ** 2)[-1]
        return max(float(x @ beta), floor)

    def predict(
        self,
        history: np.ndarray,
        horizons: Sequence[int],
        n_samples: int = 2000,
        random_state: int | np.random.Generator | None = None,
    ) -> ReturnForecast:
        hs = validate_horizons(horizons)
        grid = quantile_grid(n_samples)
        out = np.empty((n_samples, len(hs)))
        for j, h in enumerate(hs):
            _, z, _ = self._fit_horizon(h)
            probs = (np.arange(z.size) + 0.5) / z.size
            out[:, j] = np.sqrt(h * self.variance_forecast(history, h)) * np.interp(grid, probs, z)
        return ReturnForecast(hs, out)
