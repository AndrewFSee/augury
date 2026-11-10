"""Forecasters that combine GJR-GARCH with an external volatility view."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
import pandas as pd

from augury.forecasters.base import ReturnForecast, ReturnForecaster
from augury.forecasters.gjr import GJRGarchForecaster

__all__ = ["ImpliedVolForecaster"]


def _as_dated(history: np.ndarray | pd.Series, what: str) -> pd.Series:
    if not isinstance(history, pd.Series) or not isinstance(history.index, pd.DatetimeIndex):
        raise TypeError(
            f"ImpliedVolForecaster.{what} needs returns as a pd.Series with a DatetimeIndex "
            "to look up the implied vol on each date"
        )
    return history


class ImpliedVolForecaster(ReturnForecaster):
    """GJR-GARCH whose starting volatility is pulled toward an implied-vol index.

    At each forecast origin the model's expected variance over
    ``iv_horizon`` periods is blended with the variance implied by the index
    on that date::

        target = (1 - weight) * GJR variance + weight * k * iv**2 * iv_horizon / periods_per_year

    and the forecast's variance is scaled by ``target / GJR variance`` at
    every horizon (``scaling="proportional"``, the default). The fitted
    dynamics, leverage effect and skew-t shape are unchanged.

    Implied vol is usually above the volatility that follows (the variance
    risk premium). With ``calibrate=True``, ``k`` is estimated at each
    :meth:`fit` as realized
    ``iv_horizon``-period variance divided by implied variance over the
    training window, using only pairs whose realized window ends inside it.
    That also absorbs units: MOVE (bond-yield vol in basis points) can be
    passed for a Treasury ETF without converting it to price vol.

    Evidence (walk-forward 2012-2026, VIX for SPY and IWM, VXN for QQQ, GVZ
    for GLD): ``weight=0.5`` at face value improved CRPS over plain GJR by
    0.3-0.6% at 1, 5 and 21 days on all four (Diebold-Mariano p < 0.02 in
    10 of 12 cases) with every 1% VaR backtest Basel green. ``weight=1``
    was worse than GJR. On TLT with MOVE, calibrated, the gain was ~0.

    It needs dated returns: pass a ``pd.Series`` with a DatetimeIndex to
    :func:`~augury.evaluation.walk_forward` or
    :func:`~augury.forecasters.forecast_prices`.

    Args:
        implied_vol: Implied-vol index by date, annualized, e.g. ``VIX / 100``.
            The value on or before each origin's date is used.
        base: GJR forecaster to adjust (default: skew-t, zero mean).
        weight: Weight on the implied-vol variance, in ``[0, 1]``.
        iv_horizon: Horizon the index refers to, in periods (VIX: 30
            calendar days, about 21 trading days).
        scaling: How the target is applied. ``"proportional"`` multiplies the
            whole variance path by ``target / GJR variance``, so every horizon
            moves by the same factor. ``"start"`` rescales only today's
            variance and lets GJR mean-revert, which concentrates the change
            in the first few days.
        calibrate: Estimate ``k`` on each training window instead of using
            the index at face value (``k = 1``, the default). Needed when the
            index is not in price-vol units (MOVE for TLT). For VIX, VXN and
            GVZ the face value scored better out of sample.
        periods_per_year: Periods per year for annualized ``implied_vol``.
        name: Label for evaluation tables.
    """

    needs_index = True

    def __init__(
        self,
        implied_vol: pd.Series,
        *,
        base: GJRGarchForecaster | None = None,
        weight: float = 0.5,
        iv_horizon: int = 21,
        calibrate: bool = False,
        scaling: Literal["proportional", "start"] = "proportional",
        periods_per_year: int = 252,
        name: str | None = None,
    ) -> None:
        if not 0.0 <= weight <= 1.0:
            raise ValueError("weight must be in [0, 1]")
        if iv_horizon < 1:
            raise ValueError("iv_horizon must be >= 1")
        iv = pd.Series(implied_vol, dtype=float).dropna()
        if not isinstance(iv.index, pd.DatetimeIndex):
            raise TypeError("implied_vol must be indexed by date")
        if (iv <= 0).any():
            raise ValueError("implied_vol must be positive")
        self.implied_vol = iv.sort_index()
        self.base = base if base is not None else GJRGarchForecaster(dist="skewt")
        self.weight = float(weight)
        self.iv_horizon = int(iv_horizon)
        if scaling not in ("proportional", "start"):
            raise ValueError("scaling must be 'proportional' or 'start'")
        self.calibrate = calibrate
        self.scaling = scaling
        self.periods_per_year = periods_per_year
        self.name = name or f"{self.base.name}+iv{weight:g}"
        self.k = 1.0
        self.last_scale = 1.0

    def _iv_on(self, dates: pd.Index) -> np.ndarray:
        """Implied vol known at each date (last value on or before it)."""
        pos = self.implied_vol.index.searchsorted(dates, side="right") - 1
        vals = self.implied_vol.to_numpy()
        out = np.full(len(dates), np.nan)
        ok = pos >= 0
        out[ok] = vals[pos[ok]]
        return out

    def fit(self, returns: np.ndarray | pd.Series) -> ImpliedVolForecaster:
        r = _as_dated(returns, "fit")
        self.base.fit(r.to_numpy(dtype=float))
        if self.calibrate:
            h = self.iv_horizon
            x = r.to_numpy(dtype=float)
            iv = self._iv_on(r.index)
            c = np.concatenate([[0.0], np.cumsum(x**2)])
            n = x.size
            # IV on date t vs realized variance of returns t+1 .. t+h
            t = np.arange(n - h)
            rv = c[t + 1 + h] - c[t + 1]
            ivar = iv[t] ** 2 * h / self.periods_per_year
            ok = np.isfinite(ivar)
            self.k = float(rv[ok].sum() / ivar[ok].sum()) if ok.sum() >= 21 else 1.0
        return self

    def predict(
        self,
        history: np.ndarray | pd.Series,
        horizons: Sequence[int],
        n_samples: int = 2000,
        random_state: int | np.random.Generator | None = None,
    ) -> ReturnForecast:
        h_series = _as_dated(history, "predict")
        x = h_series.to_numpy(dtype=float)
        iv = float(self._iv_on(h_series.index[-1:])[0])
        ratio = 1.0
        if np.isfinite(iv) and self.weight > 0:
            garch = float(self.base.expected_variance(x, self.iv_horizon).sum())
            implied = self.k * iv**2 * self.iv_horizon / self.periods_per_year
            target = (1.0 - self.weight) * garch + self.weight * implied
            ratio = target / garch
        if self.scaling == "start" and ratio != 1.0:
            target = ratio * float(self.base.expected_variance(x, self.iv_horizon).sum())
            scale = self.base.variance_scale_for(x, self.iv_horizon, target)
            self.last_scale = scale
            return self.base.predict(x, horizons, n_samples, random_state, variance_scale=scale)
        self.last_scale = ratio
        fc = self.base.predict(x, horizons, n_samples, random_state)
        return ReturnForecast(fc.horizons, fc.samples * np.sqrt(ratio))
