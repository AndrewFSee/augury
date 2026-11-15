"""Known event days (company earnings) as extra variance on top of any forecaster."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
import pandas as pd

from augury.data.events import trading_days_after
from augury.forecasters.base import ReturnForecast, ReturnForecaster
from augury.forecasters.gjr import GJRGarchForecaster

__all__ = ["EventForecaster", "event_multiplier"]


def event_multiplier(
    returns: pd.Series,
    events: pd.DatetimeIndex,
    *,
    halflife: float = 16.0,
    prior_events: float = 5.0,
    pre_window: int = 20,
) -> float:
    """How many times an ordinary day's variance an event day carries, from past events.

    Each event's squared return is divided by the mean squared return of the
    ``pre_window`` non-event sessions before it. Because the mean of r^2 over a noisy
    variance estimate is above 1 even on ordinary days, the ratios are divided by the
    same statistic on non-event days (last 504). Events are weighted exponentially
    (``halflife`` in events, latest weighs most) and shrunk toward 1 with
    ``prior_events`` pseudo-events. Never below 1.

    Args:
        returns: Dated log returns; only these are used (pass data up to the fit date).
        events: Sessions whose return carries an event.
    """
    r = returns.dropna()
    is_event = r.index.isin(events)
    quiet2 = (r.where(~is_event) ** 2).to_numpy()
    x2 = (r**2).to_numpy()
    ratios = []
    for p in np.flatnonzero(is_event):
        pre = quiet2[max(0, p - 3 * pre_window) : p]
        pre = pre[np.isfinite(pre)][-pre_window:]
        if pre.size >= pre_window // 2 and pre.mean() > 0:
            ratios.append(x2[p] / pre.mean())
    if not ratios:
        return 1.0
    q = pd.Series(quiet2).dropna()
    qr = (q / q.rolling(pre_window).mean().shift(1)).dropna().to_numpy()[-504:]
    bias = float(qr.mean()) if qr.size >= 100 else 1.0
    ratios_arr = np.asarray(ratios) / bias
    w = 0.5 ** (np.arange(ratios_arr.size)[::-1] / halflife)
    return max(1.0, float((w @ ratios_arr + prior_events) / (w.sum() + prior_events)))


class EventForecaster(ReturnForecaster):
    """Wrap a forecaster so known event days (earnings) get their extra variance.

    At each :meth:`fit` two things are estimated from past events: the multiplier ``m``
    (see :func:`event_multiplier`) and the event-day variance itself, an exponentially
    weighted mean of past event-day squared returns (``halflife`` in events). Event-day
    returns are divided by ``sqrt(m)`` before the base model sees them, so a jump does
    not read as a lasting volatility shock. Each future event day inside the horizon
    adds an independent normal move that lifts that day's variance to the event-day
    variance (``size="absolute"``, the default; never below the ordinary day's), or to
    ``m`` times the base forecast's average daily variance (``size="proportional"``).

    Earnings moves are about the same size whatever the volatility level (mean absolute
    move 4.5% / 5.5% / 5.7% on earnings days in calm / middle / volatile thirds), so
    scaling them with current volatility made forecasts too narrow in calm markets and
    too wide in volatile ones; the absolute size fixes that.

    Evidence (walk-forward 2016-2026, augury's default model on AAPL, MSFT, NVDA, AMZN,
    META, GOOGL, TSLA, JPM; earnings dates from Yahoo), absolute size: on earnings
    sessions the 1-day CRPS improved 15.3% and the 1% VaR breach rate fell from 16.5% to
    2.3% (90% band coverage 37% -> 90%); over all days CRPS -1.08% at 1 day, -1.08% at
    5 days, -0.31% at 21 days, every stock better, both 2016-20 and 2021-26
    (proportional: -1.03%, -0.96%, -0.23%). The same overlay for CPI, jobs, PPI and FOMC
    days did not help (index ETFs ~0, bond / credit ETFs -0.3%, equities slightly worse),
    so it is not used there.

    Args:
        events: Sessions whose close-to-close return carries an event (see
            :func:`augury.data.events.event_sessions`).
        base: Forecaster to adjust (default: GJR-GARCH skew-t, zero mean).
        history: Optional longer dated return history for estimating ``m``; only
            dates up to each fit window's end are used. Without it, the fit window.
        halflife, prior_events, pre_window: See :func:`event_multiplier`.
        size: ``"absolute"`` (event-day variance from past event moves; falls back to
            proportional with fewer than 4 past events) or ``"proportional"``.
        clean: Divide event-day returns by ``sqrt(m)`` before the base model sees them.
        name: Label for evaluation tables.
    """

    needs_index = True

    def __init__(
        self,
        events: Sequence[pd.Timestamp] | pd.DatetimeIndex,
        base: ReturnForecaster | None = None,
        *,
        history: pd.Series | None = None,
        halflife: float = 16.0,
        prior_events: float = 5.0,
        pre_window: int = 20,
        size: Literal["absolute", "proportional"] = "absolute",
        clean: bool = True,
        name: str | None = None,
    ) -> None:
        if size not in ("absolute", "proportional"):
            raise ValueError("size must be 'absolute' or 'proportional'")
        self.events = pd.DatetimeIndex(list(events)).normalize().unique().sort_values()
        self.base = base if base is not None else GJRGarchForecaster(dist="skewt", mean="zero")
        self.history = (
            None if history is None else pd.Series(history, dtype=float).dropna().sort_index()
        )
        self.halflife, self.prior_events, self.pre_window = halflife, prior_events, pre_window
        self.size = size
        self.clean = clean
        self.name = name or f"{self.base.name}+events"
        self.multiplier = 1.0
        self.event_variance: float | None = None
        self.last_event_days: list[int] = []

    def _base_input(self, x: pd.Series) -> Any:
        return x if getattr(self.base, "needs_index", False) else x.to_numpy(dtype=float)

    def _cleaned(self, x: pd.Series) -> pd.Series:
        if not self.clean or self.multiplier <= 1.0:
            return x
        hit = x.index.isin(self.events)
        return x.where(~hit, x / np.sqrt(self.multiplier))

    def fit(self, returns: np.ndarray | pd.Series) -> EventForecaster:
        if not isinstance(returns, pd.Series) or not isinstance(returns.index, pd.DatetimeIndex):
            raise TypeError("EventForecaster needs returns as a pd.Series with a DatetimeIndex")
        asof = returns.index[-1]
        src = returns if self.history is None else self.history.loc[:asof]
        self.multiplier = event_multiplier(
            src,
            self.events,
            halflife=self.halflife,
            prior_events=self.prior_events,
            pre_window=self.pre_window,
        )
        past = (src[src.index.isin(self.events)] ** 2).to_numpy()
        self.event_variance = None
        if self.size == "absolute" and past.size >= 4:
            w = 0.5 ** (np.arange(past.size)[::-1] / self.halflife)
            self.event_variance = float(w @ past / w.sum())
        self.base.fit(self._base_input(self._cleaned(returns)))
        return self

    def predict(
        self,
        history: np.ndarray | pd.Series,
        horizons: Sequence[int],
        n_samples: int = 2000,
        random_state: int | np.random.Generator | None = None,
    ) -> ReturnForecast:
        if not isinstance(history, pd.Series) or not isinstance(history.index, pd.DatetimeIndex):
            raise TypeError("EventForecaster needs history as a pd.Series with a DatetimeIndex")
        rng = np.random.default_rng(random_state)
        fc = self.base.predict(self._base_input(self._cleaned(history)), horizons, n_samples, rng)
        hs = list(fc.horizons)
        future = trading_days_after(history.index[-1], max(hs))
        days = [k for k, d in enumerate(future, start=1) if d in self.events]
        self.last_event_days = days
        if not days or (self.multiplier <= 1.0 and self.event_variance is None):
            return fc
        daily = {h: float(fc.at(h).var()) / h for h in hs}
        add = np.zeros_like(fc.samples)
        for k in days:
            v = daily[next(h for h in hs if h >= k)]
            target = self.event_variance if self.event_variance is not None else self.multiplier * v
            extra = max(0.0, target - v)
            jump = np.sqrt(extra) * rng.standard_normal(fc.samples.shape[0])
            add[:, [j for j, h in enumerate(hs) if h >= k]] += jump[:, None]
        return ReturnForecast(fc.horizons, fc.samples + add)
