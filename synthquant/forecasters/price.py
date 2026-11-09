"""One-call price forecasts with the library's default model."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from synthquant.forecasters.base import ReturnForecast, ReturnForecaster
from synthquant.forecasters.gjr import GJRGarchForecaster

__all__ = ["default_forecaster", "forecast_prices", "PriceForecast", "DEFAULT_LOOKBACK"]

#: Trading days of history used to fit the default model.
DEFAULT_LOOKBACK = 504


def default_forecaster() -> ReturnForecaster:
    """The library's default model: GJR-GARCH(1,1) with Hansen skew-t innovations.

    Chosen on out-of-sample evidence: in a daily walk-forward over 2012-2026
    on SPY, QQQ, IWM, TLT and GLD it had the best CRPS at 1, 5 and 21 days
    (about 3.4% better than a rolling-volatility Gaussian) and was the only
    model to pass Kupiec and Christoffersen 1% VaR backtests on most assets.
    It beat the regime-switching and four-engine ensemble forecasters at
    every horizon. See ``examples/31_model_comparison.py``.

    Settings were also checked out of sample (2016-2026, same assets): an
    estimated constant mean helped only the assets that trended up over that
    period (SPY/QQQ/GLD, up to +2.8% CRPS at 21 days) and hurt TLT and IWM,
    so the mean stays at zero; fitting on 4 or 6 years instead of 2 changed
    CRPS by less than 0.3% on average, so ``DEFAULT_LOOKBACK`` stays at 504.

    Calibration notes (2012-2026 walk-forward, five ETFs): one-day forecasts
    are well calibrated. At 21 days the zero mean makes equity forecasts sit
    left of outcomes during a bull market (SPY 5% VaR breached 2.4% of the
    time), which looks like a too-heavy left tail. It is drift, not shape:
    adding each asset's realised drift balances the 5% tails (4.0% / 3.9%),
    while removing the model's negative skew makes CRPS clearly worse
    (realised SPY 21-day skew is -1.0). Variance level and mean-reversion
    speed were also checked and are not the cause. If you hold a drift view
    (e.g. an equity risk premium), pass it explicitly via
    ``forecast_prices(..., annual_drift=...)``.
    """
    return GJRGarchForecaster(dist="skewt", mean="zero")


@dataclass(frozen=True)
class PriceForecast:
    """Forecast distribution of future prices for each of the next ``horizon`` periods.

    Returns are simple (``price / last_price - 1``) in the risk methods below,
    with negative values meaning losses.

    Attributes:
        last_price: Price at the forecast origin.
        as_of: Date of the last observed price (None for array input).
        dates: Forecast dates for horizons ``1..horizon`` (business days by
            default; exchange holidays are not removed), or None.
        returns: Underlying cumulative log-return forecast for horizons
            ``1..horizon``.
        model: Name of the forecaster that produced it.
    """

    last_price: float
    as_of: pd.Timestamp | None
    dates: pd.DatetimeIndex | None
    returns: ReturnForecast
    model: str

    @property
    def horizon(self) -> int:
        """Longest forecast horizon, in periods."""
        return self.returns.horizons[-1]

    @property
    def prices(self) -> np.ndarray:
        """Price samples, shape ``(n_samples, horizon)``; column ``h-1`` is ``h`` periods out."""
        return self.last_price * np.exp(self.returns.samples)

    def _column(self, horizon: int | None) -> np.ndarray:
        h = self.horizon if horizon is None else int(horizon)
        if not 1 <= h <= self.horizon:
            raise ValueError(f"horizon must be in 1..{self.horizon}, got {h}")
        return self.prices[:, h - 1]

    def bands(self, quantiles: Sequence[float] = (0.05, 0.25, 0.5, 0.75, 0.95)) -> pd.DataFrame:
        """Price quantiles per horizon: rows are horizons (or dates), columns quantiles."""
        q = np.asarray(quantiles, dtype=float)
        values = np.quantile(self.prices, q, axis=0).T
        index: pd.Index = (
            self.dates if self.dates is not None else pd.RangeIndex(1, self.horizon + 1)
        )
        return pd.DataFrame(values, index=index, columns=[f"q{100 * x:g}" for x in q])

    def value_at_risk(self, alpha: float = 0.05, horizon: int | None = None) -> float:
        """The ``alpha`` quantile of the simple return at ``horizon`` (default: last)."""
        return float(np.quantile(self._column(horizon) / self.last_price - 1.0, alpha))

    def expected_shortfall(self, alpha: float = 0.05, horizon: int | None = None) -> float:
        """Mean simple return in the worst ``alpha`` fraction of outcomes."""
        r = self._column(horizon) / self.last_price - 1.0
        return float(r[r <= np.quantile(r, alpha)].mean())

    def prob_above(self, level: float, horizon: int | None = None) -> float:
        """Probability that the price ``horizon`` periods out exceeds ``level``."""
        return float(np.mean(self._column(horizon) > level))

    def summary(self, horizons: Sequence[int] | None = None) -> pd.DataFrame:
        """Key numbers per horizon: price bands, 95% VaR / ES and P(up)."""
        if horizons is None:
            horizons = [h for h in (1, 5, 10, 21, 63) if h <= self.horizon] or [self.horizon]
        rows = []
        for h in horizons:
            p = self._column(h)
            rows.append(
                {
                    "horizon": h,
                    "date": self.dates[h - 1] if self.dates is not None else None,
                    "q5": np.quantile(p, 0.05),
                    "median": np.median(p),
                    "q95": np.quantile(p, 0.95),
                    "var_95": self.value_at_risk(0.05, h),
                    "es_95": self.expected_shortfall(0.05, h),
                    "p_up": self.prob_above(self.last_price, h),
                }
            )
        out = pd.DataFrame(rows).set_index("horizon")
        return out.drop(columns="date") if self.dates is None else out


def forecast_prices(
    prices: pd.Series | np.ndarray,
    horizon: int = 21,
    *,
    forecaster: ReturnForecaster | None = None,
    lookback: int = DEFAULT_LOOKBACK,
    n_samples: int = 10_000,
    freq: str = "B",
    annual_drift: float = 0.0,
    periods_per_year: int = 252,
    random_state: int | np.random.Generator | None = None,
) -> PriceForecast:
    """Fit a forecaster to recent prices and forecast the next ``horizon`` periods.

    Example:
        >>> import yfinance as yf
        >>> from synthquant import forecast_prices
        >>> close = yf.download("SPY", period="5y", auto_adjust=True)["Close"].squeeze()
        >>> fc = forecast_prices(close, horizon=21)
        >>> fc.summary()            # bands, VaR / ES and P(up) at 1, 5, 10, 21 days
        >>> fc.bands()              # daily price quantiles for a fan chart

    Args:
        prices: Price history, oldest first (a Series with a DatetimeIndex
            also labels the forecast dates).
        horizon: Number of periods to forecast.
        forecaster: Model to use; defaults to :func:`default_forecaster`.
        lookback: Number of most recent returns used to fit the model.
        n_samples: Number of simulated outcomes.
        freq: Pandas frequency for forecast dates (``"B"`` business days for
            equities, ``"D"`` for 24/7 markets such as crypto).
        annual_drift: Assumed annual drift of the *log* price, added on top of
            the model's forecast (``annual_drift * h / periods_per_year`` at
            horizon ``h``). The default 0 makes no directional call; set it to
            express a view such as an equity risk premium (e.g. 0.05). It is
            deliberately not estimated: trailing means were tested out of
            sample and helped only assets that happened to trend.
        periods_per_year: Periods per year for ``annual_drift`` (252 for
            equities, 365 for 24/7 markets).
        random_state: Seed or Generator.

    Returns:
        :class:`PriceForecast`.
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    if isinstance(prices, pd.Series):
        p = prices.dropna()
        values = p.to_numpy(dtype=float)
        as_of = p.index[-1] if isinstance(p.index, pd.DatetimeIndex) else None
    else:
        values = np.asarray(prices, dtype=float).ravel()
        values = values[np.isfinite(values)]
        as_of = None
    if np.any(values <= 0):
        raise ValueError("prices must be positive")
    returns = np.diff(np.log(values))[-lookback:]
    if returns.size < 100:
        raise ValueError(f"Need at least 101 prices, got {values.size}")

    model = forecaster if forecaster is not None else default_forecaster()
    model.fit(returns)
    fc = model.predict(
        returns, range(1, horizon + 1), n_samples=n_samples, random_state=random_state
    )
    if annual_drift:
        steps = np.arange(1, horizon + 1, dtype=float)
        shift = annual_drift / periods_per_year * steps
        fc = ReturnForecast(fc.horizons, fc.samples + shift[np.newaxis, :])
    dates = None
    if as_of is not None:
        step = pd.tseries.frequencies.to_offset(freq)
        dates = pd.date_range(as_of.normalize() + step, periods=horizon, freq=freq)
    return PriceForecast(
        last_price=float(values[-1]), as_of=as_of, dates=dates, returns=fc, model=model.name
    )
