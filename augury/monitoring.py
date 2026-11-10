"""Daily risk monitoring: today's forecast and a running out-of-sample track record.

Used by the dashboard (``dashboard/app.py``) and by the hub publisher
(``scripts/publish_to_hub.py``), so both show the same numbers.

The model is the library default (GJR-GARCH skew-t, zero mean). Assets with a
matching implied-vol index get the implied-vol view on top
(:class:`~augury.forecasters.ImpliedVolForecaster`, weight 0.5), but only the
five assets it was validated on: in a 2012-2026 walk-forward it improved CRPS
by 0.3-0.6% on SPY, QQQ, IWM and GLD with every 1% VaR backtest in the Basel
green zone. TLT uses MOVE (a Treasury vol index, not TLT's own), so its view
is calibrated to TLT's realised variance first. Every other asset uses
plain GJR.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from augury.evaluation.walk_forward import WalkForwardResult, walk_forward
from augury.forecasters.base import ReturnForecaster
from augury.forecasters.price import (
    DEFAULT_LOOKBACK,
    PriceForecast,
    default_forecaster,
    forecast_prices,
)
from augury.forecasters.views import ImpliedVolForecaster

__all__ = [
    "IV_VIEWS",
    "last_final_session",
    "HORIZONS",
    "monitor_model",
    "live_forecast",
    "risk_table",
    "track_record",
    "track_models",
    "score_track",
]

#: Ticker -> (implied-vol index symbol, calibrate the index to the ticker's realised variance).
IV_VIEWS: dict[str, tuple[str, bool]] = {
    "SPY": ("^VIX", False),
    "QQQ": ("^VXN", False),
    "IWM": ("^VIX", False),
    "TLT": ("^MOVE", True),
    "GLD": ("^GVZ", False),
}

#: Horizons (trading days) shown and scored.
HORIZONS: tuple[int, ...] = (1, 5, 21)
VAR_LEVELS: tuple[float, ...] = (0.01, 0.05)
INTERVAL_LEVELS: tuple[float, ...] = (0.5, 0.9)
_REFIT = 21
_TRACK_COLUMNS = ["origin", "model", "horizon", "y", "crps", "pit", "std",
                  "var_1", "hit_1", "var_5", "hit_5", "cov_50", "width_50", "cov_90", "width_90"]


def last_final_session(now: datetime | None = None) -> date:
    """Latest date whose daily bar is final: today only after 4:15 PM New York time.

    Daily downloads during the session include a partial bar for today; forecasting from it
    would treat an intraday price as a close.
    """
    now = now or datetime.now(ZoneInfo("America/New_York"))
    d = now.date()
    if now.weekday() < 5 and (now.hour, now.minute) >= (16, 15):
        return d
    d -= timedelta(days=1)
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d


def monitor_model(ticker: str, implied_vol: pd.Series | None = None) -> ReturnForecaster:
    """The model the monitor uses for ``ticker`` (a fresh, unfitted instance).

    Args:
        ticker: Asset symbol.
        implied_vol: The asset's implied-vol index as a decimal (0.20 = 20%),
            dated. Only used for tickers in :data:`IV_VIEWS`.
    """
    view = IV_VIEWS.get(ticker.upper())
    if view is not None and implied_vol is not None and not implied_vol.dropna().empty:
        return ImpliedVolForecaster(implied_vol.dropna(), weight=0.5, calibrate=view[1])
    return default_forecaster()


def live_forecast(
    prices: pd.Series,
    model: ReturnForecaster,
    *,
    horizon: int = 21,
    n_samples: int = 20_000,
    random_state: int | None = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Today's forecast: key risk numbers per horizon and daily price bands.

    Returns are simple returns (``price / last - 1``); VaR and expected
    shortfall are negative numbers (losses).

    Args:
        prices: Daily closes with a DatetimeIndex, oldest first.
        model: Forecaster, e.g. from :func:`monitor_model`.
        horizon: Longest horizon in trading days.
        n_samples: Simulated outcomes.
        random_state: Seed.

    Returns:
        ``(risk, bands)``. ``risk`` has one row per horizon in
        :data:`HORIZONS` (up to ``horizon``) with ``as_of``, ``last_price``,
        ``vol_ann`` (forecast volatility of the h-day log return, annualised),
        ``var_99``, ``var_95``, ``es_99``, ``es_95``, ``lo_90`` / ``hi_90``
        (price interval), ``median`` and ``p_up``. ``bands`` has price quantiles
        ``q5 .. q95`` for each day ``1..horizon``, indexed by forecast date.
    """
    fc = forecast_prices(prices, horizon=horizon, forecaster=model,
                         n_samples=n_samples, random_state=random_state)
    bands = fc.bands((0.05, 0.25, 0.5, 0.75, 0.95))
    bands.index.name = "date"
    return risk_table(fc, [h for h in HORIZONS if h <= horizon]), bands


def risk_table(fc: PriceForecast, horizons: Sequence[int]) -> pd.DataFrame:
    """The columns of :func:`live_forecast`'s ``risk`` table for any horizons of ``fc``."""
    rows = []
    for h in horizons:
        logr = fc.returns.samples[:, h - 1]
        p = fc.prices[:, h - 1]
        rows.append({
            "horizon": h,
            "as_of": fc.as_of,
            "last_price": fc.last_price,
            "vol_ann": float(np.std(logr) * np.sqrt(252 / h)),
            "var_99": fc.value_at_risk(0.01, h),
            "var_95": fc.value_at_risk(0.05, h),
            "es_99": fc.expected_shortfall(0.01, h),
            "es_95": fc.expected_shortfall(0.05, h),
            "lo_90": float(np.quantile(p, 0.05)),
            "median": float(np.median(p)),
            "hi_90": float(np.quantile(p, 0.95)),
            "p_up": fc.prob_above(fc.last_price, h),
        })
    risk = pd.DataFrame(rows)
    risk["model"] = fc.model
    return risk


def track_record(
    prices: pd.Series,
    models: Mapping[str, ReturnForecaster],
    *,
    start: str | pd.Timestamp = "2016-01-01",
    previous: pd.DataFrame | None = None,
    redo_days: int = 40,
    n_samples: int = 2000,
) -> pd.DataFrame:
    """Daily out-of-sample forecasts scored against what happened.

    Each origin's forecast uses only the 504 returns before it (refit every
    21 days), as in :func:`~augury.evaluation.walk_forward`. With
    ``previous`` (an earlier output of this function), only (at least) the last
    ``redo_days`` origins of it are recomputed, which also fills in the
    5- and 21-day outcomes that were still open; older rows are kept as they
    were.

    Args:
        prices: Daily closes with a DatetimeIndex.
        models: Name -> fresh forecaster.
        start: First origin date when there is no ``previous``.
        previous: Earlier records to extend.
        redo_days: Trailing origins of ``previous`` to recompute.
        n_samples: Draws per forecast.

    Returns:
        One row per (origin, model, horizon): realised log return ``y``,
        ``crps``, ``pit``, forecast ``std``, 1% / 5% VaR quantiles
        ``var_1`` / ``var_5`` (log returns) with breach flags ``hit_1`` /
        ``hit_5``, and 50% / 90% interval coverage flags.
    """
    r = np.log(prices.dropna()).diff().dropna()
    keep = None
    first = pd.Timestamp(start)
    if previous is not None and not previous.empty and set(previous["model"]) >= set(models):
        origins = pd.DatetimeIndex(sorted(previous["origin"].unique()))
        if len(origins) > redo_days:
            # restart on a refit date of the original schedule (every 21st origin), so the
            # recomputed rows use the same fits a full rebuild would
            i = (len(origins) - redo_days) // _REFIT * _REFIT
            first = origins[i]
            keep = previous[(previous["origin"] < first) & previous["model"].isin(list(models))]
    if len(r) < DEFAULT_LOOKBACK + 1 or r.index[-1] < first:
        return previous if previous is not None else pd.DataFrame(columns=_TRACK_COLUMNS)
    res = walk_forward(r, dict(models), horizons=HORIZONS, lookback=DEFAULT_LOOKBACK,
                       refit_every=_REFIT, n_samples=n_samples, var_levels=VAR_LEVELS,
                       interval_levels=INTERVAL_LEVELS, start=first)
    new = res.records[_TRACK_COLUMNS]
    out = new if keep is None else pd.concat([keep[_TRACK_COLUMNS], new], ignore_index=True)
    return out.sort_values(["origin", "model", "horizon"]).reset_index(drop=True)


def score_track(
    records: pd.DataFrame,
    model: str,
    *,
    baseline: str | None = None,
    recent: int = 250,
) -> dict:
    """Headline scores of a track record for one model.

    Args:
        records: Output of :func:`track_record`.
        model: Model to score.
        baseline: Model to compare CRPS with (e.g. plain GJR), if present.
        recent: Trading days in the recent window (250 is the Basel window).

    Returns:
        Dict with ``n`` (scored 1-day forecasts), ``since``, ``var``
        (per tail level: breaches and rate overall, Kupiec / Christoffersen
        p-values, breaches and Basel zone over the last ``recent`` days),
        ``coverage_90`` and ``coverage_50`` per horizon, and, with a
        baseline, ``skill`` per horizon (1 - CRPS / baseline CRPS).
    """
    df = records[records["model"].isin([m for m in (model, baseline) if m])]
    if df.empty or not (df["model"] == model).any():
        return {"n": 0}
    one = df[(df["model"] == model) & (df["horizon"] == 1)].sort_values("origin")
    result = WalkForwardResult(records=df, horizons=HORIZONS, var_levels=VAR_LEVELS,
                               interval_levels=INTERVAL_LEVELS, eval_every=1,
                               baseline=baseline)
    bt_all = result.var_backtest(1).set_index(["model", "alpha"]).loc[model]
    recent_rows = df[(df["horizon"] == 1) & df["origin"].isin(one["origin"].iloc[-recent:])]
    recent_result = WalkForwardResult(records=recent_rows, horizons=HORIZONS, var_levels=VAR_LEVELS,
                                      interval_levels=INTERVAL_LEVELS, eval_every=1)
    bt_recent = recent_result.var_backtest(1).set_index(["model", "alpha"]).loc[model]
    var = {}
    for a in VAR_LEVELS:
        full, last = bt_all.loc[a], bt_recent.loc[a]
        var[f"{a:g}"] = {
            "breaches": int(full["violations"]), "rate": float(full["rate"]),
            "kupiec_p": float(full["kupiec_p"]),
            "christoffersen_p": float(full["christoffersen_p"]),
            "recent_n": int(last["n"]), "recent_breaches": int(last["violations"]),
            "recent_expected": float(last["n"] * a), "recent_zone": str(last["basel"]),
        }
    summ = result.summary().set_index(["model", "horizon"])
    out: dict = {
        "n": int(len(one)),
        "since": one["origin"].iloc[0],
        "through": one["origin"].iloc[-1],
        "var": var,
        "coverage_90": {int(h): float(summ.loc[(model, h), "cov_90"]) for h in HORIZONS
                        if (model, h) in summ.index},
        "coverage_50": {int(h): float(summ.loc[(model, h), "cov_50"]) for h in HORIZONS
                        if (model, h) in summ.index},
    }
    if baseline and (df["model"] == baseline).any():
        out["skill"] = {int(h): float(summ.loc[(model, h), "skill"]) for h in HORIZONS
                        if (model, h) in summ.index and pd.notna(summ.loc[(model, h), "skill"])}
    return out


def track_models(ticker: str, implied_vol: pd.Series | None = None) -> dict[str, ReturnForecaster]:
    """Models to track for ``ticker``: the monitor's model, plus plain GJR when it has a view."""
    model = monitor_model(ticker, implied_vol)
    if isinstance(model, ImpliedVolForecaster):
        return {"augury": model, "gjr": default_forecaster()}
    return {"augury": model}
