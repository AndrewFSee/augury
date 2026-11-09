"""Example 28 - VaR backtesting report
====================================

The canonical risk-management exercise: forecast daily Value-at-Risk with
progressively better models and test whether the forecasts are calibrated.
Everything here uses the library's forecasters and walk-forward harness, so
the numbers in the README can be reproduced exactly.

Models (each re-fit every 21 days on a rolling 504-day window, forecast daily):
  historical  empirical quantile of trailing returns (bootstrap)
  gaussian    trailing volatility x normal quantile
  ewma        RiskMetrics EWMA volatility (lambda = 0.94) x normal
  laplace     MLE Laplace on the last 21 days
  har         HAR volatility with empirical standardized quantiles
  gjr-normal  GJR-GARCH conditional volatility x normal
  fhs         GJR-GARCH volatility x bootstrapped residuals
  gjr-skewt   GJR-GARCH conditional volatility x Hansen skew-t (the default)

Tests:
  Kupiec (1995)          unconditional coverage: right violation rate?
  Christoffersen (1998)  independence: are violations clustered?
  Basel traffic light    green / yellow / red, from the binomial distribution

Run::

    python examples/28_var_backtesting.py
"""

from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd
import yfinance as yf

from augury.evaluation import walk_forward
from augury.forecasters import (
    FilteredHistoricalSimulation,
    GaussianForecaster,
    GJRGarchForecaster,
    HARForecaster,
    HistoricalSimulation,
    LaplaceForecaster,
)

warnings.filterwarnings("ignore")

TICKER = "SPY"
START = "2012-01-01"
LOOKBACK = 504
REFIT_EVERY = 21
CRISES = [
    ("COVID crash", "2020-02-19", "2020-03-23"),
    ("2022 bear market", "2022-01-03", "2022-10-12"),
    ("Aug 2024 selloff", "2024-07-15", "2024-08-15"),
]

lines: list[str] = []


def log(msg: str = "") -> None:
    print(msg)
    lines.append(msg)


t0 = time.time()
close = yf.download(TICKER, start=START, progress=False, auto_adjust=True)["Close"].squeeze()
returns = np.log(close).diff().dropna()

models = [
    HistoricalSimulation(),
    GaussianForecaster(),
    GaussianForecaster(ewma_lambda=0.94),
    LaplaceForecaster(),
    HARForecaster(),
    GJRGarchForecaster("normal"),
    FilteredHistoricalSimulation(),
    GJRGarchForecaster("skewt"),
]
res = walk_forward(
    returns,
    models,
    horizons=(1,),
    lookback=LOOKBACK,
    refit_every=REFIT_EVERY,
    n_samples=5000,
    var_levels=(0.01, 0.05),
    baseline="gaussian",
)
rec = res.records
n_days = rec["origin"].nunique()
first, last = rec["origin"].min(), rec["origin"].max()

log("=" * 88)
log(f"  VaR BACKTEST  {TICKER}  {first:%Y-%m-%d} -> {last:%Y-%m-%d}  ({n_days:,} days)")
log(f"  rolling {LOOKBACK}-day window, refit every {REFIT_EVERY} days")
log("=" * 88)

vb = res.var_backtest(horizon=1)
for a in (0.01, 0.05):
    log(f"\n{a:.0%} VaR  (expected violations ~ {a * n_days:.0f})")
    t = vb[vb["alpha"] == a].drop(columns=["alpha", "n"])
    log(t.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

log("\nForecast accuracy (1-day CRPS, skill vs rolling Gaussian)")
s = res.summary()[["model", "crps", "skill", "dm_p", "pit_ks_p"]]
log(s.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

log("\nCrisis periods (1% VaR)")
for name, start, end in CRISES:
    window = rec[(rec["origin"] >= start) & (rec["origin"] <= end)]
    if window.empty:
        continue
    log(f"\n  {name}  ({window['origin'].nunique()} days)")
    rows = []
    for model, g in window.groupby("model", sort=False):
        worst = g.loc[g["y"].idxmin()]
        rows.append(
            {
                "model": model,
                "violations": int(g["hit_1"].sum()),
                "rate": g["hit_1"].mean(),
                "worst_day": f"{worst['y']:+.2%}",
                "var_that_day": f"{worst['var_1']:+.2%}",
            }
        )
    log(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:.1%}"))

log(f"\nRuntime {time.time() - t0:.0f}s")
with open("_var_backtest_output.txt", "w", encoding="utf-8") as fh:
    fh.write("\n".join(lines))
