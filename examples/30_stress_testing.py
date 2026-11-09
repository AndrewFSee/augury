"""Example 30 - Stress testing and scenario analysis
==================================================

Forward-looking stress scenarios for a multi-asset portfolio, built from the
library's default model.

The approach:
  1. Fit GJR-GARCH skew-t to each asset (SPY, TLT, GLD) on the last 2 years.
  2. Estimate the cross-asset dependence of the standardized residuals and
     use it as a Gaussian copula, so assets move together realistically.
  3. Stress scenarios multiply today's conditional variance (2x, 3x, 4x
     volatility). The shock then decays at each asset's fitted GJR
     persistence, which is how volatility behaves after real crises.
  4. Simulate correlated daily paths, hold the portfolio, and report VaR,
     CVaR, drawdowns and loss probabilities.
  5. Place past crises (GFC, COVID, 2022) inside each scenario's distribution.

Run::

    python examples/30_stress_testing.py
"""

from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats

from augury.forecasters import GJRGarchForecaster

warnings.filterwarnings("ignore")

TICKERS = ["SPY", "TLT", "GLD"]
WEIGHTS = np.array([0.60, 0.30, 0.10])
START = "2006-01-01"
LOOKBACK = 504
N_PATHS = 10_000
HORIZON = 63  # trading days
SEED = 42
SCENARIOS = {"Current": 1.0, "Vol 2x": 2.0, "COVID 3x": 3.0, "GFC 4x": 4.0}
CRISES = [
    ("GFC (Sep-Nov 2008)", "2008-09-15", "2008-11-20"),
    ("COVID (Feb-Mar 2020)", "2020-02-19", "2020-03-23"),
    ("2022 bear (Mar-Jun)", "2022-03-29", "2022-06-16"),
]

lines: list[str] = []


def log(msg: str = "") -> None:
    print(msg)
    lines.append(msg)


t0 = time.time()
closes = pd.concat(
    {
        t: yf.download(t, start=START, progress=False, auto_adjust=True)["Close"].squeeze()
        for t in TICKERS
    },
    axis=1,
).dropna()
rets = np.log(closes).diff().dropna()
window = rets.iloc[-LOOKBACK:]

log("=" * 84)
log(
    "  STRESS TESTING  " + " / ".join(f"{t} {w:.0%}" for t, w in zip(TICKERS, WEIGHTS, strict=True))
)
log(f"  as of {rets.index[-1]:%Y-%m-%d}  |  horizon {HORIZON}d  |  {N_PATHS:,} correlated paths")
log("=" * 84)

# 1. Fit each asset
models = {t: GJRGarchForecaster("skewt").fit(window[t].to_numpy()) for t in TICKERS}
log("\n  Asset          current vol   long-run vol   leverage (gamma)")
for t, m in models.items():
    p = m.params
    cur = np.sqrt(m.conditional_variance(window[t].to_numpy()) * 252)
    persist = p["alpha"] + p["gamma"] / 2 + p["beta"]
    lr = np.sqrt(p["omega"] / max(1 - persist, 1e-6) * 252) / m.scale
    log(f"  {t:<14s} {cur:>10.1%}   {lr:>11.1%}   {p['gamma']:>14.3f}")

# 2. Gaussian copula from standardized residuals (normal scores of ranks)
resid = np.column_stack([models[t].standardized_residuals() for t in TICKERS])
resid = resid[np.all(np.isfinite(resid), axis=1)]
scores = stats.norm.ppf(stats.rankdata(resid, axis=0) / (len(resid) + 1))
corr = np.corrcoef(scores, rowvar=False)
log("\n  Residual dependence (copula correlation)")
log(pd.DataFrame(corr, index=TICKERS, columns=TICKERS).round(2).to_string())

rng = np.random.default_rng(SEED)
z = rng.standard_normal((N_PATHS, HORIZON, len(TICKERS))) @ np.linalg.cholesky(corr).T
uniforms = stats.norm.cdf(z)  # shared across scenarios: differences are the shock alone

# 3-4. Simulate each scenario
results = {}
for name, vol_mult in SCENARIOS.items():
    wealth_assets = {}
    for i, t in enumerate(TICKERS):
        daily = models[t].simulate(
            window[t].to_numpy(), HORIZON, variance_scale=vol_mult**2, uniforms=uniforms[:, :, i]
        )
        wealth_assets[t] = np.exp(np.cumsum(daily, axis=1))
    wealth = sum(w * wealth_assets[t] for w, t in zip(WEIGHTS, TICKERS, strict=True))
    ret = wealth[:, -1] - 1.0
    dd = np.min(wealth / np.maximum.accumulate(np.maximum(wealth, 1.0), axis=1) - 1, axis=1)
    var1 = np.quantile(ret, 0.01)
    standalone = sum(
        w * np.quantile(wealth_assets[t][:, -1] - 1.0, 0.01)
        for w, t in zip(WEIGHTS, TICKERS, strict=True)
    )
    results[name] = {
        "VaR 1%": var1,
        "CVaR 1%": ret[ret <= var1].mean(),
        "VaR 5%": np.quantile(ret, 0.05),
        "Median max DD": np.median(dd),
        "Worst max DD (1%)": np.quantile(dd, 0.01),
        "P(loss > 10%)": np.mean(ret < -0.10),
        "P(loss > 20%)": np.mean(ret < -0.20),
        "Diversification": 1 - var1 / standalone,
        "_wealth": wealth,
    }

table = pd.DataFrame(
    {k: {m: v for m, v in r.items() if not m.startswith("_")} for k, r in results.items()}
)
log(f"\n  Portfolio risk over {HORIZON} trading days")
log(table.map(lambda v: f"{v:+.1%}" if v < 0 else f"{v:.1%}").to_string())

# 5. Where would past crises have landed?
log("\n  Historical crises vs scenarios: P(simulated loss at least as bad)")
port = (np.exp(rets) - 1) @ WEIGHTS
rows = []
for name, start, end in CRISES:
    seg = port[(rets.index >= start) & (rets.index <= end)]
    if seg.empty:
        continue
    n = len(seg)
    if n > HORIZON:
        raise ValueError(f"{name} spans {n} days, longer than the {HORIZON}-day horizon")
    actual = float(np.prod(1 + seg.to_numpy()) - 1)
    row = {"crisis": name, "days": n, "portfolio": f"{actual:+.1%}"}
    for sc, r in results.items():
        row[sc] = f"{np.mean(r['_wealth'][:, n - 1] - 1 <= actual):.1%}"
    rows.append(row)
log(pd.DataFrame(rows).to_string(index=False))

log(f"\n  Runtime {time.time() - t0:.0f}s")
with open("_stress_test_output.txt", "w", encoding="utf-8") as fh:
    fh.write("\n".join(lines))
