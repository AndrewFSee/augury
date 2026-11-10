"""Example 32 — Implied-vol views on top of GJR-GARCH.

Does pulling the GJR volatility forecast toward an implied-vol index improve
out-of-sample forecasts? Each ETF is paired with its index:

    SPY, IWM -> VIX     QQQ -> VXN     GLD -> GVZ     TLT -> MOVE

``ImpliedVolForecaster`` blends GJR's 21-day expected variance with the
index's implied variance and scales the whole forecast so every horizon moves
by the same factor. The index is used at face value except MOVE (bond-yield
vol in basis points), whose scale is estimated on each training window.

Everything runs through the leak-free ``walk_forward`` harness: daily
origins, 504-day rolling window, monthly refits, scored at 1, 5 and 21 days.
Takes ~10 minutes.

Result (data through 2026-11-06): weight 0.5 improves CRPS by 0.3-0.8% on
SPY/QQQ/IWM/GLD at every horizon (Diebold-Mariano p < 0.02 in 10 of 12
cases) and keeps every 1% VaR backtest Basel green. Weight 1 (trusting the
index fully) is worse than GJR. TLT gains ~nothing.

For your own view instead of an index, use
``forecast_prices(prices, 21, target_vol=0.25)``.
"""

from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd
import yfinance as yf

from augury.evaluation import walk_forward
from augury.forecasters import GJRGarchForecaster, ImpliedVolForecaster

warnings.filterwarnings("ignore")

PAIRS = {"SPY": "^VIX", "QQQ": "^VXN", "IWM": "^VIX", "GLD": "^GVZ", "TLT": "^MOVE"}
START, END = "2010-01-01", "2026-11-07"


def close(ticker: str) -> pd.Series:
    df = yf.download(ticker, start=START, end=END, progress=False, auto_adjust=True)
    return df["Close"].squeeze().dropna()


def main() -> None:
    skill, var1 = [], []
    for ticker, index in PAIRS.items():
        t0 = time.time()
        returns = np.log(close(ticker)).diff().dropna()
        iv = close(index) / 100.0
        calibrate = index == "^MOVE"  # not in price-vol units
        models = {
            "gjr": GJRGarchForecaster("skewt"),
            "iv 0.25": ImpliedVolForecaster(iv, weight=0.25, calibrate=calibrate),
            "iv 0.5": ImpliedVolForecaster(iv, weight=0.5, calibrate=calibrate),
            "iv 1.0": ImpliedVolForecaster(iv, weight=1.0, calibrate=calibrate),
        }
        res = walk_forward(returns, models, horizons=(1, 5, 21), n_samples=2000, baseline="gjr")
        s = res.summary()
        s.insert(0, "ticker", ticker)
        skill.append(s)
        v = res.var_backtest(1)
        v.insert(0, "ticker", ticker)
        var1.append(v)
        print(f"{ticker} vs {index}: {time.time() - t0:.0f}s", flush=True)

    S = pd.concat(skill, ignore_index=True)
    V = pd.concat(var1, ignore_index=True)
    pd.set_option("display.width", 200)
    print("\nCRPS skill vs GJR skew-t (positive = better)")
    print(
        S[S.model != "gjr"]
        .pivot_table(index=["model", "horizon"], columns="ticker", values="skill")
        .map(lambda x: f"{x:+.2%}")
    )
    print("\nDiebold-Mariano p-value vs GJR")
    print(
        S[S.model != "gjr"]
        .pivot_table(index=["model", "horizon"], columns="ticker", values="dm_p")
        .round(3)
    )
    print("\n1-day 1% VaR: violations (expected ~37) and Basel zone")
    v1 = V[V.alpha == 0.01]
    print(v1.pivot_table(index="model", columns="ticker", values="violations"))
    print(v1.pivot_table(index="model", columns="ticker", values="basel", aggfunc="first"))


if __name__ == "__main__":
    main()
