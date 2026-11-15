"""Example 33 — Known event days: earnings, and a short-dated implied vol for 1 day.

Augury's default model (GJR-GARCH skew-t) does not know when a company reports, so
its forecasts for earnings days are far too narrow: on those days returns run about
13x the variance it forecasts. ``EventForecaster`` estimates that multiplier from past
releases (only those before each fit date), shrinks event-day returns before the
model sees them, and adds the extra variance to forecasts whose window holds a
release. Release dates and times come from Yahoo (``fetch_earnings_dates``);
``event_sessions`` maps before-the-open reports to that session and after-the-close
ones to the next.

The second part swaps SPY's view for the 1-day horizon from the 30-day VIX to CBOE's
9-day VIX (``TermForecaster``).

Same leak-free ``walk_forward`` as example 32 (daily origins from 2016, 504-day
window, refit every 21 days). Takes ~5 minutes.

Result (data through 2026-11-13), CRPS vs the current model, negative = better:

    8 stocks, earnings   1 day -1.08%   5 days -1.08%   21 days -0.31%
      on earnings days   1 day -15.3%, 1% VaR breaches 16.5% -> 2.3%
    SPY, 9-day VIX       1 day -0.23% (t -3.3; -0.21% 2016-20, -0.24% 2021-26)

The earnings move is sized from the stock's past earnings moves (``size="absolute"``):
those are about the same size whatever the volatility level, so scaling them with
current volatility (``size="proportional"``: -1.03% / -0.96% / -0.23%) made forecasts
too narrow in calm markets and too wide in volatile ones.

Every stock improved, in both halves. The same overlay for CPI, jobs, PPI and FOMC
days did not help (~0 on index ETFs, about -0.3% on bond / credit ETFs, slightly worse
on equities whose views already price the events), so it is not offered.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import yfinance as yf

from augury.data.events import event_sessions, fetch_earnings_dates
from augury.evaluation import walk_forward
from augury.forecasters import (
    EventForecaster,
    GJRGarchForecaster,
    ImpliedVolForecaster,
    TermForecaster,
)

warnings.filterwarnings("ignore")

STOCKS = ["AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "TSLA", "JPM"]
START, END = "2010-01-01", "2026-11-14"


def close(ticker: str) -> pd.Series:
    df = yf.download(ticker, start=START, end=END, progress=False, auto_adjust=True)
    return df["Close"].squeeze().dropna()


def score(returns: pd.Series, models: dict) -> pd.DataFrame:
    res = walk_forward(
        returns, models, horizons=(1, 5, 21), lookback=504, refit_every=21, start="2016-01-01"
    )
    return res.records


def main() -> None:
    rows = []
    for t in STOCKS:
        r = np.log(close(t)).diff().dropna()
        events = event_sessions(fetch_earnings_dates(t))
        rec = score(
            r, {"gjr": GJRGarchForecaster("skewt"), "earnings": EventForecaster(events, history=r)}
        )
        rows.append(rec.assign(ticker=t))
        print(f"{t}: {len(events)} releases")
    rec = pd.concat(rows)
    for h in (1, 5, 21):
        g = rec[rec["horizon"] == h].pivot_table(
            index=["ticker", "origin"], columns="model", values="crps"
        )
        print(f"stocks, {h:>2} days: CRPS {g['earnings'].sum() / g['gjr'].sum() * 100 - 100:+.2f}%")

    spy = np.log(close("SPY")).diff().dropna()
    vix, vix9 = close("^VIX") / 100, close("^VIX9D") / 100
    month = ImpliedVolForecaster(vix, weight=0.5)
    term = TermForecaster(
        ImpliedVolForecaster(vix, weight=0.5),
        {1: ImpliedVolForecaster(vix9, weight=0.5, iv_horizon=6)},
    )
    rec = score(spy, {"vix": month, "vix9d for 1 day": term})
    g = rec[rec["horizon"] == 1].pivot_table(index="origin", columns="model", values="crps")
    print(f"SPY 1 day: CRPS {g['vix9d for 1 day'].sum() / g['vix'].sum() * 100 - 100:+.2f}%")


if __name__ == "__main__":
    main()
