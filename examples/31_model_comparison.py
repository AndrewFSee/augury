"""Example 31 - Which forecaster actually works? A walk-forward bake-off.

Every distributional forecaster in the library is scored out of sample with
the same harness, the same rolling window and the same forecast origins:

  Baselines     gaussian (rolling vol), ewma (RiskMetrics 0.94),
                historical simulation, laplace (the CrunchDAO winner)
  Vol models    gjr-normal, gjr-skewt, fhs (filtered historical sim), har
  Regime models regime-hmm (ProbabilisticForecaster),
                ensemble (EnsembleForecaster: Heston/Merton/RS/RB)

Part 1 evaluates the fast models every day on several ETFs.
Part 2 adds the regime models on SPY, evaluated monthly (they are slow and
only condition on data at refit time), with all models on the same dates.

Scores are CRPS in log-return units (lower is better). ``skill`` is the
CRPS improvement over the rolling-Gaussian baseline and ``dm_p`` the
Diebold-Mariano p-value of that difference.

Run::

    python examples/31_model_comparison.py
"""

from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd
import yfinance as yf

from synthquant.ensemble_forecast import EnsembleForecaster
from synthquant.evaluation import walk_forward
from synthquant.forecast import ProbabilisticForecaster
from synthquant.forecasters import (
    FilteredHistoricalSimulation,
    GaussianForecaster,
    GJRGarchForecaster,
    HARForecaster,
    HistoricalSimulation,
    LaplaceForecaster,
    PathModelForecaster,
)

warnings.filterwarnings("ignore")
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)

TICKERS = ["SPY", "QQQ", "IWM", "TLT", "GLD"]
START = "2010-01-01"
HORIZONS = (1, 5, 21)
LOOKBACK = 504
REFIT_EVERY = 21
N_SAMPLES = 2000


def fast_models() -> list:
    return [
        GaussianForecaster(),
        GaussianForecaster(ewma_lambda=0.94),
        HistoricalSimulation(),
        LaplaceForecaster(),
        GJRGarchForecaster("normal"),
        GJRGarchForecaster("skewt"),
        FilteredHistoricalSimulation(),
        HARForecaster(),
    ]


def load_log_returns(ticker: str) -> pd.Series:
    df = yf.download(ticker, start=START, progress=False, auto_adjust=True)
    close = df["Close"].squeeze().dropna()
    return np.log(close).diff().dropna().rename(ticker)


def show(title: str, df: pd.DataFrame) -> None:
    print(f"\n{title}")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


t_start = time.time()
data = {t: load_log_returns(t) for t in TICKERS}

# --- Part 1: fast models, daily origins, several assets ---------------------
print("=" * 100)
print("PART 1  Fast models, daily walk-forward")
print(f"        lookback={LOOKBACK}d refit={REFIT_EVERY}d horizons={HORIZONS} samples={N_SAMPLES}")
print("=" * 100)

skill_rows = []
for ticker, r in data.items():
    t0 = time.time()
    res = walk_forward(
        r,
        fast_models(),
        horizons=HORIZONS,
        lookback=LOOKBACK,
        refit_every=REFIT_EVERY,
        n_samples=N_SAMPLES,
    )
    summ = res.summary()
    print(
        f"\n### {ticker}: {res.records['origin'].min():%Y-%m-%d} -> "
        f"{res.records['origin'].max():%Y-%m-%d}  ({time.time() - t0:.0f}s)"
    )
    show(
        "Scores",
        summ[
            [
                "model",
                "horizon",
                "n",
                "crps",
                "skill",
                "dm_p",
                "pit_ks_p",
                "cov_90",
                "hit_1",
                "hit_5",
            ]
        ],
    )
    vb = res.var_backtest(horizon=1)
    show("1-day VaR backtests", vb[vb["alpha"] == 0.01].drop(columns="alpha"))
    for _, row in summ.iterrows():
        skill_rows.append(
            {
                "ticker": ticker,
                "model": row["model"],
                "horizon": row["horizon"],
                "skill": row["skill"],
            }
        )

skill = pd.DataFrame(skill_rows)
table = skill.pivot_table(index="model", columns=["horizon", "ticker"], values="skill")
avg = skill.groupby(["model", "horizon"])["skill"].mean().unstack()
avg.columns = [f"avg_h{h}" for h in avg.columns]
print("\n" + "=" * 100)
print("CRPS skill vs rolling Gaussian (positive = better), averaged over assets")
print("=" * 100)
print(avg.sort_values("avg_h1", ascending=False).to_string(float_format=lambda x: f"{x:+.2%}"))

# --- Part 2: add the regime models on SPY, monthly origins ------------------
print("\n" + "=" * 100)
print("PART 2  SPY, monthly origins: fast models vs regime / ensemble forecasters")
print("=" * 100)
models = fast_models() + [
    PathModelForecaster(
        lambda: ProbabilisticForecaster(n_paths=N_SAMPLES, random_state=0), name="regime-hmm"
    ),
    PathModelForecaster(
        lambda: EnsembleForecaster(n_paths=N_SAMPLES, random_state=0, vol_model="GJRGARCH"),
        name="ensemble",
    ),
]
t0 = time.time()
res = walk_forward(
    data["SPY"],
    models,
    horizons=HORIZONS,
    lookback=LOOKBACK,
    refit_every=REFIT_EVERY,
    eval_every=REFIT_EVERY,
    n_samples=N_SAMPLES,
)
print(f"({time.time() - t0:.0f}s, fit failures: {res.fit_failures})")
show(
    "Scores",
    res.summary()[
        ["model", "horizon", "n", "crps", "skill", "dm_p", "pit_ks_p", "cov_50", "cov_90"]
    ],
)

print(f"\nTotal runtime {time.time() - t_start:.0f}s")
