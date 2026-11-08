"""Example 20 — Market-Implied Regime Adjustment for Forecast Calibration.

Tests whether market-implied data (VIX term structure, SKEW, credit spreads,
MOVE) **improves the probabilistic forecast** by adjusting regime probabilities
that control simulation parameters — NOT by trading on them.

Compares two strategies at each walk-forward step:

  1. **HMM-only** — regime probabilities from returns alone.
  2. **Market-adjusted** — regime probabilities adjusted via Bayesian update
     using options/credit market data.

We measure **forecast quality**, not trading P&L:
  - CRPS (lower = better calibrated forecast)
  - 90% coverage (closer to 90% = better calibrated)
  - VaR 95 accuracy (P(actual < VaR) should ≈ 5%)
  - Expected return accuracy (|E[r] - actual|)
  - Regime accuracy (did the regime detector correctly identify calm vs crisis?)

The hypothesis: VIX backwardation, elevated SKEW, widening credit spreads,
and MOVE spikes provide evidence that the true crisis probability is higher
than the HMM estimates from returns alone.  This shifts the simulation
toward more Merton JD paths (jumps), higher vol, and more negative drift,
producing a forecast distribution that better captures tail risk.
"""

from __future__ import annotations

import time
import warnings

import numpy as np

# ---------------------------------------------------------------------------
_out_path = "_market_regime_adjustment_output.txt"
_out_file = open(_out_path, "w", encoding="utf-8")  # noqa: SIM115
_builtin_print = print


def print(*args, **kwargs):  # noqa: A001
    _builtin_print(*args, **kwargs)
    kwargs.pop("file", None)
    _builtin_print(*args, file=_out_file, **kwargs)
    _out_file.flush()


# ---------------------------------------------------------------------------
TICKERS = ["SPY", "QQQ", "IWM", "DIA", "EFA", "TLT", "GLD"]
PERIODS = [
    ("2015-2019", "2015-01-01", "2020-01-01"),
    ("2020-2022", "2020-01-01", "2023-01-01"),
    ("2023-2026", "2023-01-01", "2026-12-31"),
]

LOOKBACK = 504
HORIZON = 21
STEP = 21
N_PATHS = 3000

print("=" * 80)
print("  MARKET-IMPLIED REGIME ADJUSTMENT FOR FORECAST CALIBRATION")
print(f"  Tickers: {', '.join(TICKERS)}")
print(f"  Periods: {', '.join(p[0] for p in PERIODS)}")
print(f"  Lookback={LOOKBACK} | Horizon={HORIZON} | Step={STEP} | Paths={N_PATHS}")
print("=" * 80)
print()

# ---------------------------------------------------------------------------
from synthquant.data.ingest import DataIngestor
from synthquant.data.market_data import MarketDataFetcher
from synthquant.ensemble_forecast import EnsembleForecaster

ingestor = DataIngestor()

# Pre-fetch market-implied data
print("Fetching market-implied data (VIX, VIX3M, SKEW, MOVE, HYG, IEF)...")
try:
    mkt_fetcher = MarketDataFetcher(start="2014-06-01", end="2026-12-31")
    print(f"  Available series: {mkt_fetcher.available_series}")
except Exception as e:
    print(f"  Market data fetch failed: {e}")
    mkt_fetcher = None
print()


# ---------------------------------------------------------------------------
def walk_forward_score(
    df,
    use_market_data: bool,
) -> dict[str, list[float]]:
    """Walk-forward forecast scoring.

    At each step:
    1. Fit EnsembleForecaster on lookback window
    2. Generate forecast (with or without market adjustment)
    3. Score forecast against realised returns

    Returns dict of metric lists.
    """
    returns = np.log(df["close"] / df["close"].shift(1)).dropna().values
    n = len(returns)

    crps_list = []
    coverage_90 = []
    coverage_50 = []
    var_breaches = []  # 1 if actual < VaR95 (predicted tail breach)
    mean_abs_errors = []
    regime_at_forecast = []
    crisis_proba_at_forecast = []
    engine_weights_list = []

    t = LOOKBACK
    while t + HORIZON <= n:
        window_returns = returns[t - LOOKBACK : t]
        actual_fwd = returns[t : t + HORIZON]

        # Get market data for this window if available
        market_snapshot = None
        if use_market_data and mkt_fetcher is not None:
            eq_dates = df.index[1:]  # aligned with returns
            window_dates = eq_dates[t - LOOKBACK : t]
            if len(window_dates) > 0:
                market_snapshot = mkt_fetcher.slice(
                    window_dates[0], window_dates[-1]
                )
                if market_snapshot.n_days < 30:
                    market_snapshot = None

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fc = EnsembleForecaster(
                    detector="hmm",
                    n_regimes=2,
                    n_paths=N_PATHS,
                    random_state=42,
                    use_filtered=True,
                )
                fc.fit(window_returns)

                forecast = fc.forecast(
                    horizon=HORIZON,
                    current_price=100.0,
                    market_data=market_snapshot,
                    market_adjust_strength=0.5,
                )

                # Score against realised
                scores = fc.score(forecast, actual_fwd)

                crps_list.append(scores["crps"])
                coverage_90.append(1.0 if scores["actual_in_90pct_interval"] else 0.0)
                coverage_50.append(1.0 if scores["actual_in_50pct_interval"] else 0.0)

                # VaR accuracy: was actual < VaR95?
                actual_terminal = float(np.sum(actual_fwd))
                var_breach = 1.0 if actual_terminal < forecast.var_95 else 0.0
                var_breaches.append(var_breach)

                # Absolute forecast error
                mae = abs(forecast.expected_return - actual_terminal)
                mean_abs_errors.append(mae)

                # Regime info
                regime_at_forecast.append(forecast.current_regime)
                crisis_proba_at_forecast.append(
                    float(forecast.regime_probabilities[-1])  # last state = crisis
                )
                engine_weights_list.append(forecast.metadata["engine_weights"])

        except Exception:
            pass

        t += STEP

    return {
        "crps": crps_list,
        "coverage_90": coverage_90,
        "coverage_50": coverage_50,
        "var_breaches": var_breaches,
        "mae": mean_abs_errors,
        "regimes": regime_at_forecast,
        "crisis_proba": crisis_proba_at_forecast,
        "engine_weights": engine_weights_list,
    }


# ---------------------------------------------------------------------------
results_db: list[dict] = []
cell = 0
total_cells = len(TICKERS) * len(PERIODS)

for ticker in TICKERS:
    try:
        df_full = ingestor.fetch(ticker, start="2014-06-01", end="2026-12-31")
    except Exception as e:
        print(f"  SKIP {ticker}: {e}")
        continue

    for period_name, p_start, p_end in PERIODS:
        cell += 1
        mask = (df_full.index >= p_start) & (df_full.index <= p_end)
        df = df_full.loc[mask]

        if len(df) < LOOKBACK + HORIZON + 10:
            print(f"  [{cell}/{total_cells}] {ticker} / {period_name}  SKIP")
            continue

        print(f"  [{cell}/{total_cells}] {ticker} / {period_name}  ({len(df)} days)")
        t0 = time.perf_counter()

        hmm_scores = walk_forward_score(df, use_market_data=False)
        mkt_scores = walk_forward_score(df, use_market_data=True)

        elapsed = time.perf_counter() - t0
        n_obs = len(hmm_scores["crps"])

        if n_obs == 0:
            print(f"    No valid forecasts  ({elapsed:.0f}s)")
            continue

        # Summarise
        hmm_crps = np.mean(hmm_scores["crps"])
        mkt_crps = np.mean(mkt_scores["crps"])
        hmm_cov90 = np.mean(hmm_scores["coverage_90"])
        mkt_cov90 = np.mean(mkt_scores["coverage_90"])
        hmm_cov50 = np.mean(hmm_scores["coverage_50"])
        mkt_cov50 = np.mean(mkt_scores["coverage_50"])
        hmm_var_breach = np.mean(hmm_scores["var_breaches"])
        mkt_var_breach = np.mean(mkt_scores["var_breaches"])
        hmm_mae = np.mean(hmm_scores["mae"])
        mkt_mae = np.mean(mkt_scores["mae"])
        hmm_crisis_p = np.mean(hmm_scores["crisis_proba"])
        mkt_crisis_p = np.mean(mkt_scores["crisis_proba"])

        crps_delta = (mkt_crps - hmm_crps) / max(abs(hmm_crps), 1e-8)
        crps_better = "+" if mkt_crps < hmm_crps else "-"

        print(f"    {n_obs} forecasts  ({elapsed:.0f}s)")
        print(
            f"    HMM-only:  CRPS={hmm_crps:.5f}  Cov90={hmm_cov90:.0%}  "
            f"Cov50={hmm_cov50:.0%}  VaR95={hmm_var_breach:.0%}  "
            f"MAE={hmm_mae:.4f}  P(crisis)={hmm_crisis_p:.3f}"
        )
        print(
            f"    Mkt-Adj:   CRPS={mkt_crps:.5f}  Cov90={mkt_cov90:.0%}  "
            f"Cov50={mkt_cov50:.0%}  VaR95={mkt_var_breach:.0%}  "
            f"MAE={mkt_mae:.4f}  P(crisis)={mkt_crisis_p:.3f}  "
            f"CRPS Δ={crps_delta:+.1%} [{crps_better}]"
        )

        results_db.append({
            "ticker": ticker,
            "period": period_name,
            "n_obs": n_obs,
            "hmm_crps": hmm_crps,
            "mkt_crps": mkt_crps,
            "hmm_cov90": hmm_cov90,
            "mkt_cov90": mkt_cov90,
            "hmm_cov50": hmm_cov50,
            "mkt_cov50": mkt_cov50,
            "hmm_var_breach": hmm_var_breach,
            "mkt_var_breach": mkt_var_breach,
            "hmm_mae": hmm_mae,
            "mkt_mae": mkt_mae,
            "hmm_crisis_p": hmm_crisis_p,
            "mkt_crisis_p": mkt_crisis_p,
        })

print()

# ---------------------------------------------------------------------------
if results_db:
    print("=" * 80)
    print("  AGGREGATE RESULTS")
    print("=" * 80)

    # CRPS comparison (lower = better)
    hmm_crps_all = np.mean([r["hmm_crps"] for r in results_db])
    mkt_crps_all = np.mean([r["mkt_crps"] for r in results_db])
    crps_wins = sum(1 for r in results_db if r["mkt_crps"] < r["hmm_crps"])

    print("\n  CRPS (lower = better forecast):")
    print(f"    HMM-only:   {hmm_crps_all:.5f}")
    print(f"    Mkt-Adj:    {mkt_crps_all:.5f}")
    print(f"    Mkt wins:   {crps_wins}/{len(results_db)}")

    # Coverage comparison (closer to nominal = better)
    hmm_cov90 = np.mean([r["hmm_cov90"] for r in results_db])
    mkt_cov90 = np.mean([r["mkt_cov90"] for r in results_db])
    hmm_cov90_err = abs(hmm_cov90 - 0.90)
    mkt_cov90_err = abs(mkt_cov90 - 0.90)

    print("\n  90% Coverage (target = 90%):")
    print(f"    HMM-only:   {hmm_cov90:.1%}  (error={hmm_cov90_err:.1%})")
    print(f"    Mkt-Adj:    {mkt_cov90:.1%}  (error={mkt_cov90_err:.1%})")
    cov_better = "Mkt-Adj" if mkt_cov90_err < hmm_cov90_err else "HMM-only"
    print(f"    Better:     {cov_better}")

    # VaR accuracy (target = 5%)
    hmm_var = np.mean([r["hmm_var_breach"] for r in results_db])
    mkt_var = np.mean([r["mkt_var_breach"] for r in results_db])
    hmm_var_err = abs(hmm_var - 0.05)
    mkt_var_err = abs(mkt_var - 0.05)

    print("\n  VaR 95% breach rate (target = 5%):")
    print(f"    HMM-only:   {hmm_var:.1%}  (error={hmm_var_err:.1%})")
    print(f"    Mkt-Adj:    {mkt_var:.1%}  (error={mkt_var_err:.1%})")
    var_better = "Mkt-Adj" if mkt_var_err < hmm_var_err else "HMM-only"
    print(f"    Better:     {var_better}")

    # Mean absolute error
    hmm_mae_all = np.mean([r["hmm_mae"] for r in results_db])
    mkt_mae_all = np.mean([r["mkt_mae"] for r in results_db])
    mae_wins = sum(1 for r in results_db if r["mkt_mae"] < r["hmm_mae"])

    print("\n  Forecast MAE (lower = better):")
    print(f"    HMM-only:   {hmm_mae_all:.5f}")
    print(f"    Mkt-Adj:    {mkt_mae_all:.5f}")
    print(f"    Mkt wins:   {mae_wins}/{len(results_db)}")

    # Average crisis probability
    hmm_cp = np.mean([r["hmm_crisis_p"] for r in results_db])
    mkt_cp = np.mean([r["mkt_crisis_p"] for r in results_db])

    print("\n  Avg P(crisis):")
    print(f"    HMM-only:   {hmm_cp:.3f}")
    print(f"    Mkt-Adj:    {mkt_cp:.3f}")

    # Per-period analysis
    print("\n  --- Per-Period CRPS Wins ---")
    for period_name, _, _ in PERIODS:
        subset = [r for r in results_db if r["period"] == period_name]
        if subset:
            wins = sum(1 for r in subset if r["mkt_crps"] < r["hmm_crps"])
            print(f"    {period_name}: Mkt-Adj wins {wins}/{len(subset)}")

print()
print("=" * 80)
print("  Done.")
print("=" * 80)

_out_file.close()
