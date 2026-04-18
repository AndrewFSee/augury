"""Example 13 — Regime detection stability diagnostic.

Compares smoothed vs filtered HMM regime detection across rolling windows
to quantify the instability problem: how much do regime assignments at
overlapping time points change when the window shifts forward?

Also re-runs the default robustness test with use_filtered=True to see
if forward-only filtering improves real-world performance.
"""

from __future__ import annotations

import sys
import time

import numpy as np

# ---------------------------------------------------------------------------
# Output to both stdout and file
# ---------------------------------------------------------------------------
_out_path = "_regime_stability_output.txt"
_out_file = open(_out_path, "w", encoding="utf-8")  # noqa: SIM115
_builtin_print = print


def print(*args, **kwargs):  # noqa: A001
    _builtin_print(*args, **kwargs)
    kwargs.pop("file", None)
    _builtin_print(*args, file=_out_file, **kwargs)
    _out_file.flush()


# ===========================================================================
# Part 1: HMM regime stability diagnostic
# ===========================================================================

print("=" * 80)
print("  REGIME STABILITY DIAGNOSTIC")
print("  Comparing smoothed vs filtered HMM on overlapping windows")
print("=" * 80)

# --- Fetch data ---
from synthquant.data.ingest import DataIngestor

ingestor = DataIngestor()
df = ingestor.fetch("SPY", start="2015-01-01", end="2026-01-01")
returns = np.log(df["close"] / df["close"].shift(1)).dropna().values
n = len(returns)

print(f"\n  SPY daily returns: {n} days")

from synthquant.regime.hmm import HMMRegimeDetector

lookback = 504
step = 21
windows = []
start = 0
while start + lookback <= n:
    windows.append((start, start + lookback))
    start += step

print(f"  Windows: {len(windows)} overlapping ({lookback}d lookback, {step}d step)")
print()

# --- Measure regime assignment stability ---
# For each pair of consecutive windows, check how the regime assignment
# at overlapping time points changes.

print("  Fitting HMM on each window...", end=" ", flush=True)
t0 = time.perf_counter()

smoothed_regimes = []
filtered_regimes = []
smoothed_probas = []
filtered_probas = []

for w_start, w_end in windows:
    det = HMMRegimeDetector(n_components=2, random_state=42)
    det.fit(returns[w_start:w_end])

    smoothed_regimes.append(det.predict(returns[w_start:w_end]))
    filtered_regimes.append(det.predict_filtered(returns[w_start:w_end]))
    smoothed_probas.append(det.predict_proba(returns[w_start:w_end]))
    filtered_probas.append(det.predict_proba_filtered(returns[w_start:w_end]))

print(f"done ({time.perf_counter() - t0:.1f}s)")
print()

# Compute stability: for consecutive windows, measure disagreement
# in regime labels at overlapping time points
overlap = lookback - step  # number of overlapping days

smoothed_flips = []
filtered_flips = []
smoothed_proba_diffs = []
filtered_proba_diffs = []

for i in range(len(windows) - 1):
    # Window i covers [w_start_i, w_end_i)
    # Window i+1 covers [w_start_i + step, w_end_i + step)
    # Overlap: window i positions [step:] vs window i+1 positions [:overlap]
    s_old = smoothed_regimes[i][step:]
    s_new = smoothed_regimes[i + 1][:overlap]
    smoothed_flips.append(np.mean(s_old != s_new))

    f_old = filtered_regimes[i][step:]
    f_new = filtered_regimes[i + 1][:overlap]
    filtered_flips.append(np.mean(f_old != f_new))

    # Probability difference at overlap
    sp_old = smoothed_probas[i][step:]
    sp_new = smoothed_probas[i + 1][:overlap]
    smoothed_proba_diffs.append(np.mean(np.abs(sp_old - sp_new)))

    fp_old = filtered_probas[i][step:]
    fp_new = filtered_probas[i + 1][:overlap]
    filtered_proba_diffs.append(np.mean(np.abs(fp_old - fp_new)))

print("  REGIME ASSIGNMENT STABILITY (overlapping windows)")
print("  -------------------------------------------------")
print(f"  Smoothed (forward-backward):")
print(f"    Mean flip rate:        {np.mean(smoothed_flips):.1%}")
print(f"    Max flip rate:         {np.max(smoothed_flips):.1%}")
print(f"    Mean proba |delta|:    {np.mean(smoothed_proba_diffs):.4f}")
print()
print(f"  Filtered (forward-only):")
print(f"    Mean flip rate:        {np.mean(filtered_flips):.1%}")
print(f"    Max flip rate:         {np.max(filtered_flips):.1%}")
print(f"    Mean proba |delta|:    {np.mean(filtered_proba_diffs):.4f}")
print()

# Last-day regime: the one that matters for trading
smoothed_last_regime = [sr[-1] for sr in smoothed_regimes]
filtered_last_regime = [fr[-1] for fr in filtered_regimes]
smoothed_last_proba = [sp[-1] for sp in smoothed_probas]
filtered_last_proba = [fp[-1] for fp in filtered_probas]

# How often does the "current regime" (last day) differ between methods?
agree = sum(1 for s, f in zip(smoothed_last_regime, filtered_last_regime) if s == f)
print(f"  Smoothed vs Filtered agreement on LAST-DAY regime: {agree}/{len(windows)} "
      f"({agree/len(windows):.0%})")

# Confidence (max probability) at last day
smoothed_conf = [max(p) for p in smoothed_last_proba]
filtered_conf = [max(p) for p in filtered_last_proba]
print(f"  Smoothed last-day confidence:  mean={np.mean(smoothed_conf):.2f}, "
      f"min={np.min(smoothed_conf):.2f}")
print(f"  Filtered last-day confidence:  mean={np.mean(filtered_conf):.2f}, "
      f"min={np.min(filtered_conf):.2f}")
print()

# ===========================================================================
# Part 2: Signal comparison — smoothed vs filtered
# ===========================================================================

print("=" * 80)
print("  SIGNAL COMPARISON: SMOOTHED vs FILTERED")
print("  Default ForecastSignal on SPY, 2015-2026")
print("=" * 80)
print()

from synthquant.ensemble_forecast import EnsembleForecaster
from synthquant.strategy.forecast_signals import ForecastSignal

LOOKBACK = 504
HORIZON = 21
STEP = 21
N_PATHS = 3000


def run_walk_forward(use_filtered: bool) -> dict:
    """Run walk-forward signal generation with smoothed or filtered regime."""
    label = "Filtered" if use_filtered else "Smoothed"
    fc = EnsembleForecaster(
        detector="hmm",
        n_regimes=2,
        n_paths=N_PATHS,
        random_state=42,
        use_filtered=use_filtered,
    )
    sig = ForecastSignal()

    signals = []
    convictions = []
    regimes = []
    actual_rets = []

    n_snap = 0
    t = LOOKBACK
    while t + HORIZON <= len(returns):
        window = returns[t - LOOKBACK : t]
        future = returns[t : t + HORIZON]

        fc.fit(window)
        forecast = fc.forecast(horizon=HORIZON, current_price=100.0)
        recent = returns[max(0, t - 63) : t]
        result = sig.generate(forecast, recent_returns=recent)

        signals.append(result.signal)
        convictions.append(result.conviction)
        regimes.append(result.regime)
        actual_rets.append(float(np.sum(future)))

        n_snap += 1
        t += STEP

    signals = np.array(signals)
    convictions = np.array(convictions)
    actual_rets = np.array(actual_rets)

    # Strategy return: signal × conviction × actual
    strat_rets = signals * convictions * actual_rets
    bh_rets = actual_rets

    strat_sharpe = (
        np.mean(strat_rets) / np.std(strat_rets) * np.sqrt(252 / HORIZON)
        if np.std(strat_rets) > 0 else 0.0
    )
    bh_sharpe = (
        np.mean(bh_rets) / np.std(bh_rets) * np.sqrt(252 / HORIZON)
        if np.std(bh_rets) > 0 else 0.0
    )

    pnl_strat = np.sum(strat_rets)
    pnl_bh = np.sum(bh_rets)
    pct_long = np.mean(signals == 1)
    pct_flat = np.mean(signals == 0)

    return {
        "label": label,
        "n_snap": n_snap,
        "strat_sharpe": strat_sharpe,
        "bh_sharpe": bh_sharpe,
        "pnl_strat": pnl_strat,
        "pnl_bh": pnl_bh,
        "pct_long": pct_long,
        "pct_flat": pct_flat,
        "capture": pnl_strat / pnl_bh if abs(pnl_bh) > 1e-8 else 0.0,
    }


for use_filt in [False, True]:
    label = "Filtered" if use_filt else "Smoothed"
    print(f"  Running {label}...", end=" ", flush=True)
    t0 = time.perf_counter()
    res = run_walk_forward(use_filt)
    elapsed = time.perf_counter() - t0
    print(f"done ({elapsed:.0f}s)")

    print(f"    {res['n_snap']} snapshots | "
          f"Long {res['pct_long']:.0%} | Flat {res['pct_flat']:.0%}")
    print(f"    Signal:  Sharpe={res['strat_sharpe']:+.2f}  PnL={res['pnl_strat']:+.1%}")
    print(f"    B&H:     Sharpe={res['bh_sharpe']:+.2f}  PnL={res['pnl_bh']:+.1%}")
    print(f"    Capture: {res['capture']:.0%}")
    print()

print("=" * 80)
print(f"  Output saved to {_out_path}")
print("=" * 80)

_out_file.close()
