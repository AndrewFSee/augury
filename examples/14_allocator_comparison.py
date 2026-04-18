"""Example 14 — Regime allocator vs composite signal vs buy-and-hold.

Compares three approaches across multiple tickers and periods:

  1. **Buy-and-Hold** — 100% invested at all times.
  2. **ForecastSignal** — composite 6-component signal (long/flat).
  3. **RegimeAllocator** — continuous exposure based on regime probability.
  4. **RegimeAllocator + vol target** — adds 15% vol targeting overlay.

For each ticker/period, runs a walk-forward backtest and reports Sharpe,
cumulative PnL, max drawdown, and B&H capture ratio.
"""

from __future__ import annotations

import sys
import time
import warnings

import numpy as np

# ---------------------------------------------------------------------------
# Dual output
# ---------------------------------------------------------------------------
_out_path = "_allocator_comparison_output.txt"
_out_file = open(_out_path, "w", encoding="utf-8")  # noqa: SIM115
_builtin_print = print


def print(*args, **kwargs):  # noqa: A001
    _builtin_print(*args, **kwargs)
    kwargs.pop("file", None)
    _builtin_print(*args, file=_out_file, **kwargs)
    _out_file.flush()


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
TICKERS = ["SPY", "QQQ", "IWM", "DIA", "EFA", "TLT", "GLD"]
PERIODS = [
    ("2015-2019", "2015-01-01", "2020-01-01"),
    ("2020-2022", "2020-01-01", "2023-01-01"),
    ("2023-2026", "2023-01-01", "2026-12-31"),
    ("Full",      "2015-01-01", "2026-12-31"),
]

LOOKBACK = 504
HORIZON = 21
STEP = 21
N_PATHS = 3000

print("=" * 80)
print("  REGIME ALLOCATOR vs SIGNAL vs BUY-AND-HOLD")
print(f"  Tickers: {', '.join(TICKERS)}")
print(f"  Periods: {', '.join(p[0] for p in PERIODS)}")
print(f"  Lookback: {LOOKBACK}d | Horizon: {HORIZON}d | Step: {STEP}d | Paths: {N_PATHS}")
print("=" * 80)
print()

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
from synthquant.data.ingest import DataIngestor
from synthquant.ensemble_forecast import EnsembleForecaster
from synthquant.strategy.forecast_signals import ForecastSignal
from synthquant.strategy.regime_allocator import RegimeAllocator

ingestor = DataIngestor()


# ---------------------------------------------------------------------------
# Walk-forward engine
# ---------------------------------------------------------------------------
def walk_forward(
    returns: np.ndarray,
    mode: str,
) -> dict[str, float]:
    """Run walk-forward and return performance metrics.

    mode: 'bh', 'signal', 'alloc', 'alloc_vol'
    """
    n = len(returns)

    signal_gen = ForecastSignal() if mode == "signal" else None
    allocator = None
    if mode == "alloc":
        allocator = RegimeAllocator(calm_weight=1.0, crisis_weight=0.25)
    elif mode == "alloc_vol":
        allocator = RegimeAllocator(
            calm_weight=1.0, crisis_weight=0.25, vol_target=0.15,
        )

    snap_returns = []
    t = LOOKBACK
    while t + HORIZON <= n:
        actual_fwd = float(np.sum(returns[t : t + HORIZON]))

        if mode == "bh":
            snap_returns.append(actual_fwd)
        else:
            window = returns[t - LOOKBACK : t]
            recent = returns[max(0, t - 63) : t]

            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    fc = EnsembleForecaster(
                        detector="hmm",
                        n_regimes=2,
                        n_paths=N_PATHS,
                        random_state=42,
                        use_filtered=(mode in ("alloc", "alloc_vol")),
                    )
                    fc.fit(window)
                    forecast = fc.forecast(horizon=HORIZON, current_price=100.0)

                if mode == "signal":
                    result = signal_gen.generate(forecast, recent_returns=recent)
                    snap_returns.append(
                        result.signal * result.conviction * actual_fwd
                    )
                else:
                    result = allocator.allocate(forecast, recent_returns=recent)
                    snap_returns.append(result.exposure * actual_fwd)
            except Exception:
                snap_returns.append(0.0)

        t += STEP

    snap_returns = np.array(snap_returns)
    n_snaps = len(snap_returns)

    if n_snaps == 0:
        return {"sharpe": 0.0, "pnl": 0.0, "max_dd": 0.0, "n_snaps": 0}

    # Sharpe (annualised from horizon-step returns)
    mu = np.mean(snap_returns)
    sigma = np.std(snap_returns)
    sharpe = mu / sigma * np.sqrt(252 / HORIZON) if sigma > 0 else 0.0

    # Cumulative PnL (sum of log returns)
    pnl = float(np.sum(snap_returns))

    # Max drawdown on cumulative equity curve
    cum = np.cumsum(snap_returns)
    peak = np.maximum.accumulate(cum)
    dd = cum - peak
    max_dd = float(np.min(dd))

    return {
        "sharpe": sharpe,
        "pnl": pnl,
        "max_dd": max_dd,
        "n_snaps": n_snaps,
    }


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
results_db: list[dict] = []
cell = 0
total_cells = len(TICKERS) * len(PERIODS)

for ticker in TICKERS:
    # Fetch full range once, then slice per period
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
            print(f"  [{cell}/{total_cells}] {ticker} / {period_name}  SKIP (too few rows)")
            continue

        returns = np.log(df["close"] / df["close"].shift(1)).dropna().values
        print(
            f"  [{cell}/{total_cells}] {ticker} / {period_name}  "
            f"({len(returns)} days, ${df['close'].iloc[0]:.2f}-${df['close'].iloc[-1]:.2f})"
        )

        t0 = time.perf_counter()

        # B&H is instant
        bh = walk_forward(returns, "bh")

        # Signal (reuse existing composite approach)
        sig = walk_forward(returns, "signal")

        # Regime Allocator
        alloc = walk_forward(returns, "alloc")

        # Regime Allocator + vol targeting
        alloc_v = walk_forward(returns, "alloc_vol")

        elapsed = time.perf_counter() - t0

        # Capture ratios (vs B&H)
        bh_pnl = bh["pnl"]
        sig_cap = sig["pnl"] / bh_pnl if abs(bh_pnl) > 1e-8 else 0.0
        alloc_cap = alloc["pnl"] / bh_pnl if abs(bh_pnl) > 1e-8 else 0.0
        alloc_v_cap = alloc_v["pnl"] / bh_pnl if abs(bh_pnl) > 1e-8 else 0.0

        print(
            f"    {bh['n_snaps']} snaps | "
            f"B&H: Sharpe={bh['sharpe']:+.2f} PnL={bh['pnl']:+.1%} DD={bh['max_dd']:+.1%} | "
            f"Signal: Sharpe={sig['sharpe']:+.2f} PnL={sig['pnl']:+.1%} Cap={sig_cap:.0%} | "
            f"Alloc: Sharpe={alloc['sharpe']:+.2f} PnL={alloc['pnl']:+.1%} Cap={alloc_cap:.0%} | "
            f"AllocVol: Sharpe={alloc_v['sharpe']:+.2f} PnL={alloc_v['pnl']:+.1%} Cap={alloc_v_cap:.0%}"
            f"  ({elapsed:.0f}s)"
        )

        results_db.append({
            "ticker": ticker,
            "period": period_name,
            "bh_sharpe": bh["sharpe"],
            "bh_pnl": bh["pnl"],
            "bh_dd": bh["max_dd"],
            "sig_sharpe": sig["sharpe"],
            "sig_pnl": sig["pnl"],
            "sig_cap": sig_cap,
            "alloc_sharpe": alloc["sharpe"],
            "alloc_pnl": alloc["pnl"],
            "alloc_cap": alloc_cap,
            "alloc_v_sharpe": alloc_v["sharpe"],
            "alloc_v_pnl": alloc_v["pnl"],
            "alloc_v_cap": alloc_v_cap,
        })

print()

# ---------------------------------------------------------------------------
# Summary tables
# ---------------------------------------------------------------------------
if results_db:
    print("=" * 80)
    print("  SHARPE RATIO COMPARISON")
    print("=" * 80)
    print()
    print(f"  {'Ticker':<8} {'Period':<12} {'B&H':>8} {'Signal':>8} {'Alloc':>8} {'AllocVol':>8}")
    print(f"  {'-'*8} {'-'*12} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for r in results_db:
        print(
            f"  {r['ticker']:<8} {r['period']:<12} "
            f"{r['bh_sharpe']:+8.2f} {r['sig_sharpe']:+8.2f} "
            f"{r['alloc_sharpe']:+8.2f} {r['alloc_v_sharpe']:+8.2f}"
        )

    print()
    print("=" * 80)
    print("  B&H CAPTURE RATIO (PnL strategy / PnL B&H)")
    print("=" * 80)
    print()
    print(f"  {'Ticker':<8} {'Period':<12} {'Signal':>8} {'Alloc':>8} {'AllocVol':>8}")
    print(f"  {'-'*8} {'-'*12} {'-'*8} {'-'*8} {'-'*8}")
    for r in results_db:
        print(
            f"  {r['ticker']:<8} {r['period']:<12} "
            f"{r['sig_cap']:+8.0%} {r['alloc_cap']:+8.0%} {r['alloc_v_cap']:+8.0%}"
        )

    # Aggregates
    print()
    print("=" * 80)
    print("  AGGREGATE STATISTICS")
    print("=" * 80)
    print()

    for label, key_s, key_p, key_c in [
        ("Signal",   "sig_sharpe",     "sig_pnl",     "sig_cap"),
        ("Alloc",    "alloc_sharpe",   "alloc_pnl",   "alloc_cap"),
        ("AllocVol", "alloc_v_sharpe", "alloc_v_pnl", "alloc_v_cap"),
    ]:
        sharpes = [r[key_s] for r in results_db]
        caps = [r[key_c] for r in results_db]
        pnls = [r[key_p] for r in results_db]
        bh_sharpes = [r["bh_sharpe"] for r in results_db]
        deltas = [s - b for s, b in zip(sharpes, bh_sharpes)]
        wins = sum(1 for d in deltas if d > 0)

        print(f"  {label}:")
        print(f"    Avg Sharpe:         {np.mean(sharpes):+.2f}  (B&H: {np.mean(bh_sharpes):+.2f})")
        print(f"    Median Sharpe:      {np.median(sharpes):+.2f}")
        print(f"    Avg Sharpe delta:   {np.mean(deltas):+.2f}")
        print(f"    Beats B&H Sharpe:   {wins}/{len(results_db)}")
        print(f"    Avg capture ratio:  {np.mean(caps):.0%}")
        print(f"    Median capture:     {np.median(caps):.0%}")
        print()

print("=" * 80)
print(f"  Output saved to {_out_path}")
print("=" * 80)

_out_file.close()
