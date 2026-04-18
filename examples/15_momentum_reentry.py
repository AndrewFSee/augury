"""Example 15 — Momentum re-entry comparison.

Isolates the impact of the momentum re-entry feature by comparing:

  1. **Buy-and-Hold** — 100% invested at all times.
  2. **Alloc (no momentum)** — regime allocator with reentry disabled.
  3. **Alloc + Momentum** — regime allocator with momentum re-entry (default).
  4. **Alloc + Mom + Vol** — adds 15% vol targeting on top.

The hypothesis: plain regime allocation cuts exposure during crashes (good)
but keeps it low during V-shaped recoveries because the HMM still sees
elevated vol.  Momentum re-entry detects the positive trailing returns
and ramps exposure back up before the HMM switches back to calm.
"""

from __future__ import annotations

import sys
import time
import warnings

import numpy as np

# ---------------------------------------------------------------------------
# Dual output
# ---------------------------------------------------------------------------
_out_path = "_momentum_reentry_output.txt"
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
print("  MOMENTUM RE-ENTRY COMPARISON")
print(f"  Tickers: {', '.join(TICKERS)}")
print(f"  Periods: {', '.join(p[0] for p in PERIODS)}")
print(f"  Lookback={LOOKBACK} | Horizon={HORIZON} | Step={STEP} | Paths={N_PATHS}")
print(f"  Reentry: lookback=20, scale=0.04 (4% trailing = full re-entry)")
print("=" * 80)
print()

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
from synthquant.data.ingest import DataIngestor
from synthquant.ensemble_forecast import EnsembleForecaster
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

    mode: 'bh', 'alloc_plain', 'alloc_mom', 'alloc_mom_vol'
    """
    n = len(returns)

    allocator = None
    if mode == "alloc_plain":
        allocator = RegimeAllocator(
            calm_weight=1.0, crisis_weight=0.25, reentry_lookback=0,
        )
    elif mode == "alloc_mom":
        allocator = RegimeAllocator(
            calm_weight=1.0, crisis_weight=0.25,
            reentry_lookback=20, reentry_scale=0.04,
        )
    elif mode == "alloc_mom_vol":
        allocator = RegimeAllocator(
            calm_weight=1.0, crisis_weight=0.25,
            reentry_lookback=20, reentry_scale=0.04,
            vol_target=0.15,
        )

    snap_returns = []
    snap_exposures = []
    snap_boosts = []
    t = LOOKBACK
    while t + HORIZON <= n:
        actual_fwd = float(np.sum(returns[t : t + HORIZON]))

        if mode == "bh":
            snap_returns.append(actual_fwd)
            snap_exposures.append(1.0)
            snap_boosts.append(0.0)
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
                        use_filtered=True,
                    )
                    fc.fit(window)
                    forecast = fc.forecast(horizon=HORIZON, current_price=100.0)

                result = allocator.allocate(forecast, recent_returns=recent)
                snap_returns.append(result.exposure * actual_fwd)
                snap_exposures.append(result.exposure)
                snap_boosts.append(result.momentum_boost)
            except Exception:
                snap_returns.append(0.0)
                snap_exposures.append(0.0)
                snap_boosts.append(0.0)

        t += STEP

    snap_returns = np.array(snap_returns)
    snap_exposures = np.array(snap_exposures)
    snap_boosts = np.array(snap_boosts)
    n_snaps = len(snap_returns)

    if n_snaps == 0:
        return {
            "sharpe": 0.0, "pnl": 0.0, "max_dd": 0.0,
            "n_snaps": 0, "avg_exp": 0.0, "avg_boost": 0.0,
            "pct_boosted": 0.0,
        }

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
        "avg_exp": float(np.mean(snap_exposures)),
        "avg_boost": float(np.mean(snap_boosts)),
        "pct_boosted": float(np.mean(snap_boosts > 0)),
    }


# ---------------------------------------------------------------------------
# Main loop
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

        returns = np.log(df["close"] / df["close"].shift(1)).dropna().values
        print(
            f"  [{cell}/{total_cells}] {ticker} / {period_name}  "
            f"({len(returns)} days)"
        )

        t0 = time.perf_counter()

        bh = walk_forward(returns, "bh")
        plain = walk_forward(returns, "alloc_plain")
        mom = walk_forward(returns, "alloc_mom")
        mom_vol = walk_forward(returns, "alloc_mom_vol")

        elapsed = time.perf_counter() - t0
        bh_pnl = bh["pnl"]

        def cap(x: dict) -> float:
            return x["pnl"] / bh_pnl if abs(bh_pnl) > 1e-8 else 0.0

        print(
            f"    B&H:   Sharpe={bh['sharpe']:+.2f}  PnL={bh['pnl']:+.1%}"
        )
        print(
            f"    Plain: Sharpe={plain['sharpe']:+.2f}  PnL={plain['pnl']:+.1%}  "
            f"Cap={cap(plain):.0%}  AvgExp={plain['avg_exp']:.2f}"
        )
        print(
            f"    Mom:   Sharpe={mom['sharpe']:+.2f}  PnL={mom['pnl']:+.1%}  "
            f"Cap={cap(mom):.0%}  AvgExp={mom['avg_exp']:.2f}  "
            f"AvgBoost={mom['avg_boost']:.3f}  Boosted={mom['pct_boosted']:.0%}"
        )
        print(
            f"    MomVol:Sharpe={mom_vol['sharpe']:+.2f}  PnL={mom_vol['pnl']:+.1%}  "
            f"Cap={cap(mom_vol):.0%}  AvgExp={mom_vol['avg_exp']:.2f}"
            f"  ({elapsed:.0f}s)"
        )

        results_db.append({
            "ticker": ticker,
            "period": period_name,
            "bh_sharpe": bh["sharpe"], "bh_pnl": bh["pnl"], "bh_dd": bh["max_dd"],
            "plain_sharpe": plain["sharpe"], "plain_pnl": plain["pnl"],
            "plain_cap": cap(plain), "plain_dd": plain["max_dd"],
            "mom_sharpe": mom["sharpe"], "mom_pnl": mom["pnl"],
            "mom_cap": cap(mom), "mom_dd": mom["max_dd"],
            "mom_avg_exp": mom["avg_exp"], "mom_avg_boost": mom["avg_boost"],
            "mom_pct_boosted": mom["pct_boosted"],
            "mv_sharpe": mom_vol["sharpe"], "mv_pnl": mom_vol["pnl"],
            "mv_cap": cap(mom_vol), "mv_dd": mom_vol["max_dd"],
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
    print(f"  {'Ticker':<8} {'Period':<12} {'B&H':>8} {'Plain':>8} {'Mom':>8} {'MomVol':>8}")
    print(f"  {'-'*8} {'-'*12} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for r in results_db:
        print(
            f"  {r['ticker']:<8} {r['period']:<12} "
            f"{r['bh_sharpe']:+8.2f} {r['plain_sharpe']:+8.2f} "
            f"{r['mom_sharpe']:+8.2f} {r['mv_sharpe']:+8.2f}"
        )

    print()
    print("=" * 80)
    print("  B&H CAPTURE RATIO")
    print("=" * 80)
    print()
    print(f"  {'Ticker':<8} {'Period':<12} {'Plain':>8} {'Mom':>8} {'MomVol':>8}")
    print(f"  {'-'*8} {'-'*12} {'-'*8} {'-'*8} {'-'*8}")
    for r in results_db:
        print(
            f"  {r['ticker']:<8} {r['period']:<12} "
            f"{r['plain_cap']:>7.0%} {r['mom_cap']:>7.0%} {r['mv_cap']:>7.0%}"
        )

    print()
    print("=" * 80)
    print("  MAX DRAWDOWN")
    print("=" * 80)
    print()
    print(f"  {'Ticker':<8} {'Period':<12} {'B&H':>8} {'Plain':>8} {'Mom':>8} {'MomVol':>8}")
    print(f"  {'-'*8} {'-'*12} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for r in results_db:
        print(
            f"  {r['ticker']:<8} {r['period']:<12} "
            f"{r['bh_dd']:+7.1%} {r['plain_dd']:+7.1%} "
            f"{r['mom_dd']:+7.1%} {r['mv_dd']:+7.1%}"
        )

    # Aggregates
    print()
    print("=" * 80)
    print("  AGGREGATE SUMMARY")
    print("=" * 80)
    print()
    for label, prefix in [("Plain", "plain"), ("Mom", "mom"), ("MomVol", "mv")]:
        sharpes = [r[f"{prefix}_sharpe"] for r in results_db]
        caps = [r[f"{prefix}_cap"] for r in results_db]
        dds = [r[f"{prefix}_dd"] for r in results_db]
        bh_sharpes = [r["bh_sharpe"] for r in results_db]
        beats = sum(1 for s, b in zip(sharpes, bh_sharpes) if s >= b)

        print(f"  {label}:")
        print(f"    Avg Sharpe:  {np.mean(sharpes):+.2f}  (B&H: {np.mean(bh_sharpes):+.2f})")
        print(f"    Med Sharpe:  {np.median(sharpes):+.2f}")
        print(f"    Avg Capture: {np.mean(caps):.0%}")
        print(f"    Med Capture: {np.median(caps):.0%}")
        print(f"    Avg MaxDD:   {np.mean(dds):+.1%}  (B&H: {np.mean([r['bh_dd'] for r in results_db]):+.1%})")
        print(f"    Beats B&H:   {beats}/{len(results_db)}")
        print()

    # Momentum-specific stats
    print("  Momentum Re-entry Stats:")
    avg_boost = np.mean([r["mom_avg_boost"] for r in results_db])
    avg_pct = np.mean([r["mom_pct_boosted"] for r in results_db])
    avg_exp_plain = np.mean([r.get("plain_cap", 0) for r in results_db])
    avg_exp_mom = np.mean([r["mom_avg_exp"] for r in results_db])
    print(f"    Avg momentum boost:  {avg_boost:.4f}")
    print(f"    Pct snaps boosted:   {avg_pct:.0%}")
    print(f"    Avg exposure (Mom):  {avg_exp_mom:.2f}")
    print()

print("Done.")
_out_file.close()
