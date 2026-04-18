"""Example 16 — Multivariate HMM feature comparison.

Compares the regime allocator with:

  1. **B&H** — 100% buy-and-hold.
  2. **Univariate** — HMM trained on raw returns only.
  3. **Multivariate** — HMM trained on [returns, signed_vol, momentum].
  4. **MV + Momentum re-entry** — multivariate HMM + momentum re-entry.

The hypothesis: the multivariate HMM can distinguish crashes (high vol,
negative momentum) from recoveries (high vol, positive momentum), so it
should switch back to full investment faster during V-shaped bounces.
"""

from __future__ import annotations

import sys
import time
import warnings

import numpy as np

# ---------------------------------------------------------------------------
_out_path = "_mv_hmm_output.txt"
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
    ("Full",      "2015-01-01", "2026-12-31"),
]

LOOKBACK = 504
HORIZON = 21
STEP = 21
N_PATHS = 3000

print("=" * 80)
print("  MULTIVARIATE HMM FEATURE COMPARISON")
print(f"  Tickers: {', '.join(TICKERS)}")
print(f"  Periods: {', '.join(p[0] for p in PERIODS)}")
print(f"  Lookback={LOOKBACK} | Horizon={HORIZON} | Step={STEP} | Paths={N_PATHS}")
print("=" * 80)
print()

# ---------------------------------------------------------------------------
from synthquant.data.ingest import DataIngestor
from synthquant.ensemble_forecast import EnsembleForecaster
from synthquant.strategy.regime_allocator import RegimeAllocator

ingestor = DataIngestor()


# ---------------------------------------------------------------------------
def walk_forward(returns: np.ndarray, mode: str) -> dict[str, float]:
    """Walk-forward backtest.

    mode: 'bh', 'uni', 'mv', 'mv_mom'
    """
    n = len(returns)

    use_features = mode in ("mv", "mv_mom")
    use_mom = mode == "mv_mom"

    allocator = None
    if mode != "bh":
        allocator = RegimeAllocator(
            calm_weight=1.0, crisis_weight=0.25,
            reentry_lookback=20 if use_mom else 0,
            reentry_scale=0.04,
        )

    snap_returns = []
    snap_exposures = []
    snap_regimes = []
    t = LOOKBACK
    while t + HORIZON <= n:
        actual_fwd = float(np.sum(returns[t : t + HORIZON]))

        if mode == "bh":
            snap_returns.append(actual_fwd)
            snap_exposures.append(1.0)
            snap_regimes.append(0)
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
                        use_features=use_features,
                    )
                    fc.fit(window)
                    forecast = fc.forecast(horizon=HORIZON, current_price=100.0)

                result = allocator.allocate(forecast, recent_returns=recent)
                snap_returns.append(result.exposure * actual_fwd)
                snap_exposures.append(result.exposure)
                snap_regimes.append(result.regime)
            except Exception:
                snap_returns.append(0.0)
                snap_exposures.append(0.0)
                snap_regimes.append(-1)

        t += STEP

    snap_returns = np.array(snap_returns)
    snap_exposures = np.array(snap_exposures)
    n_snaps = len(snap_returns)

    if n_snaps == 0:
        return {"sharpe": 0.0, "pnl": 0.0, "max_dd": 0.0,
                "n_snaps": 0, "avg_exp": 0.0, "pct_crisis": 0.0}

    mu = np.mean(snap_returns)
    sigma = np.std(snap_returns)
    sharpe = mu / sigma * np.sqrt(252 / HORIZON) if sigma > 0 else 0.0

    pnl = float(np.sum(snap_returns))
    cum = np.cumsum(snap_returns)
    peak = np.maximum.accumulate(cum)
    max_dd = float(np.min(cum - peak))

    snap_regimes = np.array(snap_regimes)
    pct_crisis = float(np.mean(snap_regimes == 1)) if n_snaps > 0 else 0.0

    return {
        "sharpe": sharpe, "pnl": pnl, "max_dd": max_dd,
        "n_snaps": n_snaps, "avg_exp": float(np.mean(snap_exposures)),
        "pct_crisis": pct_crisis,
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

        returns = np.log(df["close"] / df["close"].shift(1)).dropna().values
        print(f"  [{cell}/{total_cells}] {ticker} / {period_name}  ({len(returns)} days)")

        t0 = time.perf_counter()

        bh = walk_forward(returns, "bh")
        uni = walk_forward(returns, "uni")
        mv = walk_forward(returns, "mv")
        mv_mom = walk_forward(returns, "mv_mom")

        elapsed = time.perf_counter() - t0
        bh_pnl = bh["pnl"]

        def cap(x: dict) -> float:
            return x["pnl"] / bh_pnl if abs(bh_pnl) > 1e-8 else 0.0

        print(f"    B&H:    Sharpe={bh['sharpe']:+.2f}  PnL={bh['pnl']:+.1%}")
        print(
            f"    Uni:    Sharpe={uni['sharpe']:+.2f}  PnL={uni['pnl']:+.1%}  "
            f"Cap={cap(uni):.0%}  AvgExp={uni['avg_exp']:.2f}  "
            f"Crisis={uni['pct_crisis']:.0%}"
        )
        print(
            f"    MV:     Sharpe={mv['sharpe']:+.2f}  PnL={mv['pnl']:+.1%}  "
            f"Cap={cap(mv):.0%}  AvgExp={mv['avg_exp']:.2f}  "
            f"Crisis={mv['pct_crisis']:.0%}"
        )
        print(
            f"    MV+Mom: Sharpe={mv_mom['sharpe']:+.2f}  PnL={mv_mom['pnl']:+.1%}  "
            f"Cap={cap(mv_mom):.0%}  AvgExp={mv_mom['avg_exp']:.2f}  "
            f"Crisis={mv_mom['pct_crisis']:.0%}"
            f"  ({elapsed:.0f}s)"
        )

        results_db.append({
            "ticker": ticker, "period": period_name,
            "bh_sharpe": bh["sharpe"], "bh_pnl": bh["pnl"], "bh_dd": bh["max_dd"],
            "uni_sharpe": uni["sharpe"], "uni_pnl": uni["pnl"],
            "uni_cap": cap(uni), "uni_dd": uni["max_dd"],
            "uni_crisis": uni["pct_crisis"],
            "mv_sharpe": mv["sharpe"], "mv_pnl": mv["pnl"],
            "mv_cap": cap(mv), "mv_dd": mv["max_dd"],
            "mv_crisis": mv["pct_crisis"],
            "mvm_sharpe": mv_mom["sharpe"], "mvm_pnl": mv_mom["pnl"],
            "mvm_cap": cap(mv_mom), "mvm_dd": mv_mom["max_dd"],
            "mvm_crisis": mv_mom["pct_crisis"],
        })

print()

# ---------------------------------------------------------------------------
if results_db:
    print("=" * 80)
    print("  SHARPE RATIO COMPARISON")
    print("=" * 80)
    print()
    print(f"  {'Ticker':<8} {'Period':<12} {'B&H':>8} {'Uni':>8} {'MV':>8} {'MV+Mom':>8}")
    print(f"  {'-'*8} {'-'*12} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for r in results_db:
        print(
            f"  {r['ticker']:<8} {r['period']:<12} "
            f"{r['bh_sharpe']:+8.2f} {r['uni_sharpe']:+8.2f} "
            f"{r['mv_sharpe']:+8.2f} {r['mvm_sharpe']:+8.2f}"
        )

    print()
    print("=" * 80)
    print("  B&H CAPTURE RATIO")
    print("=" * 80)
    print()
    print(f"  {'Ticker':<8} {'Period':<12} {'Uni':>8} {'MV':>8} {'MV+Mom':>8}")
    print(f"  {'-'*8} {'-'*12} {'-'*8} {'-'*8} {'-'*8}")
    for r in results_db:
        print(
            f"  {r['ticker']:<8} {r['period']:<12} "
            f"{r['uni_cap']:>7.0%} {r['mv_cap']:>7.0%} {r['mvm_cap']:>7.0%}"
        )

    print()
    print("=" * 80)
    print("  CRISIS DETECTION RATE (pct of snaps in crisis regime)")
    print("=" * 80)
    print()
    print(f"  {'Ticker':<8} {'Period':<12} {'Uni':>8} {'MV':>8} {'MV+Mom':>8}")
    print(f"  {'-'*8} {'-'*12} {'-'*8} {'-'*8} {'-'*8}")
    for r in results_db:
        print(
            f"  {r['ticker']:<8} {r['period']:<12} "
            f"{r['uni_crisis']:>7.0%} {r['mv_crisis']:>7.0%} {r['mvm_crisis']:>7.0%}"
        )

    print()
    print("=" * 80)
    print("  MAX DRAWDOWN")
    print("=" * 80)
    print()
    print(f"  {'Ticker':<8} {'Period':<12} {'B&H':>8} {'Uni':>8} {'MV':>8} {'MV+Mom':>8}")
    print(f"  {'-'*8} {'-'*12} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for r in results_db:
        print(
            f"  {r['ticker']:<8} {r['period']:<12} "
            f"{r['bh_dd']:+7.1%} {r['uni_dd']:+7.1%} "
            f"{r['mv_dd']:+7.1%} {r['mvm_dd']:+7.1%}"
        )

    # Aggregates
    print()
    print("=" * 80)
    print("  AGGREGATE SUMMARY")
    print("=" * 80)
    print()
    for label, prefix in [("Univariate", "uni"), ("Multivariate", "mv"),
                           ("MV+Momentum", "mvm")]:
        sharpes = [r[f"{prefix}_sharpe"] for r in results_db]
        caps = [r[f"{prefix}_cap"] for r in results_db]
        dds = [r[f"{prefix}_dd"] for r in results_db]
        crises = [r[f"{prefix}_crisis"] for r in results_db]
        bh_sharpes = [r["bh_sharpe"] for r in results_db]
        beats = sum(1 for s, b in zip(sharpes, bh_sharpes) if s >= b)

        print(f"  {label}:")
        print(f"    Avg Sharpe:     {np.mean(sharpes):+.2f}  (B&H: {np.mean(bh_sharpes):+.2f})")
        print(f"    Med Sharpe:     {np.median(sharpes):+.2f}")
        print(f"    Avg Capture:    {np.mean(caps):.0%}")
        print(f"    Med Capture:    {np.median(caps):.0%}")
        print(f"    Avg MaxDD:      {np.mean(dds):+.1%}  (B&H: {np.mean([r['bh_dd'] for r in results_db]):+.1%})")
        print(f"    Avg Crisis Pct: {np.mean(crises):.0%}")
        print(f"    Beats B&H:      {beats}/{len(results_db)}")
        print()

print("Done.")
_out_file.close()
