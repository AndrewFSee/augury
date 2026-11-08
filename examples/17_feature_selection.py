"""Example 17 — HMM Feature Selection Experiment.

Tests different feature subsets to find which combination best identifies
regimes for the regime allocator.  Evaluates each subset via walk-forward
backtesting across multiple tickers and periods.

Metrics:
  - Sharpe ratio (risk-adjusted return)
  - B&H capture (upside participation)
  - Max drawdown (tail protection)
  - Crisis detection rate (regime sensitivity)

The experiment is structured as:
  Phase 1: Individual feature value — test each feature alone + returns
  Phase 2: Feature combinations — test promising pairs/triples from Phase 1
  Phase 3: Summary and ranking
"""

from __future__ import annotations

import itertools
import time
import warnings

import numpy as np

# ---------------------------------------------------------------------------
_out_path = "_feature_selection_output.txt"
_out_file = open(_out_path, "w", encoding="utf-8")  # noqa: SIM115
_builtin_print = print


def print(*args, **kwargs):  # noqa: A001
    _builtin_print(*args, **kwargs)
    kwargs.pop("file", None)
    _builtin_print(*args, file=_out_file, **kwargs)
    _out_file.flush()


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
TICKERS = ["SPY", "QQQ", "IWM", "DIA", "EFA", "TLT", "GLD"]
PERIODS = [
    ("2015-2019", "2015-01-01", "2020-01-01"),
    ("2020-2022", "2020-01-01", "2023-01-01"),
    ("Full",      "2015-01-01", "2026-12-31"),
]

LOOKBACK = 504
HORIZON = 21
STEP = 21
N_PATHS = 3000

# All available features from the registry
from synthquant.regime.features import ALL_FEATURE_NAMES

print("=" * 80)
print("  HMM FEATURE SELECTION EXPERIMENT")
print("=" * 80)
print(f"  Tickers:  {', '.join(TICKERS)}")
print(f"  Periods:  {', '.join(p[0] for p in PERIODS)}")
print(f"  Lookback={LOOKBACK} | Horizon={HORIZON} | Step={STEP} | Paths={N_PATHS}")
print(f"  Available features ({len(ALL_FEATURE_NAMES)}):")
for f in ALL_FEATURE_NAMES:
    print(f"    - {f}")
print()

# ---------------------------------------------------------------------------
from synthquant.data.ingest import DataIngestor
from synthquant.ensemble_forecast import EnsembleForecaster
from synthquant.strategy.regime_allocator import RegimeAllocator

ingestor = DataIngestor()


# ---------------------------------------------------------------------------
def walk_forward(
    returns: np.ndarray,
    feature_names: list[str] | None,
) -> dict[str, float]:
    """Walk-forward backtest with HMM features.

    feature_names=None means univariate (no features).
    """
    n = len(returns)
    use_features = feature_names is not None

    allocator = RegimeAllocator(calm_weight=1.0, crisis_weight=0.25)

    snap_returns = []
    snap_exposures = []
    snap_regimes = []
    t = LOOKBACK

    while t + HORIZON <= n:
        actual_fwd = float(np.sum(returns[t : t + HORIZON]))
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
                    feature_names=feature_names,
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
    n_snaps = len(snap_returns)

    if n_snaps == 0:
        return {"sharpe": 0.0, "pnl": 0.0, "max_dd": 0.0,
                "avg_exp": 0.0, "pct_crisis": 0.0, "n_snaps": 0}

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
        "avg_exp": float(np.mean(snap_exposures)),
        "pct_crisis": pct_crisis, "n_snaps": n_snaps,
    }


# ---------------------------------------------------------------------------
# Load all ticker data
# ---------------------------------------------------------------------------
print("Loading data...")
ticker_data: dict[str, dict[str, np.ndarray]] = {}
for ticker in TICKERS:
    try:
        df_full = ingestor.fetch(ticker, start="2014-06-01", end="2026-12-31")
    except Exception as e:
        print(f"  SKIP {ticker}: {e}")
        continue

    ticker_data[ticker] = {}
    for period_name, p_start, p_end in PERIODS:
        mask = (df_full.index >= p_start) & (df_full.index <= p_end)
        df = df_full.loc[mask]
        if len(df) < LOOKBACK + HORIZON + 10:
            continue
        returns = np.log(df["close"] / df["close"].shift(1)).dropna().values
        ticker_data[ticker][period_name] = returns

print(f"  Loaded: {list(ticker_data.keys())}")
avail_cells = sum(len(v) for v in ticker_data.values())
print(f"  Total cells: {avail_cells}")
print()


# ---------------------------------------------------------------------------
def evaluate_feature_set(
    name: str,
    feature_names: list[str] | None,
) -> dict:
    """Run a feature set across all tickers/periods and aggregate."""
    results = []

    for ticker, periods in ticker_data.items():
        for period_name, returns in periods.items():
            # B&H baseline (only need to compute once, cache it)
            bh_pnl = float(np.sum(returns[LOOKBACK:]))

            res = walk_forward(returns, feature_names)

            cap = res["pnl"] / bh_pnl if abs(bh_pnl) > 1e-8 else 0.0

            results.append({
                "ticker": ticker, "period": period_name,
                "sharpe": res["sharpe"], "pnl": res["pnl"],
                "max_dd": res["max_dd"], "cap": cap,
                "avg_exp": res["avg_exp"], "pct_crisis": res["pct_crisis"],
            })

    if not results:
        return {"name": name, "features": feature_names,
                "avg_sharpe": 0.0, "avg_cap": 0.0, "avg_dd": 0.0,
                "avg_crisis": 0.0, "n_cells": 0, "results": []}

    sharpes = [r["sharpe"] for r in results]
    caps = [r["cap"] for r in results]
    dds = [r["max_dd"] for r in results]
    crises = [r["pct_crisis"] for r in results]

    return {
        "name": name,
        "features": feature_names,
        "avg_sharpe": np.mean(sharpes),
        "med_sharpe": np.median(sharpes),
        "avg_cap": np.mean(caps),
        "med_cap": np.median(caps),
        "avg_dd": np.mean(dds),
        "avg_crisis": np.mean(crises),
        "n_cells": len(results),
        "results": results,
    }


# ---------------------------------------------------------------------------
# Phase 1: Baseline + individual feature value
# ---------------------------------------------------------------------------
print("=" * 80)
print("  PHASE 1: INDIVIDUAL FEATURE VALUE")
print("  Testing: returns-only baseline, then each feature paired with returns")
print("=" * 80)
print()

phase1_results: list[dict] = []

# Baseline: univariate (returns only, no multivariate features)
print("  [Baseline] Univariate HMM (returns only, no features)...")
t0 = time.perf_counter()
baseline = evaluate_feature_set("univariate", None)
elapsed = time.perf_counter() - t0
print(f"    Sharpe={baseline['avg_sharpe']:+.3f}  Cap={baseline['avg_cap']:.0%}  "
      f"DD={baseline['avg_dd']:+.1%}  Crisis={baseline['avg_crisis']:.0%}  "
      f"({elapsed:.0f}s)")
phase1_results.append(baseline)

# Returns-only in multivariate mode
print("  [returns] Returns only (MV mode, 1 feature)...")
t0 = time.perf_counter()
ret_only = evaluate_feature_set("returns_only", ["returns"])
elapsed = time.perf_counter() - t0
print(f"    Sharpe={ret_only['avg_sharpe']:+.3f}  Cap={ret_only['avg_cap']:.0%}  "
      f"DD={ret_only['avg_dd']:+.1%}  Crisis={ret_only['avg_crisis']:.0%}  "
      f"({elapsed:.0f}s)")
phase1_results.append(ret_only)

# Each feature paired with returns
CANDIDATE_FEATURES = [f for f in ALL_FEATURE_NAMES if f != "returns"]

for feat in CANDIDATE_FEATURES:
    label = f"returns+{feat}"
    feature_set = ["returns", feat]
    print(f"  [{label}]...")
    t0 = time.perf_counter()
    res = evaluate_feature_set(label, feature_set)
    elapsed = time.perf_counter() - t0
    print(f"    Sharpe={res['avg_sharpe']:+.3f}  Cap={res['avg_cap']:.0%}  "
          f"DD={res['avg_dd']:+.1%}  Crisis={res['avg_crisis']:.0%}  "
          f"({elapsed:.0f}s)")
    phase1_results.append(res)

print()
print("-" * 80)
print("  PHASE 1 RANKING (by Sharpe)")
print("-" * 80)
print(f"  {'Feature Set':<35} {'Sharpe':>8} {'Cap':>8} {'MaxDD':>8} {'Crisis':>8}")
print(f"  {'-'*35} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
for r in sorted(phase1_results, key=lambda x: x["avg_sharpe"], reverse=True):
    print(f"  {r['name']:<35} {r['avg_sharpe']:+8.3f} {r['avg_cap']:>7.0%} "
          f"{r['avg_dd']:+7.1%} {r['avg_crisis']:>7.0%}")
print()

# ---------------------------------------------------------------------------
# Phase 2: Promising combinations (top features from Phase 1)
# ---------------------------------------------------------------------------
print("=" * 80)
print("  PHASE 2: FEATURE COMBINATIONS")
print("  Testing pairs and triples from top Phase 1 features")
print("=" * 80)
print()

# Rank Phase 1 by Sharpe (exclude baseline/returns_only)
paired_results = [r for r in phase1_results
                  if r["name"] not in ("univariate", "returns_only")]
paired_sorted = sorted(paired_results, key=lambda x: x["avg_sharpe"], reverse=True)

# Take top 6 features for combination testing
top_k = 6
top_features = []
for r in paired_sorted[:top_k]:
    # Extract the second feature name (the one paired with returns)
    feat = r["features"][1] if len(r["features"]) > 1 else r["features"][0]
    top_features.append(feat)

print(f"  Top {top_k} features from Phase 1: {top_features}")
print()

phase2_results: list[dict] = []

# Test all pairs of top features (with returns always included)
print("  --- Pairs (returns + 2 features) ---")
for f1, f2 in itertools.combinations(top_features, 2):
    label = f"ret+{f1}+{f2}"
    feature_set = ["returns", f1, f2]
    print(f"  [{label}]...")
    t0 = time.perf_counter()
    res = evaluate_feature_set(label, feature_set)
    elapsed = time.perf_counter() - t0
    print(f"    Sharpe={res['avg_sharpe']:+.3f}  Cap={res['avg_cap']:.0%}  "
          f"DD={res['avg_dd']:+.1%}  Crisis={res['avg_crisis']:.0%}  "
          f"({elapsed:.0f}s)")
    phase2_results.append(res)

# Test top 4 triples
print()
print("  --- Triples (returns + 3 features) ---")
top_for_triples = top_features[:4]
for f1, f2, f3 in itertools.combinations(top_for_triples, 3):
    label = f"ret+{f1}+{f2}+{f3}"
    feature_set = ["returns", f1, f2, f3]
    print(f"  [{label}]...")
    t0 = time.perf_counter()
    res = evaluate_feature_set(label, feature_set)
    elapsed = time.perf_counter() - t0
    print(f"    Sharpe={res['avg_sharpe']:+.3f}  Cap={res['avg_cap']:.0%}  "
          f"DD={res['avg_dd']:+.1%}  Crisis={res['avg_crisis']:.0%}  "
          f"({elapsed:.0f}s)")
    phase2_results.append(res)

# Also test all top features together
print()
print("  --- Kitchen sink (returns + all top features) ---")
label = f"ret+ALL_TOP{top_k}"
feature_set = ["returns"] + top_features
print(f"  [{label}]...")
t0 = time.perf_counter()
res = evaluate_feature_set(label, feature_set)
elapsed = time.perf_counter() - t0
print(f"    Sharpe={res['avg_sharpe']:+.3f}  Cap={res['avg_cap']:.0%}  "
      f"DD={res['avg_dd']:+.1%}  Crisis={res['avg_crisis']:.0%}  "
      f"({elapsed:.0f}s)")
phase2_results.append(res)

print()
print("-" * 80)
print("  PHASE 2 RANKING (by Sharpe)")
print("-" * 80)
print(f"  {'Feature Set':<45} {'Sharpe':>8} {'Cap':>8} {'MaxDD':>8} {'Crisis':>8}")
print(f"  {'-'*45} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
for r in sorted(phase2_results, key=lambda x: x["avg_sharpe"], reverse=True):
    print(f"  {r['name']:<45} {r['avg_sharpe']:+8.3f} {r['avg_cap']:>7.0%} "
          f"{r['avg_dd']:+7.1%} {r['avg_crisis']:>7.0%}")

# ---------------------------------------------------------------------------
# Phase 3: Grand summary
# ---------------------------------------------------------------------------
print()
print("=" * 80)
print("  GRAND SUMMARY — ALL TESTED COMBINATIONS")
print("=" * 80)
print()

all_results = phase1_results + phase2_results
all_sorted = sorted(all_results, key=lambda x: x["avg_sharpe"], reverse=True)

print(f"  {'Rank':<5} {'Feature Set':<45} {'Sharpe':>8} {'MedSh':>8} {'Cap':>8} "
      f"{'MedCap':>8} {'MaxDD':>8} {'Crisis':>8}")
print(f"  {'-'*5} {'-'*45} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
for rank, r in enumerate(all_sorted, 1):
    print(f"  {rank:<5} {r['name']:<45} {r['avg_sharpe']:+8.3f} "
          f"{r.get('med_sharpe', 0):+8.3f} {r['avg_cap']:>7.0%} "
          f"{r.get('med_cap', 0):>7.0%} {r['avg_dd']:+7.1%} "
          f"{r['avg_crisis']:>7.0%}")

# Composite score: weighted combination emphasizing Sharpe + capture
print()
print("-" * 80)
print("  COMPOSITE SCORE = 0.4 × norm(Sharpe) + 0.3 × norm(Capture) + 0.3 × norm(-DD)")
print("-" * 80)

# Normalize each metric to [0, 1] range
sharpes = np.array([r["avg_sharpe"] for r in all_sorted])
caps = np.array([r["avg_cap"] for r in all_sorted])
dds = np.array([-r["avg_dd"] for r in all_sorted])  # negate so higher is better

def _normalize(arr):
    rng = arr.max() - arr.min()
    return (arr - arr.min()) / rng if rng > 1e-12 else np.zeros_like(arr)

n_sharpe = _normalize(sharpes)
n_cap = _normalize(caps)
n_dd = _normalize(dds)

composite = 0.4 * n_sharpe + 0.3 * n_cap + 0.3 * n_dd

ranked = sorted(zip(composite, all_sorted, strict=False), key=lambda x: x[0], reverse=True)

print(f"  {'Rank':<5} {'Feature Set':<45} {'Score':>8} {'Sharpe':>8} {'Cap':>8} {'MaxDD':>8}")
print(f"  {'-'*5} {'-'*45} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
for rank, (score, r) in enumerate(ranked, 1):
    print(f"  {rank:<5} {r['name']:<45} {score:8.3f} {r['avg_sharpe']:+8.3f} "
          f"{r['avg_cap']:>7.0%} {r['avg_dd']:+7.1%}")

# Per-ticker detail for top 3
print()
print("=" * 80)
print("  PER-TICKER DETAIL — TOP 3 FEATURE SETS")
print("=" * 80)

for idx, (score, r) in enumerate(ranked[:3], 1):
    print(f"\n  #{idx} {r['name']} (score={score:.3f})")
    print(f"  Features: {r['features']}")
    print(f"  {'Ticker':<8} {'Period':<12} {'Sharpe':>8} {'Cap':>8} {'MaxDD':>8} {'Crisis':>8}")
    print(f"  {'-'*8} {'-'*12} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for cell in r.get("results", []):
        print(f"  {cell['ticker']:<8} {cell['period']:<12} "
              f"{cell['sharpe']:+8.2f} {cell['cap']:>7.0%} "
              f"{cell['max_dd']:+7.1%} {cell['pct_crisis']:>7.0%}")

print()
print("=" * 80)
print("  RECOMMENDATION")
print("=" * 80)
if ranked:
    best_score, best = ranked[0]
    print(f"  Best feature set: {best['name']}")
    print(f"  Features: {best['features']}")
    print(f"  Composite score: {best_score:.3f}")
    print(f"  Avg Sharpe: {best['avg_sharpe']:+.3f}")
    print(f"  Avg Capture: {best['avg_cap']:.0%}")
    print(f"  Avg MaxDD: {best['avg_dd']:+.1%}")
    print(f"  Crisis rate: {best['avg_crisis']:.0%}")

print()
print("Done.")
_out_file.close()
