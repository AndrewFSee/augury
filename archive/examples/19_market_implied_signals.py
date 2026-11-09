"""Example 19 — Market-Implied Signals: OHLCV vs OHLCV + Options/Credit Data.

Compares five strategies across multiple tickers and time periods:

  1. **B&H** — 100% buy-and-hold.
  2. **Regime-only** — :class:`RegimeAllocator` (reactive, no predictive layer).
  3. **OHLCV-only** — :class:`PredictiveAllocator` with the 8 OHLCV-derived
     signals (vol term structure, RSI, range expansion, etc.).
  4. **OHLCV + Cross-asset** — adds TLT-based flight-to-safety signal.
  5. **OHLCV + Market-implied** — adds VIX term structure, SKEW,
     credit spread (HYG/IEF), and MOVE index from options/credit markets.

The hypothesis: market-implied data encodes the *forward expectations* of
options and credit traders, providing genuinely predictive (not backward-
looking) information that OHLCV signals cannot capture.
"""

from __future__ import annotations

import time
import warnings

import numpy as np

# ---------------------------------------------------------------------------
_out_path = "_market_implied_output.txt"
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
print("  MARKET-IMPLIED SIGNALS: OHLCV vs OHLCV + OPTIONS/CREDIT DATA")
print(f"  Tickers: {', '.join(TICKERS)}")
print(f"  Periods: {', '.join(p[0] for p in PERIODS)}")
print(f"  Lookback={LOOKBACK} | Horizon={HORIZON} | Step={STEP} | Paths={N_PATHS}")
print("=" * 80)
print()

# ---------------------------------------------------------------------------
from synthquant.data.ingest import DataIngestor
from synthquant.data.market_data import MarketDataFetcher
from synthquant.ensemble_forecast import EnsembleForecaster
from synthquant.strategy.predictive_allocator import PredictiveAllocator
from synthquant.strategy.regime_allocator import RegimeAllocator

ingestor = DataIngestor()

# Pre-fetch TLT for cross-asset signals
print("Fetching TLT for cross-asset signals...")
try:
    tlt_df = ingestor.fetch("TLT", start="2014-06-01", end="2026-12-31")
    tlt_returns = np.log(tlt_df["close"] / tlt_df["close"].shift(1)).dropna().values
    tlt_index = tlt_df.index[1:]
    print(f"  TLT: {len(tlt_returns)} days loaded")
except Exception as e:
    print(f"  TLT fetch failed: {e} — cross-asset signals disabled")
    tlt_returns = None
    tlt_index = None

# Pre-fetch market-implied data (VIX, VIX3M, SKEW, MOVE, HYG, IEF)
print("Fetching market-implied data (VIX, VIX3M, SKEW, MOVE, HYG, IEF)...")
try:
    market_fetcher = MarketDataFetcher(start="2014-06-01", end="2026-12-31")
    print(f"  Available series: {market_fetcher.available_series}")
except Exception as e:
    print(f"  Market data fetch failed: {e}")
    market_fetcher = None
print()


# ---------------------------------------------------------------------------
def walk_forward(
    df,
    mode: str,
    tlt_rets: np.ndarray | None = None,
    tlt_idx=None,
    mkt_fetcher: MarketDataFetcher | None = None,
) -> dict[str, float]:
    """Walk-forward backtest with full OHLCV + market-implied data.

    mode: 'bh', 'regime', 'ohlcv', 'ohlcv_xa', 'market_implied'
    """
    close_arr = df["close"].values
    high_arr = df["high"].values
    low_arr = df["low"].values
    volume_arr = df["volume"].values.astype(float)
    returns = np.log(df["close"] / df["close"].shift(1)).dropna().values
    n = len(returns)

    close_aligned = close_arr[1:]
    high_aligned = high_arr[1:]
    low_aligned = low_arr[1:]
    volume_aligned = volume_arr[1:]

    regime_allocator = RegimeAllocator(
        calm_weight=1.0, crisis_weight=0.25,
        reentry_lookback=0,
    )
    pred_allocator = PredictiveAllocator(
        calm_weight=1.0, crisis_weight=0.25,
        predictive_strength=0.5,
    )
    pred_xa_allocator = PredictiveAllocator(
        calm_weight=1.0, crisis_weight=0.25,
        predictive_strength=0.5,
    )
    mkt_allocator = PredictiveAllocator(
        calm_weight=1.0, crisis_weight=0.25,
        predictive_strength=0.5,
    )

    snap_returns = []
    snap_exposures = []
    snap_signals = []
    snap_regimes = []
    t = LOOKBACK

    eq_dates = df.index[1:]  # aligned with returns

    while t + HORIZON <= n:
        actual_fwd = float(np.sum(returns[t : t + HORIZON]))

        if mode == "bh":
            snap_returns.append(actual_fwd)
            snap_exposures.append(1.0)
            snap_signals.append(0.0)
            snap_regimes.append(0)
        else:
            window_returns = returns[t - LOOKBACK : t]
            window_close = close_aligned[t - LOOKBACK : t]
            window_high = high_aligned[t - LOOKBACK : t]
            window_low = low_aligned[t - LOOKBACK : t]
            window_vol = volume_aligned[t - LOOKBACK : t]

            # Get TLT returns aligned to same dates if available
            safe_haven = None
            if mode in ("ohlcv_xa", "market_implied") and tlt_rets is not None and tlt_idx is not None:
                eq_start_date = eq_dates[t - LOOKBACK]
                eq_end_date = eq_dates[t - 1]
                tlt_mask = (tlt_idx >= eq_start_date) & (tlt_idx <= eq_end_date)
                safe_haven_window = tlt_rets[tlt_mask.values if hasattr(tlt_mask, 'values') else tlt_mask]
                if len(safe_haven_window) >= 63:
                    safe_haven = safe_haven_window

            # Get market-implied snapshot aligned to window dates
            mkt_snapshot = None
            if mode == "market_implied" and mkt_fetcher is not None:
                window_dates = eq_dates[t - LOOKBACK : t]
                mkt_snapshot = mkt_fetcher.slice_by_index(window_dates)

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
                    forecast = fc.forecast(horizon=HORIZON, current_price=100.0)

                if mode == "regime":
                    result = regime_allocator.allocate(
                        forecast, recent_returns=window_returns[-63:]
                    )
                    snap_returns.append(result.exposure * actual_fwd)
                    snap_exposures.append(result.exposure)
                    snap_signals.append(0.0)
                    snap_regimes.append(result.regime)
                elif mode == "ohlcv":
                    result = pred_allocator.allocate(
                        forecast,
                        close=window_close,
                        returns=window_returns,
                        high=window_high,
                        low=window_low,
                        volume=window_vol,
                    )
                    snap_returns.append(result.exposure * actual_fwd)
                    snap_exposures.append(result.exposure)
                    snap_signals.append(result.signal.composite)
                    snap_regimes.append(result.regime)
                elif mode == "ohlcv_xa":
                    result = pred_xa_allocator.allocate(
                        forecast,
                        close=window_close,
                        returns=window_returns,
                        high=window_high,
                        low=window_low,
                        volume=window_vol,
                        safe_haven_returns=safe_haven,
                    )
                    snap_returns.append(result.exposure * actual_fwd)
                    snap_exposures.append(result.exposure)
                    snap_signals.append(result.signal.composite)
                    snap_regimes.append(result.regime)
                elif mode == "market_implied":
                    result = mkt_allocator.allocate(
                        forecast,
                        close=window_close,
                        returns=window_returns,
                        high=window_high,
                        low=window_low,
                        volume=window_vol,
                        safe_haven_returns=safe_haven,
                        market_data=mkt_snapshot,
                    )
                    snap_returns.append(result.exposure * actual_fwd)
                    snap_exposures.append(result.exposure)
                    snap_signals.append(result.signal.composite)
                    snap_regimes.append(result.regime)

            except Exception:
                snap_returns.append(0.0)
                snap_exposures.append(0.0)
                snap_signals.append(0.0)
                snap_regimes.append(-1)

        t += STEP

    snap_returns = np.array(snap_returns)
    snap_exposures = np.array(snap_exposures)
    snap_signals = np.array(snap_signals)
    n_snaps = len(snap_returns)

    if n_snaps == 0:
        return {"sharpe": 0.0, "pnl": 0.0, "max_dd": 0.0,
                "n_snaps": 0, "avg_exp": 0.0, "avg_signal": 0.0}

    mu = np.mean(snap_returns)
    sigma = np.std(snap_returns)
    sharpe = mu / sigma * np.sqrt(252 / HORIZON) if sigma > 0 else 0.0

    pnl = float(np.sum(snap_returns))
    cum = np.cumsum(snap_returns)
    peak = np.maximum.accumulate(cum)
    max_dd = float(np.min(cum - peak))

    return {
        "sharpe": sharpe, "pnl": pnl, "max_dd": max_dd,
        "n_snaps": n_snaps, "avg_exp": float(np.mean(snap_exposures)),
        "avg_signal": float(np.mean(snap_signals)),
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

        bh = walk_forward(df, "bh")
        regime = walk_forward(df, "regime")
        ohlcv = walk_forward(df, "ohlcv")
        ohlcv_xa = walk_forward(df, "ohlcv_xa", tlt_returns, tlt_index)
        mkt = walk_forward(df, "market_implied", tlt_returns, tlt_index, market_fetcher)

        elapsed = time.perf_counter() - t0
        bh_pnl = bh["pnl"]

        def cap(x, bh_pnl=bh_pnl):
            return x["pnl"] / bh_pnl if abs(bh_pnl) > 1e-8 else 0.0

        print(f"    B&H:       Sharpe={bh['sharpe']:+.2f}  PnL={bh['pnl']:+.1%}  DD={bh['max_dd']:+.1%}")
        print(
            f"    Regime:    Sharpe={regime['sharpe']:+.2f}  PnL={regime['pnl']:+.1%}  "
            f"Cap={cap(regime):.0%}  AvgExp={regime['avg_exp']:.2f}  DD={regime['max_dd']:+.1%}"
        )
        print(
            f"    OHLCV:     Sharpe={ohlcv['sharpe']:+.2f}  PnL={ohlcv['pnl']:+.1%}  "
            f"Cap={cap(ohlcv):.0%}  AvgExp={ohlcv['avg_exp']:.2f}  "
            f"DD={ohlcv['max_dd']:+.1%}  Sig={ohlcv['avg_signal']:+.3f}"
        )
        print(
            f"    OHLCV+XA:  Sharpe={ohlcv_xa['sharpe']:+.2f}  PnL={ohlcv_xa['pnl']:+.1%}  "
            f"Cap={cap(ohlcv_xa):.0%}  AvgExp={ohlcv_xa['avg_exp']:.2f}  "
            f"DD={ohlcv_xa['max_dd']:+.1%}  Sig={ohlcv_xa['avg_signal']:+.3f}"
        )
        print(
            f"    Mkt-Impl:  Sharpe={mkt['sharpe']:+.2f}  PnL={mkt['pnl']:+.1%}  "
            f"Cap={cap(mkt):.0%}  AvgExp={mkt['avg_exp']:.2f}  "
            f"DD={mkt['max_dd']:+.1%}  Sig={mkt['avg_signal']:+.3f}"
            f"  ({elapsed:.0f}s)"
        )

        results_db.append({
            "ticker": ticker, "period": period_name,
            "bh_sharpe": bh["sharpe"], "bh_pnl": bh["pnl"], "bh_dd": bh["max_dd"],
            "reg_sharpe": regime["sharpe"], "reg_pnl": regime["pnl"],
            "reg_cap": cap(regime), "reg_dd": regime["max_dd"],
            "ohlcv_sharpe": ohlcv["sharpe"], "ohlcv_pnl": ohlcv["pnl"],
            "ohlcv_cap": cap(ohlcv), "ohlcv_dd": ohlcv["max_dd"],
            "ohlcv_sig": ohlcv["avg_signal"],
            "xa_sharpe": ohlcv_xa["sharpe"], "xa_pnl": ohlcv_xa["pnl"],
            "xa_cap": cap(ohlcv_xa), "xa_dd": ohlcv_xa["max_dd"],
            "xa_sig": ohlcv_xa["avg_signal"],
            "mkt_sharpe": mkt["sharpe"], "mkt_pnl": mkt["pnl"],
            "mkt_cap": cap(mkt), "mkt_dd": mkt["max_dd"],
            "mkt_sig": mkt["avg_signal"],
        })

print()

# ---------------------------------------------------------------------------
if results_db:
    print("=" * 80)
    print("  AGGREGATE RESULTS")
    print("=" * 80)

    for mode, prefix in [
        ("B&H", "bh"), ("Regime", "reg"), ("OHLCV", "ohlcv"),
        ("OHLCV+XA", "xa"), ("Mkt-Implied", "mkt"),
    ]:
        sharpes = [r[f"{prefix}_sharpe"] for r in results_db]
        pnls = [r[f"{prefix}_pnl"] for r in results_db]
        dds = [r[f"{prefix}_dd"] for r in results_db]

        extra = ""
        if prefix in ("ohlcv", "xa", "mkt"):
            caps = [r[f"{prefix}_cap"] for r in results_db]
            sigs = [r[f"{prefix}_sig"] for r in results_db]
            extra = f"  AvgCap={np.mean(caps):.0%}  AvgSig={np.mean(sigs):+.3f}"
        elif prefix == "reg":
            caps = [r["reg_cap"] for r in results_db]
            extra = f"  AvgCap={np.mean(caps):.0%}"

        print(
            f"  {mode:14s}: Sharpe={np.mean(sharpes):+.3f}  "
            f"PnL={np.mean(pnls):+.1%}  DD={np.mean(dds):+.1%}{extra}"
        )

    # Edge analysis
    n = len(results_db)
    print()
    print("  --- Sharpe vs B&H ---")
    for label, prefix in [("OHLCV", "ohlcv"), ("OHLCV+XA", "xa"), ("Mkt-Implied", "mkt")]:
        beats = sum(1 for r in results_db if r[f"{prefix}_sharpe"] > r["bh_sharpe"])
        print(f"  {label:14s} beats B&H Sharpe: {beats}/{n}")

    print()
    print("  --- Drawdown vs B&H ---")
    for label, prefix in [("OHLCV", "ohlcv"), ("OHLCV+XA", "xa"), ("Mkt-Implied", "mkt")]:
        less_dd = sum(1 for r in results_db if r[f"{prefix}_dd"] > r["bh_dd"])
        print(f"  {label:14s} less drawdown:   {less_dd}/{n}")

    # Market-implied vs OHLCV-only head-to-head
    print()
    print("  --- Market-Implied vs OHLCV-only ---")
    mkt_beats_sharpe = sum(1 for r in results_db if r["mkt_sharpe"] > r["ohlcv_sharpe"])
    mkt_beats_dd = sum(1 for r in results_db if r["mkt_dd"] > r["ohlcv_dd"])
    print(f"  Mkt-Impl beats OHLCV Sharpe:   {mkt_beats_sharpe}/{n}")
    print(f"  Mkt-Impl less drawdown:        {mkt_beats_dd}/{n}")

print()
print("=" * 80)
print("  Done.")
print("=" * 80)

_out_file.close()
