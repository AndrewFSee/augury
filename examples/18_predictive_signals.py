"""Example 18 — Predictive Signals vs Buy-and-Hold.

Compares four strategies across multiple tickers and time periods:

  1. **B&H** — 100% buy-and-hold.
  2. **Regime-only** — :class:`RegimeAllocator` (reactive, no predictive layer).
  3. **Predictive** — :class:`PredictiveAllocator` with leading indicators
     computed from OHLCV data.
  4. **Predictive + Cross-asset** — same as above but with TLT returns as
     a safe-haven signal for flight-to-safety detection.

The hypothesis: predictive signals (vol term structure, price-volume
divergence, trend deceleration, RSI) can *anticipate* regime changes,
giving an actual edge over buy-and-hold — something the reactive regime
allocator alone cannot do.

We use full OHLCV data (not just returns) so the predictive module can
compute range expansion, volume divergence, and other leading indicators.
"""

from __future__ import annotations

import sys
import time
import warnings

import numpy as np

# ---------------------------------------------------------------------------
_out_path = "_predictive_signals_output.txt"
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
print("  PREDICTIVE SIGNALS vs BUY-AND-HOLD")
print(f"  Tickers: {', '.join(TICKERS)}")
print(f"  Periods: {', '.join(p[0] for p in PERIODS)}")
print(f"  Lookback={LOOKBACK} | Horizon={HORIZON} | Step={STEP} | Paths={N_PATHS}")
print("=" * 80)
print()

# ---------------------------------------------------------------------------
from synthquant.data.ingest import DataIngestor
from synthquant.ensemble_forecast import EnsembleForecaster
from synthquant.strategy.regime_allocator import RegimeAllocator
from synthquant.strategy.predictive_allocator import PredictiveAllocator

ingestor = DataIngestor()

# Pre-fetch TLT for cross-asset signals
print("Fetching TLT for cross-asset signals...")
try:
    tlt_df = ingestor.fetch("TLT", start="2014-06-01", end="2026-12-31")
    tlt_returns = np.log(tlt_df["close"] / tlt_df["close"].shift(1)).dropna().values
    tlt_index = tlt_df.index[1:]  # align with returns
    print(f"  TLT: {len(tlt_returns)} days loaded")
except Exception as e:
    print(f"  TLT fetch failed: {e} — cross-asset signals disabled")
    tlt_returns = None
    tlt_index = None
print()


# ---------------------------------------------------------------------------
def walk_forward(
    df,
    mode: str,
    tlt_rets: np.ndarray | None = None,
    tlt_idx=None,
) -> dict[str, float]:
    """Walk-forward backtest with full OHLCV data.

    mode: 'bh', 'regime', 'pred', 'pred_xa'
    """
    close_arr = df["close"].values
    high_arr = df["high"].values
    low_arr = df["low"].values
    volume_arr = df["volume"].values.astype(float)
    returns = np.log(df["close"] / df["close"].shift(1)).dropna().values
    n = len(returns)

    # Offset: returns[i] corresponds to close[i+1] relative to close[i]
    # So returns array has length n, close has length n+1
    # We'll align: close_aligned[i] = close[i+1], so returns[i] = log(close_aligned[i] / close_aligned[i-1])
    close_aligned = close_arr[1:]  # drop first
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

    snap_returns = []
    snap_exposures = []
    snap_signals = []
    snap_regimes = []
    t = LOOKBACK

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
            if mode == "pred_xa" and tlt_rets is not None and tlt_idx is not None:
                # Align by date index
                eq_dates = df.index[1:]  # aligned with returns
                eq_start_date = eq_dates[t - LOOKBACK]
                eq_end_date = eq_dates[t - 1]
                tlt_mask = (tlt_idx >= eq_start_date) & (tlt_idx <= eq_end_date)
                safe_haven_window = tlt_rets[tlt_mask.values if hasattr(tlt_mask, 'values') else tlt_mask]
                if len(safe_haven_window) >= 63:
                    safe_haven = safe_haven_window

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
                elif mode == "pred":
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
                elif mode == "pred_xa":
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
                "n_snaps": 0, "avg_exp": 0.0, "pct_crisis": 0.0,
                "avg_signal": 0.0}

    mu = np.mean(snap_returns)
    sigma = np.std(snap_returns)
    sharpe = mu / sigma * np.sqrt(252 / HORIZON) if sigma > 0 else 0.0

    pnl = float(np.sum(snap_returns))
    cum = np.cumsum(snap_returns)
    peak = np.maximum.accumulate(cum)
    max_dd = float(np.min(cum - peak))

    snap_regimes_arr = np.array(snap_regimes)
    pct_crisis = float(np.mean(snap_regimes_arr == 1)) if n_snaps > 0 else 0.0

    return {
        "sharpe": sharpe, "pnl": pnl, "max_dd": max_dd,
        "n_snaps": n_snaps, "avg_exp": float(np.mean(snap_exposures)),
        "pct_crisis": pct_crisis,
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
        pred = walk_forward(df, "pred")
        pred_xa = walk_forward(df, "pred_xa", tlt_returns, tlt_index)

        elapsed = time.perf_counter() - t0
        bh_pnl = bh["pnl"]

        def cap(x):
            return x["pnl"] / bh_pnl if abs(bh_pnl) > 1e-8 else 0.0

        print(f"    B&H:      Sharpe={bh['sharpe']:+.2f}  PnL={bh['pnl']:+.1%}  DD={bh['max_dd']:+.1%}")
        print(
            f"    Regime:   Sharpe={regime['sharpe']:+.2f}  PnL={regime['pnl']:+.1%}  "
            f"Cap={cap(regime):.0%}  AvgExp={regime['avg_exp']:.2f}  DD={regime['max_dd']:+.1%}"
        )
        print(
            f"    Pred:     Sharpe={pred['sharpe']:+.2f}  PnL={pred['pnl']:+.1%}  "
            f"Cap={cap(pred):.0%}  AvgExp={pred['avg_exp']:.2f}  "
            f"DD={pred['max_dd']:+.1%}  Sig={pred['avg_signal']:+.3f}"
        )
        print(
            f"    Pred+XA:  Sharpe={pred_xa['sharpe']:+.2f}  PnL={pred_xa['pnl']:+.1%}  "
            f"Cap={cap(pred_xa):.0%}  AvgExp={pred_xa['avg_exp']:.2f}  "
            f"DD={pred_xa['max_dd']:+.1%}  Sig={pred_xa['avg_signal']:+.3f}"
            f"  ({elapsed:.0f}s)"
        )

        results_db.append({
            "ticker": ticker, "period": period_name,
            "bh_sharpe": bh["sharpe"], "bh_pnl": bh["pnl"], "bh_dd": bh["max_dd"],
            "reg_sharpe": regime["sharpe"], "reg_pnl": regime["pnl"],
            "reg_cap": cap(regime), "reg_dd": regime["max_dd"],
            "pred_sharpe": pred["sharpe"], "pred_pnl": pred["pnl"],
            "pred_cap": cap(pred), "pred_dd": pred["max_dd"],
            "pred_sig": pred["avg_signal"],
            "xa_sharpe": pred_xa["sharpe"], "xa_pnl": pred_xa["pnl"],
            "xa_cap": cap(pred_xa), "xa_dd": pred_xa["max_dd"],
            "xa_sig": pred_xa["avg_signal"],
        })

print()

# ---------------------------------------------------------------------------
if results_db:
    print("=" * 80)
    print("  AGGREGATE RESULTS")
    print("=" * 80)

    for mode, prefix in [
        ("B&H", "bh"), ("Regime", "reg"), ("Predictive", "pred"), ("Pred+XA", "xa"),
    ]:
        sharpes = [r[f"{prefix}_sharpe"] for r in results_db]
        pnls = [r[f"{prefix}_pnl"] for r in results_db]
        dds = [r[f"{prefix}_dd"] for r in results_db]

        extra = ""
        if prefix in ("pred", "xa"):
            caps = [r[f"{prefix}_cap"] for r in results_db]
            sigs = [r[f"{prefix}_sig"] for r in results_db]
            extra = f"  AvgCap={np.mean(caps):.0%}  AvgSig={np.mean(sigs):+.3f}"

        if prefix == "reg":
            caps = [r["reg_cap"] for r in results_db]
            extra = f"  AvgCap={np.mean(caps):.0%}"

        print(
            f"  {mode:12s}: Sharpe={np.mean(sharpes):+.3f}  "
            f"PnL={np.mean(pnls):+.1%}  DD={np.mean(dds):+.1%}{extra}"
        )

    # Edge analysis: how often does predictive beat B&H?
    pred_beats = sum(
        1 for r in results_db if r["pred_sharpe"] > r["bh_sharpe"]
    )
    xa_beats = sum(
        1 for r in results_db if r["xa_sharpe"] > r["bh_sharpe"]
    )
    print()
    print(f"  Predictive beats B&H Sharpe: {pred_beats}/{len(results_db)}")
    print(f"  Pred+XA beats B&H Sharpe:    {xa_beats}/{len(results_db)}")

    # Drawdown analysis
    pred_less_dd = sum(
        1 for r in results_db if r["pred_dd"] > r["bh_dd"]  # less negative = better
    )
    print(f"  Predictive less drawdown:    {pred_less_dd}/{len(results_db)}")

print()
print("=" * 80)
print("  Done.")
print("=" * 80)

_out_file.close()
