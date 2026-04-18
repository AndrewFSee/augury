"""Robustness test: default signal weights across tickers and time periods.

Tests the untuned ForecastSignal on a grid of:
  - Tickers: SPY, QQQ, IWM, DIA, EFA, TLT, GLD
  - Periods: 2015-2019 (pre-COVID), 2020-2022 (COVID+bear),
             2023-2026 (recovery+bull), 2015-2026 (full)

For each (ticker, period) cell we run walk-forward forecasts and
evaluate with default weights.  Results are printed as a matrix.
"""

import sys
import time

import numpy as np
import yfinance as yf

from synthquant.strategy.signal_tuner import SignalTuner

# ── Log to file + stdout ──────────────────────────────────────────────────────

_LOG_PATH = "_robustness_output.txt"
_logfile = open(_LOG_PATH, "w", encoding="utf-8")
_orig_print = print


def print(*args, **kwargs):
    kwargs.pop("flush", None)
    _orig_print(*args, **kwargs, flush=True)
    _orig_print(*args, file=_logfile, flush=True)


# ── Configuration ─────────────────────────────────────────────────────────────

TICKERS = ["SPY", "QQQ", "IWM", "DIA", "EFA", "TLT", "GLD"]

PERIODS = {
    "2015-2019": ("2015-01-01", "2020-01-01"),
    "2020-2022": ("2020-01-01", "2023-01-01"),
    "2023-2026": ("2023-01-01", "2026-12-31"),
    "Full":      ("2015-01-01", "2026-12-31"),
}

LOOKBACK = 504
HORIZON = 21
STEP = 21
N_PATHS = 3000  # fewer paths for speed (many cells to evaluate)


def download(ticker: str, start: str, end: str) -> tuple[np.ndarray, np.ndarray]:
    hist = yf.Ticker(ticker).history(start=start, end=end)
    if hist.empty:
        return np.array([]), np.array([])
    prices = hist["Close"].values
    returns = np.diff(np.log(prices))
    mask = np.isfinite(returns)
    return returns[mask], prices


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    t_start = time.time()
    tuner = SignalTuner(
        n_paths=N_PATHS, horizon=HORIZON, lookback=LOOKBACK,
        step=STEP, random_state=42,
    )

    print("=" * 90)
    print("  DEFAULT SIGNAL ROBUSTNESS TEST")
    print(f"  Tickers: {', '.join(TICKERS)}")
    print(f"  Periods: {', '.join(PERIODS)}")
    print(f"  Lookback: {LOOKBACK}d | Horizon: {HORIZON}d | Step: {STEP}d | Paths: {N_PATHS}")
    print("=" * 90)

    # results[ticker][period] = metrics dict
    results: dict[str, dict[str, dict]] = {}

    for ti, ticker in enumerate(TICKERS):
        results[ticker] = {}
        for pi, (period_name, (start, end)) in enumerate(PERIODS.items()):
            cell = f"[{ti * len(PERIODS) + pi + 1}/{len(TICKERS) * len(PERIODS)}]"
            print(f"\n  {cell} {ticker} / {period_name}  ({start} to {end})")

            returns, prices = download(ticker, start, end)
            if len(returns) < LOOKBACK + HORIZON + STEP:
                print(f"    SKIP — only {len(returns)} days (need {LOOKBACK + HORIZON + STEP})")
                results[ticker][period_name] = None
                continue

            print(f"    {len(returns)} trading days, ${prices.min():.2f}-${prices.max():.2f}")

            t0 = time.time()
            snapshots = tuner.prepare(returns, prices)
            prep_time = time.time() - t0

            if len(snapshots) < 3:
                print(f"    SKIP — only {len(snapshots)} snapshots")
                results[ticker][period_name] = None
                continue

            metrics = tuner.evaluate(snapshots)
            metrics["n_snapshots"] = len(snapshots)

            # Buy-and-hold benchmark over the same snapshot periods
            fwd_rets = np.array([s.actual_forward_return for s in snapshots])
            bh_total = float(fwd_rets.sum())  # cumulative log return
            periods_per_year = 252 / HORIZON
            if len(fwd_rets) > 1 and np.std(fwd_rets) > 1e-12:
                bh_sharpe = float(
                    np.mean(fwd_rets) / np.std(fwd_rets)
                    * np.sqrt(periods_per_year)
                )
            else:
                bh_sharpe = 0.0
            metrics["bh_pnl"] = bh_total
            metrics["bh_sharpe"] = bh_sharpe

            results[ticker][period_name] = metrics

            print(
                f"    {len(snapshots)} snaps | Signal: Sharpe={metrics['sharpe']:+.2f} "
                f"PnL={metrics['total_pnl']:+.1%} "
                f"| B&H: Sharpe={bh_sharpe:+.2f} PnL={bh_total:+.1%}  ({prep_time:.0f}s)"
            )

    # ── Signal PnL vs Buy-and-Hold ──────────────────────────────────────

    period_names = list(PERIODS.keys())
    header = f"  {'Ticker':<8}" + "".join(f"{p:>12}" for p in period_names)
    sep = "  " + "-" * (8 + 12 * len(period_names))

    print(f"\n{'=' * 90}")
    print("  SIGNAL PnL vs BUY-AND-HOLD  (cumulative log returns)")
    print(f"{'=' * 90}")

    for tag, key in [("Signal", "total_pnl"), ("B&H", "bh_pnl")]:
        print(f"\n  {tag}:")
        print(header)
        print(sep)
        for ticker in TICKERS:
            row = f"  {ticker:<8}"
            for p in period_names:
                m = results[ticker].get(p)
                if m is None:
                    row += f"{'---':>12}"
                else:
                    row += f"{m[key]:>+11.1%} "
            print(row)

    # Signal as % of B&H
    print(f"\n  Signal / B&H  (how much of buy-and-hold the signal captures):")
    print(header)
    print(sep)
    capture_ratios = []
    for ticker in TICKERS:
        row = f"  {ticker:<8}"
        for p in period_names:
            m = results[ticker].get(p)
            if m is None:
                row += f"{'---':>12}"
            elif abs(m["bh_pnl"]) < 1e-6:
                row += f"{'N/A':>12}"
            else:
                ratio = m["total_pnl"] / m["bh_pnl"]
                row += f"{ratio:>11.0%} "
                capture_ratios.append(ratio)
        print(row)

    if capture_ratios:
        print(f"\n  Avg capture ratio: {np.mean(capture_ratios):.0%}  "
              f"Median: {np.median(capture_ratios):.0%}")

    # ── Sharpe comparison ─────────────────────────────────────────────────

    print(f"\n{'=' * 90}")
    print("  SHARPE RATIO: SIGNAL vs BUY-AND-HOLD")
    print(f"{'=' * 90}")

    for tag, key in [("Signal", "sharpe"), ("B&H", "bh_sharpe")]:
        print(f"\n  {tag}:")
        print(header)
        print(sep)
        for ticker in TICKERS:
            row = f"  {ticker:<8}"
            for p in period_names:
                m = results[ticker].get(p)
                if m is None:
                    row += f"{'---':>12}"
                else:
                    row += f"{m[key]:>+12.2f}"
            print(row)

    # Signal Sharpe minus B&H Sharpe
    print(f"\n  Signal - B&H  (positive = signal risk-adjusted better):")
    print(header)
    print(sep)
    sharpe_deltas = []
    for ticker in TICKERS:
        row = f"  {ticker:<8}"
        for p in period_names:
            m = results[ticker].get(p)
            if m is None:
                row += f"{'---':>12}"
            else:
                delta = m["sharpe"] - m["bh_sharpe"]
                row += f"{delta:>+12.2f}"
                sharpe_deltas.append(delta)
        print(row)

    if sharpe_deltas:
        print(f"\n  Avg Sharpe delta: {np.mean(sharpe_deltas):+.2f}  "
              f"Signal beats B&H: {sum(d > 0 for d in sharpe_deltas)}/{len(sharpe_deltas)} cells")

    # ── Summary ───────────────────────────────────────────────────────────

    print(f"\n{'=' * 90}")
    print("  SUMMARY")
    print(f"{'=' * 90}")
    print(f"  {'Ticker':<8} {'Sig Sharpe':>11} {'B&H Sharpe':>11} {'Sig PnL':>9} {'B&H PnL':>9} {'Capture':>9}")
    print("  " + "-" * 58)

    for ticker in TICKERS:
        valid = [m for m in results[ticker].values() if m is not None]
        if not valid:
            continue
        # Use "Full" period if available, else average
        full = results[ticker].get("Full")
        m = full if full is not None else valid[-1]
        cap = m["total_pnl"] / m["bh_pnl"] if abs(m["bh_pnl"]) > 1e-6 else 0
        print(f"  {ticker:<8} {m['sharpe']:>+11.2f} {m['bh_sharpe']:>+11.2f} "
              f"{m['total_pnl']:>+8.1%} {m['bh_pnl']:>+8.1%} {cap:>8.0%}")

    print(f"\n  {'Period':<10} {'Sig Sharpe':>11} {'B&H Sharpe':>11} {'Sig PnL':>9} {'B&H PnL':>9}")
    print("  " + "-" * 49)
    for pname in period_names:
        valid = [results[t][pname] for t in TICKERS if results[t].get(pname) is not None]
        if not valid:
            continue
        avg_ss = np.mean([m["sharpe"] for m in valid])
        avg_bs = np.mean([m["bh_sharpe"] for m in valid])
        avg_sp = np.mean([m["total_pnl"] for m in valid])
        avg_bp = np.mean([m["bh_pnl"] for m in valid])
        print(f"  {pname:<10} {avg_ss:>+11.2f} {avg_bs:>+11.2f} {avg_sp:>+8.1%} {avg_bp:>+8.1%}")

    elapsed = time.time() - t_start
    print(f"\n  Total runtime: {elapsed:.0f}s")
    print(f"{'=' * 90}")


if __name__ == "__main__":
    main()
