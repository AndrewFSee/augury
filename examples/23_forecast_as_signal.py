"""Example 23 — Forward Distribution as a Signal.

Tests whether the Monte Carlo forecast distribution adds value beyond
backward-looking indicators.  This is the core question: does the
probabilistic forecast (like Synth's API output) carry actionable
information?

Strategies tested (all with embargo, costs, walk-forward):

  1. B&H               — baseline
  2. mom_only           — 12-1 momentum direction, binary 0/1
  3. forecast_dir       — use forecast P(positive) for direction
  4. forecast_kelly     — Kelly-optimal sizing from distribution
  5. forecast_vol_edge  — long when forecast vol < realized vol
                          (vol is overpriced → market calmer than expected)
  6. mom × forecast_size — momentum direction × distribution-derived size
  7. composite          — P(positive) direction × Kelly size × vol-target

The key test: does the FORWARD distribution from the simulation engines
(Heston + Merton JD + RS-GBM) tell you something that trailing realized
vol and momentum don't already know?
"""

from __future__ import annotations

import sys
import time
import warnings
from dataclasses import dataclass, field

import numpy as np

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
_out_path = "_forecast_signal_output.txt"
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
TICKERS = ["SPY", "QQQ", "IWM", "TLT", "GLD"]
START = "2014-06-01"
END = "2026-12-31"

LOOKBACK = 504
HORIZON = 21
STEP = 21
EMBARGO = 21
N_PATHS = 2000
COST_BPS = 10
VOL_TARGET = 0.15
MOM_LOOKBACK = 252
MOM_SKIP = 21

from synthquant.data.ingest import DataIngestor  # noqa: E402
from synthquant.ensemble_forecast import EnsembleForecaster  # noqa: E402
from synthquant.strategy.directional import TimeSeriesMomentum  # noqa: E402
from synthquant.strategy.forecast_signals import ForecastSignal  # noqa: E402


# ---------------------------------------------------------------------------
# Metrics (same as Example 22)
# ---------------------------------------------------------------------------
@dataclass
class StratMetrics:
    name: str
    pnl: np.ndarray = field(default_factory=lambda: np.array([]))
    turnover_sum: float = 0.0
    exposure_sum: float = 0.0
    n_windows: int = 0

    def finalize(self, periods_per_year: float = 252 / HORIZON) -> dict[str, float]:
        pnl = self.pnl
        if len(pnl) == 0:
            return {k: 0.0 for k in [
                "sharpe", "sortino", "calmar", "max_dd", "cvar95",
                "total_return", "turnover", "cost_adj_sharpe", "avg_expo"]}
        mu, sig = pnl.mean(), pnl.std()
        sharpe = mu / sig * np.sqrt(periods_per_year) if sig > 0 else 0.0

        downside = pnl[pnl < 0]
        sortino = (mu / downside.std() * np.sqrt(periods_per_year)
                   if len(downside) and downside.std() > 0 else 0.0)

        cum = np.cumsum(pnl)
        peak = np.maximum.accumulate(cum)
        dd = cum - peak
        max_dd = float(dd.min())
        total_ret = float(cum[-1])
        calmar = total_ret / abs(max_dd) if max_dd < 0 else float("inf")
        cvar = float(np.mean(np.sort(pnl)[: max(1, len(pnl) // 20)]))

        avg_turnover = self.turnover_sum / max(1, len(pnl))
        cost_drag = (COST_BPS / 10_000) * avg_turnover
        pnl_ac = pnl - cost_drag
        mu_c, sig_c = pnl_ac.mean(), pnl_ac.std()
        ca_sharpe = mu_c / sig_c * np.sqrt(periods_per_year) if sig_c > 0 else 0.0

        avg_expo = self.exposure_sum / max(1, self.n_windows)

        return {
            "sharpe": sharpe, "sortino": sortino, "calmar": calmar,
            "max_dd": max_dd, "cvar95": cvar, "total_return": total_ret,
            "turnover": avg_turnover, "cost_adj_sharpe": ca_sharpe,
            "avg_expo": avg_expo,
        }


# ---------------------------------------------------------------------------
# Strategy functions — all return exposure in [-1, +1]
# ---------------------------------------------------------------------------

def strat_bh(**_kw) -> float:
    return 1.0


def strat_mom_only(returns_window, **_kw) -> float:
    mom = TimeSeriesMomentum(lookback_long=MOM_LOOKBACK, skip=MOM_SKIP)
    return float(mom.generate(returns_window).direction)


def strat_forecast_dir(forecast, **_kw) -> float:
    """Use the forecast's P(positive return) directly for direction + sizing.

    If >60% of MC paths end positive → long, scaled by conviction.
    If <40% → flat.  This is the most direct use of the distribution.
    """
    if forecast is None:
        return 0.0
    p_pos = forecast.prob_positive
    if p_pos >= 0.60:
        # Scale: 0.60 → 0.0, 0.80 → 1.0
        return float(min((p_pos - 0.60) / 0.20, 1.0))
    elif p_pos <= 0.40:
        return 0.0  # Could go short but keeping long-only for fair comparison
    return 0.0


def strat_forecast_kelly(forecast, **_kw) -> float:
    """Kelly-optimal fraction from the forecast distribution.

    f* = E[r] / Var[r]  (simplified Kelly for log returns)
    Capped at 1.0, floored at 0.0 (long-only).
    """
    if forecast is None:
        return 0.0
    mu = forecast.expected_return
    var = forecast.return_std ** 2
    if var <= 0:
        return 0.0
    kelly = mu / var
    # Half-Kelly is more robust in practice
    half_kelly = kelly / 2.0
    return float(np.clip(half_kelly, 0.0, 1.0))


def strat_vol_edge(forecast, returns_window, **_kw) -> float:
    """Trade the vol forecast vs realized vol.

    When forecast vol < trailing realized vol → market is calmer than
    recently observed → go long (vol is mean-reverting down).
    When forecast vol > trailing realized → reduce exposure.

    This tests whether the forward vol forecast adds info beyond
    backward-looking vol.
    """
    if forecast is None:
        return 1.0
    fwd_vol = forecast.forecast_realized_vol
    # Trailing 63-day realized vol (annualised)
    recent = returns_window[-63:]
    realized_vol = float(np.std(recent) * np.sqrt(252))
    if realized_vol <= 0:
        return 1.0
    vol_ratio = fwd_vol / realized_vol
    # ratio < 1 → forecast calmer than recent → full long
    # ratio > 1.5 → forecast more volatile → reduce
    if vol_ratio < 0.90:
        return 1.0
    elif vol_ratio > 1.30:
        return 0.2
    else:
        # Linear interpolation: 0.90 → 1.0, 1.30 → 0.2
        return float(1.0 - 0.8 * (vol_ratio - 0.90) / 0.40)


def strat_mom_forecast_size(forecast, returns_window, **_kw) -> float:
    """Momentum direction × forecast-derived position size.

    Direction: 12-1 momentum.
    Size: from generate_decomposed (regime confidence × vol penalty × tail).
    Plus vol-target overlay.
    """
    mom = TimeSeriesMomentum(lookback_long=MOM_LOOKBACK, skip=MOM_SKIP)
    direction = mom.generate(returns_window).direction
    if direction == 0 or forecast is None:
        return 0.0
    fs = ForecastSignal()
    recent = returns_window[-63:]
    result = fs.generate_decomposed(forecast, direction=direction,
                                    recent_returns=recent)
    ann_vol = float(np.mean(forecast.realized_vol_samples))
    vol_scale = min(VOL_TARGET / ann_vol, 1.0) if ann_vol > 0 else 1.0
    return float(direction * result.size * vol_scale)


def strat_composite(forecast, returns_window, **_kw) -> float:
    """Full composite: distribution direction × Kelly size × vol target.

    Direction: P(positive) from forecast (must be > 0.55)
    Size: half-Kelly from distribution
    Overlay: vol target
    """
    if forecast is None:
        return 0.0

    # Direction from distribution
    p_pos = forecast.prob_positive
    if p_pos < 0.55:
        return 0.0

    # Kelly sizing
    mu = forecast.expected_return
    var = forecast.return_std ** 2
    if var <= 0:
        return 0.0
    half_kelly = np.clip(mu / var / 2.0, 0.0, 1.0)

    # Vol target overlay
    ann_vol = forecast.forecast_realized_vol
    vol_scale = min(VOL_TARGET / ann_vol, 1.0) if ann_vol > 0 else 1.0

    return float(half_kelly * vol_scale)


STRATEGIES = {
    "bh": strat_bh,
    "mom_only": strat_mom_only,
    "fcast_dir": strat_forecast_dir,
    "fcast_kelly": strat_forecast_kelly,
    "vol_edge": strat_vol_edge,
    "mom×fsize": strat_mom_forecast_size,
    "composite": strat_composite,
}


# ---------------------------------------------------------------------------
# Walk-forward harness
# ---------------------------------------------------------------------------

def build_forecast(train_rets, current_price):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fc = EnsembleForecaster(
            detector="hmm", n_regimes=2, n_paths=N_PATHS,
            random_state=42, use_filtered=True,
        )
        fc.fit(train_rets)
        return fc.forecast(horizon=HORIZON, current_price=current_price)


def run_walk_forward(returns, prices):
    n = len(returns)
    strats = {name: StratMetrics(name) for name in STRATEGIES}
    prev_expo = {k: 0.0 for k in strats}

    # Also track forecast diagnostics
    diagnostics = {
        "p_positive": [], "expected_return": [], "forecast_vol": [],
        "realized_vol_trailing": [], "realized_fwd_vol": [],
        "actual_fwd_return": [],
    }

    t = LOOKBACK
    while t + EMBARGO + HORIZON <= n:
        train_end = t
        eval_start = t + EMBARGO
        eval_end = eval_start + HORIZON

        actual_fwd = float(np.sum(returns[eval_start:eval_end]))
        actual_fwd_vol = float(np.std(returns[eval_start:eval_end]) * np.sqrt(252))

        returns_window = returns[:train_end]
        prices_window = prices[:train_end + 1]
        recent = returns_window[-63:]
        trailing_vol = float(np.std(recent) * np.sqrt(252))

        # Build the forecast
        try:
            forecast = build_forecast(
                returns[train_end - LOOKBACK:train_end],
                float(prices[train_end]),
            )
        except Exception:
            forecast = None

        # Record diagnostics
        if forecast is not None:
            diagnostics["p_positive"].append(forecast.prob_positive)
            diagnostics["expected_return"].append(forecast.expected_return)
            diagnostics["forecast_vol"].append(forecast.forecast_realized_vol)
            diagnostics["realized_vol_trailing"].append(trailing_vol)
            diagnostics["realized_fwd_vol"].append(actual_fwd_vol)
            diagnostics["actual_fwd_return"].append(actual_fwd)

        # Compute exposures
        kw = dict(forecast=forecast, returns_window=returns_window,
                  prices_window=prices_window)
        for name, func in STRATEGIES.items():
            expo = func(**kw)
            turnover = abs(expo - prev_expo[name])
            prev_expo[name] = expo
            strats[name].pnl = np.append(strats[name].pnl, expo * actual_fwd)
            strats[name].turnover_sum += turnover
            strats[name].exposure_sum += abs(expo)
            strats[name].n_windows += 1

        t += STEP

    return strats, diagnostics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    print("=" * 80)
    print("  FORWARD DISTRIBUTION AS A SIGNAL")
    print(f"  Tickers: {', '.join(TICKERS)}  |  {START} → {END}")
    print(f"  Lookback={LOOKBACK}d  Horizon={HORIZON}d  Embargo={EMBARGO}d  "
          f"Paths={N_PATHS}  Cost={COST_BPS}bps  VolTarget={VOL_TARGET:.0%}")
    print("=" * 80)

    ingestor = DataIngestor()
    all_results: dict[str, dict[str, dict]] = {}
    all_diag: dict[str, dict] = {}

    for ticker in TICKERS:
        try:
            df = ingestor.fetch(ticker, start=START, end=END)
        except Exception as e:
            print(f"\n  SKIP {ticker}: {e}")
            continue

        close = df["close"].to_numpy(dtype=float)
        rets = np.diff(np.log(close))
        if len(rets) < LOOKBACK + HORIZON + EMBARGO + 50:
            print(f"\n  SKIP {ticker}: too few observations")
            continue

        t0 = time.perf_counter()
        strats, diag = run_walk_forward(rets, close)
        elapsed = time.perf_counter() - t0

        all_results[ticker] = {}
        all_diag[ticker] = diag

        print(f"\n  {ticker}  ({len(rets)} days, "
              f"{strats['bh'].n_windows} windows, {elapsed:.0f}s)")

        headers = ["Strategy", "Sharpe", "CostAdj", "Sortino",
                   "MaxDD", "CVaR95", "TotRet", "AvgExpo", "Turn"]
        print("    " + "".join(f"{h:>10}" for h in headers))

        for name in STRATEGIES:
            m = strats[name].finalize()
            all_results[ticker][name] = m
            print(f"    {name:>10}"
                  f"{m['sharpe']:>+10.2f}{m['cost_adj_sharpe']:>+10.2f}"
                  f"{m['sortino']:>+10.2f}{m['max_dd']:>+10.1%}"
                  f"{m['cvar95']:>+10.1%}{m['total_return']:>+10.1%}"
                  f"{m['avg_expo']:>10.2f}{m['turnover']:>10.2f}")

    # ------------------------------------------------------------------
    # Aggregate
    # ------------------------------------------------------------------
    if all_results:
        print(f"\n{'=' * 80}")
        print("  AGGREGATE (mean across tickers)")
        print(f"{'=' * 80}")
        keys = ["sharpe", "cost_adj_sharpe", "sortino", "max_dd",
                "cvar95", "total_return", "avg_expo", "turnover"]
        headers = ["Strategy"] + keys
        print("  " + "".join(f"{h:>14}" for h in headers))
        for strat in STRATEGIES:
            vals = []
            for k in keys:
                v = np.mean([all_results[t][strat][k] for t in all_results])
                vals.append(f"{v:+.3f}")
            print("  " + f"{strat:>14}" + "".join(f"{v:>14}" for v in vals))

    # ------------------------------------------------------------------
    # Forecast diagnostics
    # ------------------------------------------------------------------
    if all_diag:
        print(f"\n{'=' * 80}")
        print("  FORECAST DISTRIBUTION DIAGNOSTICS")
        print(f"{'=' * 80}")

        for ticker, diag in all_diag.items():
            if not diag["p_positive"]:
                continue
            p_pos = np.array(diag["p_positive"])
            e_ret = np.array(diag["expected_return"])
            fv = np.array(diag["forecast_vol"])
            tv = np.array(diag["realized_vol_trailing"])
            rv = np.array(diag["realized_fwd_vol"])
            ar = np.array(diag["actual_fwd_return"])

            print(f"\n  {ticker}  ({len(p_pos)} windows)")

            # P(positive) calibration
            above_55 = p_pos > 0.55
            above_60 = p_pos > 0.60
            actually_pos = ar > 0
            print(f"    P(positive) mean: {p_pos.mean():.3f}  "
                  f"std: {p_pos.std():.3f}  "
                  f"range: [{p_pos.min():.3f}, {p_pos.max():.3f}]")
            if above_55.sum() > 0:
                hit_55 = (actually_pos & above_55).sum() / above_55.sum()
                print(f"    When P(pos)>0.55: hit rate {hit_55:.1%} "
                      f"({above_55.sum()} windows)")
            if above_60.sum() > 0:
                hit_60 = (actually_pos & above_60).sum() / above_60.sum()
                print(f"    When P(pos)>0.60: hit rate {hit_60:.1%} "
                      f"({above_60.sum()} windows)")

            # Expected return vs actual
            from scipy.stats import spearmanr
            rho_ret, p_ret = spearmanr(e_ret, ar)
            print(f"    E[return] vs actual: rho={rho_ret:+.3f} (p={p_ret:.3f})")

            # Forecast vol vs realized forward vol
            rho_vol, p_vol = spearmanr(fv, rv)
            mae_vol = np.mean(np.abs(fv - rv))
            # Also compare trailing vol vs realized
            rho_trail, p_trail = spearmanr(tv, rv)
            mae_trail = np.mean(np.abs(tv - rv))
            print(f"    Forecast vol vs fwd realized: "
                  f"rho={rho_vol:+.3f} (p={p_vol:.3f})  MAE={mae_vol:.4f}")
            print(f"    Trailing vol vs fwd realized: "
                  f"rho={rho_trail:+.3f} (p={p_trail:.3f})  MAE={mae_trail:.4f}")
            vol_wins = "FORECAST" if rho_vol > rho_trail else "TRAILING"
            print(f"    Better vol predictor: {vol_wins}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n{'=' * 80}")
    print("  INTERPRETATION")
    print(f"{'=' * 80}")
    print("""
  This example tests the Synth-style question: does a probabilistic
  forward distribution carry actionable information?

  Key signals extracted from the Monte Carlo distribution:

  1. P(positive)    — directional: what fraction of paths end up?
  2. E[return]/Var  — Kelly sizing: how much to bet given the distribution
  3. Forecast vol   — vol edge: is forward vol above or below trailing?
  4. Regime + tail  — generate_decomposed: regime confidence × vol/tail penalty

  The comparison against momentum-only and B&H reveals whether the
  forward simulation engines (Heston + Merton JD + RS-GBM) add
  anything beyond what backward-looking indicators already capture.
""")

    print(f"\n  Output saved to {_out_path}")
    print(f"{'=' * 80}")
    _out_file.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
