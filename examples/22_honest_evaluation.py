"""Example 22 — Honest walk-forward evaluation of momentum × vol-sizing.

Demonstrates the post-audit architecture:

  * **Direction** comes from time-series momentum (12-1). The vol forecast
    does *not* supply direction — it only supplies *size*.
  * **Size** comes from ``ForecastSignal.generate_decomposed`` (regime
    confidence × vol penalty × tail penalty) applied to the
    :class:`EnsembleForecaster` distribution.
  * **Regime detection** uses filtered (forward-only) posteriors by default,
    removing the HMM smoother lookahead identified in the audit.
  * **Transaction costs** are applied explicitly to every rebalance.
  * **Benchmarks** include a 200-day MA regime filter and momentum-only
    (no vol forecast) so you can see what the vol forecast contributes.

Run::

    python examples/22_honest_evaluation.py

Reported metrics per strategy: Sharpe, Sortino, Calmar, MaxDD, CVaR-95
of 21-day PnL, turnover, cost-adjusted Sharpe, cumulative return.
"""

from __future__ import annotations

import sys
import time
import warnings
from dataclasses import dataclass, field

import numpy as np

# ---------------------------------------------------------------------------
# Dual output
# ---------------------------------------------------------------------------
_out_path = "_honest_evaluation_output.txt"
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

LOOKBACK = 504       # forecaster training window
HORIZON = 21         # forecast + rebalance horizon
STEP = 21            # days between rebalances
EMBARGO = 21         # gap between train end and first out-of-sample return
N_PATHS = 2000
COST_BPS = 10        # per-turnover round-trip cost (10 bps)
VOL_TARGET = 0.15    # annualised

MOM_LOOKBACK = 252   # 12-1 momentum
MOM_SKIP = 21

from synthquant.data.ingest import DataIngestor  # noqa: E402
from synthquant.ensemble_forecast import EnsembleForecaster  # noqa: E402
from synthquant.strategy.directional import TimeSeriesMomentum  # noqa: E402
from synthquant.strategy.forecast_signals import ForecastSignal  # noqa: E402


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
@dataclass
class StratMetrics:
    name: str
    pnl: np.ndarray = field(default_factory=lambda: np.array([]))
    turnover_sum: float = 0.0

    @property
    def n_periods(self) -> int:
        return len(self.pnl)

    def finalize(self, periods_per_year: float = 252 / HORIZON) -> dict[str, float]:
        pnl = self.pnl
        if len(pnl) == 0:
            return {"sharpe": 0.0, "sortino": 0.0, "calmar": 0.0,
                    "max_dd": 0.0, "cvar95": 0.0, "total_return": 0.0,
                    "turnover": 0.0, "cost_adj_sharpe": 0.0}
        mu, sig = pnl.mean(), pnl.std()
        sharpe = mu / sig * np.sqrt(periods_per_year) if sig > 0 else 0.0

        downside = pnl[pnl < 0]
        sortino = mu / downside.std() * np.sqrt(periods_per_year) if len(downside) and downside.std() > 0 else 0.0

        cum = np.cumsum(pnl)
        peak = np.maximum.accumulate(cum)
        dd = cum - peak
        max_dd = float(dd.min())
        total_ret = float(cum[-1])
        calmar = total_ret / abs(max_dd) if max_dd < 0 else np.inf

        cvar = float(np.mean(np.sort(pnl)[: max(1, len(pnl) // 20)]))

        # Turnover and cost-adjusted Sharpe
        avg_turnover = self.turnover_sum / max(1, self.n_periods)
        cost_drag_per_period = (COST_BPS / 10000.0) * avg_turnover
        pnl_after_cost = pnl - cost_drag_per_period
        mu_c, sig_c = pnl_after_cost.mean(), pnl_after_cost.std()
        cost_adj_sharpe = mu_c / sig_c * np.sqrt(periods_per_year) if sig_c > 0 else 0.0

        return {
            "sharpe": sharpe, "sortino": sortino, "calmar": calmar,
            "max_dd": max_dd, "cvar95": cvar,
            "total_return": total_ret, "turnover": avg_turnover,
            "cost_adj_sharpe": cost_adj_sharpe,
        }


# ---------------------------------------------------------------------------
# Strategies — each returns a target exposure in [-1, +1]
# ---------------------------------------------------------------------------
def exposure_bh(*_args, **_kwargs) -> float:
    return 1.0


def exposure_ma200(prices_window: np.ndarray, **_kwargs) -> float:
    """Classic 200-day MA regime filter: long when price > MA200."""
    if len(prices_window) < 200:
        return 1.0
    ma = float(np.mean(prices_window[-200:]))
    return 1.0 if prices_window[-1] > ma else 0.0


def exposure_vol_target_bh(returns_window: np.ndarray, **_kwargs) -> float:
    """Constant-target vol-scaled long."""
    realised_vol = float(np.std(returns_window[-63:]) * np.sqrt(252))
    if realised_vol <= 0:
        return 1.0
    return float(min(VOL_TARGET / realised_vol, 1.0))


def exposure_momentum_only(returns_window: np.ndarray, **_kwargs) -> float:
    """Time-series momentum direction only (0 or 1), no vol sizing."""
    mom = TimeSeriesMomentum(lookback_long=MOM_LOOKBACK, skip=MOM_SKIP)
    return float(mom.generate(returns_window).direction)


def exposure_mom_vol(
    returns_window: np.ndarray,
    forecast,
    recent: np.ndarray,
) -> float:
    """Momentum direction × forecast-derived size × vol target."""
    mom = TimeSeriesMomentum(lookback_long=MOM_LOOKBACK, skip=MOM_SKIP)
    direction = mom.generate(returns_window).direction
    if direction == 0:
        return 0.0
    fs = ForecastSignal()
    result = fs.generate_decomposed(forecast, direction=direction, recent_returns=recent)
    # Overlay 15% vol target.
    ann_vol = float(np.mean(forecast.realized_vol_samples))
    vol_scale = min(VOL_TARGET / ann_vol, 1.0) if ann_vol > 0 else 1.0
    return float(direction * result.size * vol_scale)


# ---------------------------------------------------------------------------
# Walk-forward harness
# ---------------------------------------------------------------------------
def build_forecast(train_rets: np.ndarray, current_price: float):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fc = EnsembleForecaster(
            detector="hmm", n_regimes=2, n_paths=N_PATHS,
            random_state=42, use_filtered=True,
        )
        fc.fit(train_rets)
        return fc.forecast(horizon=HORIZON, current_price=current_price)


def run_walk_forward(returns: np.ndarray, prices: np.ndarray) -> dict[str, StratMetrics]:
    n = len(returns)
    strats = {name: StratMetrics(name) for name in
              ["bh", "ma200", "vol_bh", "mom_only", "mom_vol"]}
    prev_expo = {k: 0.0 for k in strats}

    t = LOOKBACK
    while t + EMBARGO + HORIZON <= n:
        train_end = t  # train ends here
        eval_start = t + EMBARGO
        eval_end = eval_start + HORIZON

        actual_fwd = float(np.sum(returns[eval_start:eval_end]))

        returns_window = returns[:train_end]
        prices_window = prices[:train_end + 1]
        recent = returns[max(0, train_end - 63):train_end]

        need_forecast = True
        try:
            forecast = build_forecast(returns[train_end - LOOKBACK:train_end],
                                      float(prices[train_end]))
        except Exception:
            forecast = None
            need_forecast = False

        exposures = {
            "bh": exposure_bh(),
            "ma200": exposure_ma200(prices_window),
            "vol_bh": exposure_vol_target_bh(returns_window),
            "mom_only": exposure_momentum_only(returns_window),
            "mom_vol": exposure_mom_vol(returns_window, forecast, recent) if forecast is not None else 0.0,
        }
        del need_forecast

        for name, expo in exposures.items():
            turnover = abs(expo - prev_expo[name])
            prev_expo[name] = expo
            strats[name].pnl = np.append(strats[name].pnl, expo * actual_fwd)
            strats[name].turnover_sum += turnover

        t += STEP

    return strats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    print("=" * 80)
    print("  HONEST WALK-FORWARD EVALUATION")
    print(f"  Tickers: {', '.join(TICKERS)}  |  {START} → {END}")
    print(f"  Lookback={LOOKBACK}d  Horizon={HORIZON}d  Embargo={EMBARGO}d  "
          f"Paths={N_PATHS}  Cost={COST_BPS}bps  VolTarget={VOL_TARGET:.0%}")
    print("=" * 80)
    print()

    ingestor = DataIngestor()
    per_ticker: dict[str, dict[str, dict[str, float]]] = {}

    for ticker in TICKERS:
        try:
            df = ingestor.fetch(ticker, start=START, end=END)
        except Exception as e:
            print(f"  SKIP {ticker}: {e}")
            continue

        close = df["close"].to_numpy(dtype=float)
        rets = np.diff(np.log(close))
        if len(rets) < LOOKBACK + HORIZON + EMBARGO + 50:
            print(f"  SKIP {ticker}: too few observations ({len(rets)})")
            continue

        t0 = time.perf_counter()
        strats = run_walk_forward(rets, close)
        elapsed = time.perf_counter() - t0

        print(f"  {ticker}  ({len(rets)} days, {strats['bh'].n_periods} windows, {elapsed:.0f}s)")
        headers = ["Strategy", "Sharpe", "Cost-Adj", "Sortino", "Calmar",
                   "MaxDD", "CVaR95", "TotRet", "Turnover"]
        print("    " + "  ".join(f"{h:>9}" for h in headers))
        per_ticker[ticker] = {}
        for name, sm in strats.items():
            m = sm.finalize()
            per_ticker[ticker][name] = m
            print(f"    {name:>9}  "
                  f"{m['sharpe']:>+9.2f}  {m['cost_adj_sharpe']:>+9.2f}  "
                  f"{m['sortino']:>+9.2f}  {m['calmar']:>+9.2f}  "
                  f"{m['max_dd']:>+9.1%}  {m['cvar95']:>+9.1%}  "
                  f"{m['total_return']:>+9.1%}  {m['turnover']:>+9.2f}")
        print()

    # ------------------------------------------------------------------
    # Aggregate across tickers
    # ------------------------------------------------------------------
    if per_ticker:
        print("=" * 80)
        print("  AGGREGATE (mean across tickers)")
        print("=" * 80)
        strategy_names = ["bh", "ma200", "vol_bh", "mom_only", "mom_vol"]
        keys = ["sharpe", "cost_adj_sharpe", "sortino", "calmar",
                "max_dd", "cvar95", "total_return", "turnover"]
        print("  " + "  ".join(f"{k:>15}" for k in ["Strategy", *keys]))
        for strat in strategy_names:
            row = [strat]
            for k in keys:
                vals = [per_ticker[t][strat][k] for t in per_ticker]
                row.append(f"{np.mean(vals):+.3f}")
            print("  " + "  ".join(f"{v:>15}" for v in row))
        print()

    print("=" * 80)
    print(f"  Output saved to {_out_path}")
    print("=" * 80)
    _out_file.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
