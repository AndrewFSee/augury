"""Example 24 — Variance Risk Premium Strategy.

Demonstrates the core value proposition of a probabilistic volatility
forecast: identifying when the options market overprices risk.

The **Variance Risk Premium (VRP)** is the difference between implied
volatility (what the market expects) and a model's estimate of future
realized volatility.  On average, IV > RV because option sellers demand
compensation for bearing tail risk.  When this premium is unusually wide,
it signals excess fear — a good time to be long.  When it compresses,
risk may be underpriced.

Strategy logic:
  1. At each 21-day rebalance, fit EnsembleForecaster on trailing 504 days
  2. Extract forecast_realized_vol (from MC price paths)
  3. Get current VIX level (annualised implied vol)
  4. VRP = VIX/100 - forecast_vol  (VIX is quoted in % points)
  5. Signal:
     - VRP > threshold → market overpricing risk → full long
     - VRP near zero → fair pricing → partial long
     - VRP < 0 → model says MORE vol than market → reduce/exit

Benchmarks:
  - Buy & Hold
  - Trailing VRP (VIX - trailing 63-day realized vol) — to test whether
    the MC forecast vol adds anything beyond simple historical vol
  - Vol-target B&H (no VRP, just constant vol targeting)

This directly answers: does the forward simulation (Heston + Merton JD +
RS-GBM) provide a better vol estimate than trailing realized vol for
harvesting the variance risk premium?
"""

from __future__ import annotations

import sys
import time
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
_out_path = "_vrp_strategy_output.txt"
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
START = "2010-01-01"
END = "2026-12-31"

LOOKBACK = 504
HORIZON = 21
STEP = 21
EMBARGO = 21
N_PATHS = 2000
COST_BPS = 10
VOL_TARGET = 0.15

# VRP thresholds (in annualised vol points)
VRP_HIGH = 0.05     # VRP > 5 vol pts → full long
VRP_LOW = -0.02     # VRP < -2 vol pts → exit

from synthquant.data.ingest import DataIngestor  # noqa: E402
from synthquant.ensemble_forecast import EnsembleForecaster  # noqa: E402


# ---------------------------------------------------------------------------
# Metrics
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
# VRP exposure function
# ---------------------------------------------------------------------------
def vrp_exposure(vrp: float) -> float:
    """Convert VRP level to position size [0, 1].

    VRP = IV - forecast_vol (or trailing_vol).
    High VRP → market overpricing fear → go long.
    """
    if vrp >= VRP_HIGH:
        return 1.0
    elif vrp <= VRP_LOW:
        return 0.0
    else:
        # Linear interpolation
        return float((vrp - VRP_LOW) / (VRP_HIGH - VRP_LOW))


# ---------------------------------------------------------------------------
# Walk-forward
# ---------------------------------------------------------------------------
def main() -> int:
    t_start = time.perf_counter()

    print("=" * 80)
    print("  VARIANCE RISK PREMIUM STRATEGY")
    print(f"  SPY only  |  {START} → {END}")
    print(f"  Lookback={LOOKBACK}d  Horizon={HORIZON}d  Embargo={EMBARGO}d  "
          f"Paths={N_PATHS}  Cost={COST_BPS}bps")
    print(f"  VRP thresholds: full long > {VRP_HIGH:.0%}, "
          f"exit < {VRP_LOW:.0%}")
    print("=" * 80)

    # ------------------------------------------------------------------
    # Fetch SPY + VIX
    # ------------------------------------------------------------------
    ingestor = DataIngestor()
    spy_df = ingestor.fetch("SPY", start=START, end=END)
    vix_df = ingestor.fetch("^VIX", start=START, end=END)

    # Align on common dates
    common_dates = spy_df.index.intersection(vix_df.index)
    spy_df = spy_df.loc[common_dates]
    vix_df = vix_df.loc[common_dates]

    spy_close = spy_df["close"].to_numpy(dtype=float)
    vix_close = vix_df["close"].to_numpy(dtype=float)
    spy_returns = np.diff(np.log(spy_close))
    # VIX is quoted as annualised vol %, so VIX=20 means 20% ann vol
    vix_levels = vix_close[1:] / 100.0  # align with returns (day after)

    n = len(spy_returns)
    print(f"\n  SPY: {n} daily returns")
    print(f"  VIX: mean={np.mean(vix_levels):.3f}, "
          f"median={np.median(vix_levels):.3f}, "
          f"range=[{np.min(vix_levels):.3f}, {np.max(vix_levels):.3f}]")

    # ------------------------------------------------------------------
    # Strategies
    # ------------------------------------------------------------------
    strats = {name: StratMetrics(name) for name in [
        "bh", "vol_target", "vrp_trailing", "vrp_forecast"]}
    prev_expo = {k: 0.0 for k in strats}

    # Diagnostics
    diag = {
        "date": [], "vix": [], "trailing_vol": [], "forecast_vol": [],
        "vrp_trailing": [], "vrp_forecast": [],
        "expo_trailing": [], "expo_forecast": [],
        "fwd_return": [], "fwd_vol": [],
    }

    t = LOOKBACK
    n_windows = 0
    while t + EMBARGO + HORIZON <= n:
        train_end = t
        eval_start = t + EMBARGO
        eval_end = eval_start + HORIZON

        actual_fwd = float(np.sum(spy_returns[eval_start:eval_end]))
        actual_fwd_vol = float(np.std(spy_returns[eval_start:eval_end]) * np.sqrt(252))
        current_vix = float(vix_levels[train_end - 1])  # last known VIX

        # Trailing 63-day realized vol (annualised)
        recent = spy_returns[max(0, train_end - 63):train_end]
        trailing_vol = float(np.std(recent) * np.sqrt(252))

        # Build forecast
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fc = EnsembleForecaster(
                    detector="hmm", n_regimes=2, n_paths=N_PATHS,
                    random_state=42, use_filtered=True,
                )
                fc.fit(spy_returns[train_end - LOOKBACK:train_end])
                forecast = fc.forecast(
                    horizon=HORIZON,
                    current_price=float(spy_close[train_end]),
                )
                forecast_vol = forecast.forecast_realized_vol
        except Exception:
            forecast_vol = trailing_vol  # fallback

        # VRP signals
        vrp_t = current_vix - trailing_vol
        vrp_f = current_vix - forecast_vol

        # Exposures
        expo_bh = 1.0
        expo_vol_target = min(VOL_TARGET / trailing_vol, 1.0) if trailing_vol > 0 else 1.0
        expo_vrp_trailing = vrp_exposure(vrp_t)
        expo_vrp_forecast = vrp_exposure(vrp_f)

        exposures = {
            "bh": expo_bh,
            "vol_target": expo_vol_target,
            "vrp_trailing": expo_vrp_trailing,
            "vrp_forecast": expo_vrp_forecast,
        }

        for name, expo in exposures.items():
            turnover = abs(expo - prev_expo[name])
            prev_expo[name] = expo
            strats[name].pnl = np.append(strats[name].pnl, expo * actual_fwd)
            strats[name].turnover_sum += turnover
            strats[name].exposure_sum += abs(expo)
            strats[name].n_windows += 1

        # Diagnostics
        diag["date"].append(train_end)
        diag["vix"].append(current_vix)
        diag["trailing_vol"].append(trailing_vol)
        diag["forecast_vol"].append(forecast_vol)
        diag["vrp_trailing"].append(vrp_t)
        diag["vrp_forecast"].append(vrp_f)
        diag["expo_trailing"].append(expo_vrp_trailing)
        diag["expo_forecast"].append(expo_vrp_forecast)
        diag["fwd_return"].append(actual_fwd)
        diag["fwd_vol"].append(actual_fwd_vol)

        n_windows += 1
        if n_windows % 25 == 0:
            print(f"    ... {n_windows} windows complete")

        t += STEP

    elapsed = time.perf_counter() - t_start

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------
    print(f"\n  Completed {n_windows} windows in {elapsed:.0f}s\n")

    print("=" * 80)
    print("  STRATEGY PERFORMANCE")
    print("=" * 80)

    headers = ["Strategy", "Sharpe", "CostAdj", "Sortino", "Calmar",
               "MaxDD", "CVaR95", "TotRet", "AvgExpo", "Turn"]
    print("  " + "".join(f"{h:>10}" for h in headers))

    for name in ["bh", "vol_target", "vrp_trailing", "vrp_forecast"]:
        m = strats[name].finalize()
        print(f"  {name:>10}"
              f"{m['sharpe']:>+10.2f}{m['cost_adj_sharpe']:>+10.2f}"
              f"{m['sortino']:>+10.2f}{m['calmar']:>+10.2f}"
              f"{m['max_dd']:>+10.1%}{m['cvar95']:>+10.1%}"
              f"{m['total_return']:>+10.1%}{m['avg_expo']:>10.2f}"
              f"{m['turnover']:>10.2f}")

    # ------------------------------------------------------------------
    # VRP Diagnostics
    # ------------------------------------------------------------------
    vrp_t_arr = np.array(diag["vrp_trailing"])
    vrp_f_arr = np.array(diag["vrp_forecast"])
    fv_arr = np.array(diag["forecast_vol"])
    tv_arr = np.array(diag["trailing_vol"])
    vix_arr = np.array(diag["vix"])
    fwd_ret = np.array(diag["fwd_return"])
    fwd_vol = np.array(diag["fwd_vol"])

    print(f"\n{'=' * 80}")
    print("  VRP SIGNAL DIAGNOSTICS")
    print(f"{'=' * 80}")

    print(f"\n  VRP Statistics:")
    print(f"  {'':>25} {'Trailing':>12} {'Forecast':>12}")
    print(f"  {'Mean VRP':>25} {vrp_t_arr.mean():>12.4f} {vrp_f_arr.mean():>12.4f}")
    print(f"  {'Std VRP':>25} {vrp_t_arr.std():>12.4f} {vrp_f_arr.std():>12.4f}")
    print(f"  {'Min VRP':>25} {vrp_t_arr.min():>12.4f} {vrp_f_arr.min():>12.4f}")
    print(f"  {'Max VRP':>25} {vrp_t_arr.max():>12.4f} {vrp_f_arr.max():>12.4f}")
    print(f"  {'% windows VRP > 0':>25} "
          f"{100 * (vrp_t_arr > 0).mean():>11.1f}% "
          f"{100 * (vrp_f_arr > 0).mean():>11.1f}%")

    # Does VRP predict forward returns?
    from scipy.stats import spearmanr

    rho_t_ret, p_t_ret = spearmanr(vrp_t_arr, fwd_ret)
    rho_f_ret, p_f_ret = spearmanr(vrp_f_arr, fwd_ret)
    print(f"\n  VRP → Forward Return (Spearman):")
    print(f"  {'Trailing VRP vs fwd ret':>30}: rho={rho_t_ret:+.3f} (p={p_t_ret:.3f})")
    print(f"  {'Forecast VRP vs fwd ret':>30}: rho={rho_f_ret:+.3f} (p={p_f_ret:.3f})")

    # Does VRP predict forward vol?
    rho_t_vol, p_t_vol = spearmanr(vrp_t_arr, fwd_vol)
    rho_f_vol, p_f_vol = spearmanr(vrp_f_arr, fwd_vol)
    print(f"\n  VRP → Forward Vol (Spearman):")
    print(f"  {'Trailing VRP vs fwd vol':>30}: rho={rho_t_vol:+.3f} (p={p_t_vol:.3f})")
    print(f"  {'Forecast VRP vs fwd vol':>30}: rho={rho_f_vol:+.3f} (p={p_f_vol:.3f})")

    # Forecast vol vs trailing vol accuracy
    rho_fv, _ = spearmanr(fv_arr, fwd_vol)
    rho_tv, _ = spearmanr(tv_arr, fwd_vol)
    mae_fv = np.mean(np.abs(fv_arr - fwd_vol))
    mae_tv = np.mean(np.abs(tv_arr - fwd_vol))
    print(f"\n  Vol Forecast Accuracy (vs forward realized vol):")
    print(f"  {'':>25} {'Forecast':>12} {'Trailing':>12}")
    print(f"  {'Spearman rho':>25} {rho_fv:>+12.3f} {rho_tv:>+12.3f}")
    print(f"  {'MAE':>25} {mae_fv:>12.4f} {mae_tv:>12.4f}")
    vol_winner = "FORECAST" if rho_fv > rho_tv else "TRAILING"
    print(f"  {'Better predictor':>25} {vol_winner:>12}")

    # ------------------------------------------------------------------
    # Conditional analysis: when VRP is high vs low
    # ------------------------------------------------------------------
    print(f"\n{'=' * 80}")
    print("  CONDITIONAL RETURN ANALYSIS")
    print(f"{'=' * 80}")

    vrp = vrp_f_arr  # use forecast VRP
    quantiles = [0, 0.20, 0.40, 0.60, 0.80, 1.0]
    labels = ["Q1 (low)", "Q2", "Q3", "Q4", "Q5 (high)"]
    thresholds = np.quantile(vrp, quantiles)

    print(f"\n  Forward 21-day return by forecast VRP quintile:")
    print(f"  {'Quintile':<12} {'VRP range':>18} {'N':>5} "
          f"{'Avg Ret':>10} {'Hit Rate':>10} {'Avg Vol':>10}")

    for i, label in enumerate(labels):
        lo, hi = thresholds[i], thresholds[i + 1]
        if i == len(labels) - 1:
            mask = (vrp >= lo) & (vrp <= hi)
        else:
            mask = (vrp >= lo) & (vrp < hi)
        n = mask.sum()
        if n == 0:
            continue
        avg_ret = fwd_ret[mask].mean()
        hit = (fwd_ret[mask] > 0).mean()
        avg_v = fwd_vol[mask].mean()
        print(f"  {label:<12} {lo:>+8.3f} → {hi:>+6.3f} {n:>5} "
              f"{avg_ret:>+10.4f} {hit:>10.1%} {avg_v:>10.4f}")

    # ------------------------------------------------------------------
    # Where does the forecast VRP differ from trailing VRP?
    # ------------------------------------------------------------------
    print(f"\n{'=' * 80}")
    print("  WHEN DOES THE FORECAST ADD VALUE?")
    print(f"{'=' * 80}")

    # Cases where the two VRP signals disagree on exposure
    expo_t = np.array(diag["expo_trailing"])
    expo_f = np.array(diag["expo_forecast"])

    # Define "disagree" as >0.20 difference in exposure
    disagree = np.abs(expo_t - expo_f) > 0.20
    n_disagree = disagree.sum()
    print(f"\n  Windows with >20% exposure difference: "
          f"{n_disagree} / {n_windows} ({100 * n_disagree / n_windows:.1f}%)")

    if n_disagree > 5:
        # When forecast says more exposed
        fcast_more = disagree & (expo_f > expo_t)
        fcast_less = disagree & (expo_f < expo_t)

        if fcast_more.sum() > 0:
            avg_ret_more = fwd_ret[fcast_more].mean()
            hit_more = (fwd_ret[fcast_more] > 0).mean()
            print(f"  Forecast MORE exposed ({fcast_more.sum()} windows): "
                  f"avg fwd ret={avg_ret_more:+.4f}, hit={hit_more:.1%}")

        if fcast_less.sum() > 0:
            avg_ret_less = fwd_ret[fcast_less].mean()
            hit_less = (fwd_ret[fcast_less] > 0).mean()
            print(f"  Forecast LESS exposed ({fcast_less.sum()} windows): "
                  f"avg fwd ret={avg_ret_less:+.4f}, hit={hit_less:.1%}")

        # Did the forecast make the right call?
        # "Right call" = more exposed when fwd return positive, less when negative
        correct_more = fcast_more & (fwd_ret > 0)
        correct_less = fcast_less & (fwd_ret < 0)
        correct = (correct_more.sum() + correct_less.sum())
        total_disagree = fcast_more.sum() + fcast_less.sum()
        print(f"  Forecast correct when disagreeing: "
              f"{correct}/{total_disagree} ({100 * correct / total_disagree:.0f}%)")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n{'=' * 80}")
    print("  INTERPRETATION")
    print(f"{'=' * 80}")
    print("""
  The Variance Risk Premium (VRP) is the gap between what the options
  market charges for volatility (VIX) and what will actually happen
  (realized vol).  VIX systematically overestimates realized vol by
  ~2-5 vol points — this "insurance premium" is one of the most
  robust risk premia in equity markets.

  This example tests whether a probabilistic vol forecast (from
  Heston + Merton JD + Regime-Switching GBM simulation) provides a
  better VRP estimate than simple trailing realized vol.

  Key questions answered:
  1. Does VRP predict forward returns? (should be mildly positive)
  2. Does the forecast vol improve VRP timing vs trailing vol?
  3. When they disagree, who's right more often?

  The forecast's edge (if any) should show up specifically in:
  - Regime transitions (where trailing vol is stale but the HMM
    + stochastic vol models adapt faster)
  - Jump-diffusion scenarios (where Merton JD captures crash risk
    that trailing vol smooths over)
""")

    print(f"\n  Total runtime: {time.perf_counter() - t_start:.0f}s")
    print(f"  Output saved to {_out_path}")
    print("=" * 80)
    _out_file.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
