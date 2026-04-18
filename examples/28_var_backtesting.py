"""Example 28 — VaR Backtesting Report
====================================

The canonical risk management exercise: compute daily Value-at-Risk
using progressively more sophisticated models and backtest whether
the VaR estimates are correctly calibrated.

This is what every bank runs daily for Basel III/IV regulatory
compliance.  The question: "Did we lose more than our VaR forecast
on more days than expected?"

Models compared:
  1. Historical Simulation   empirical quantile of trailing returns
  2. Parametric Normal       trailing vol × Gaussian quantile
  3. Parametric Student-t    trailing vol × fitted Student-t quantile
  4. GARCH-Normal            GJR-GARCH conditional vol × Gaussian
  5. GJR-skewt               GJR-GARCH conditional vol × skewed-t
  6. GJR-VIX                 VIX-blended vol (65/35) × skewed-t

Statistical tests:
  - Kupiec (1995): unconditional coverage — correct violation rate?
  - Christoffersen (1998): independence — are violations clustered?
  - Basel traffic light: Green / Yellow / Red zone

Use case:
  This is the single most common quant risk interview topic and a
  regulatory requirement for every bank's trading book.
"""

import time
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm, t as t_dist, chi2, binom
from arch import arch_model
from arch.univariate.distribution import SkewStudent

# ── Configuration ────────────────────────────────────────────────────
TICKER = "SPY"
START = "2012-01-01"
END = "2026-12-31"
LOOKBACK = 504
REFIT_EVERY = 21
ALPHAS = [0.01, 0.05]
SCALE = 100
VIX_GARCH_W = 0.65
MODELS = ["Historical", "Normal", "Student-t",
          "GARCH-Normal", "GJR-skewt", "GJR-VIX"]


# ── Statistical tests ───────────────────────────────────────────────

def kupiec_test(violations, n_obs, alpha):
    """Kupiec (1995) unconditional coverage LR test.

    H0: true violation rate equals the nominal rate alpha.
    """
    n1 = int(np.sum(violations))
    n0 = n_obs - n1
    if n1 == 0:
        return 0.0, 1.0
    if n0 == 0:
        return np.inf, 0.0
    pi_hat = n1 / n_obs
    lr = 2 * (n1 * np.log(pi_hat / alpha)
              + n0 * np.log((1 - pi_hat) / (1 - alpha)))
    return lr, float(1 - chi2.cdf(lr, 1))


def christoffersen_test(violations):
    """Christoffersen (1998) independence LR test.

    H0: violations are independently distributed (no clustering).
    """
    v = np.asarray(violations, dtype=int)
    n00 = n01 = n10 = n11 = 0
    for i in range(1, len(v)):
        n00 += (v[i - 1] == 0) and (v[i] == 0)
        n01 += (v[i - 1] == 0) and (v[i] == 1)
        n10 += (v[i - 1] == 1) and (v[i] == 0)
        n11 += (v[i - 1] == 1) and (v[i] == 1)
    total = n00 + n01 + n10 + n11
    if total == 0:
        return 0.0, 1.0
    pi = (n01 + n11) / total
    if pi in (0.0, 1.0):
        return 0.0, 1.0
    pi_01 = n01 / max(n00 + n01, 1)
    pi_11 = n11 / max(n10 + n11, 1)
    if pi_01 in (0.0, 1.0) or pi_11 in (0.0, 1.0):
        return 0.0, 1.0
    lr = 2 * (n00 * np.log(1 - pi_01) + n01 * np.log(pi_01)
              + n10 * np.log(1 - pi_11) + n11 * np.log(pi_11)
              - (n00 + n10) * np.log(1 - pi)
              - (n01 + n11) * np.log(pi))
    return lr, float(1 - chi2.cdf(max(lr, 0), 1))


def basel_zone(n_violations, n_obs, alpha):
    """Basel traffic-light zone via binomial p-value."""
    p = 1 - binom.cdf(max(n_violations - 1, 0), n_obs, alpha)
    if p > 0.05:
        return "Green"
    elif p > 0.0001:
        return "Yellow"
    return "Red"


# ── Main ─────────────────────────────────────────────────────────────
t0 = time.time()
output_lines = []


def log(msg=""):
    print(msg)
    output_lines.append(msg)


log("=" * 80)
log("  VaR BACKTESTING REPORT")
log(f"  {TICKER}  |  Lookback={LOOKBACK}d  Refit={REFIT_EVERY}d")
log(f"  VaR levels: {', '.join(f'{a:.0%}' for a in ALPHAS)}")
log("=" * 80)

# ── Download data ────────────────────────────────────────────────────
spy_df = yf.download(TICKER, start=START, end=END, progress=False)
vix_df = yf.download("^VIX", start=START, end=END, progress=False)

spy_close = spy_df["Close"].squeeze().dropna()
vix_close = vix_df["Close"].squeeze().dropna()

returns = spy_close.pct_change().dropna()
vix = vix_close.reindex(returns.index, method="ffill")
vix_daily_vol = vix / (100 * np.sqrt(252))  # VIX → daily vol

n_total = len(returns)
n_eval = n_total - LOOKBACK
eval_start = returns.index[LOOKBACK]
eval_end = returns.index[-1]

log(f"\n  {TICKER}: {n_total} daily returns")
log(f"  Eval: {eval_start.strftime('%Y-%m-%d')} -> "
    f"{eval_end.strftime('%Y-%m-%d')}  ({n_eval} days)")

# ── Walk-forward VaR computation ─────────────────────────────────────
ret_arr = returns.values
var_forecasts = {a: {m: np.full(n_eval, np.nan) for m in MODELS} for a in ALPHAS}
actual_ret = ret_arr[LOOKBACK:]

ss = SkewStudent()
refit_points = list(range(LOOKBACK, n_total, REFIT_EVERY))
n_refits = len(refit_points)

# GARCH state between refits
gp = {}

for ri, t_start in enumerate(refit_points):
    window = ret_arr[t_start - LOOKBACK:t_start]

    # ── Simple models ──
    w_mean = np.mean(window)
    w_std = np.std(window, ddof=1)
    try:
        nu_t, loc_t, scale_t = t_dist.fit(window)
        nu_t = max(nu_t, 2.1)
    except Exception:
        nu_t, loc_t, scale_t = 5.0, w_mean, w_std

    # ── GARCH-Normal ──
    try:
        am = arch_model(window * SCALE, vol="GARCH", p=1, o=1, q=1,
                        dist="normal", mean="Constant")
        res = am.fit(disp="off", show_warning=False)
        cv = np.asarray(res.conditional_volatility)
        gp["GARCH-Normal"] = dict(
            omega=res.params["omega"], alpha=res.params["alpha[1]"],
            gamma=res.params.get("gamma[1]", 0), beta=res.params["beta[1]"],
            mu=res.params["mu"],
            h=cv[-1] ** 2,
        )
    except Exception:
        pass

    # ── GJR-skewt ──
    try:
        am = arch_model(window * SCALE, vol="GARCH", p=1, o=1, q=1,
                        dist="skewt", mean="Constant")
        res = am.fit(disp="off", show_warning=False)
        cv = np.asarray(res.conditional_volatility)
        gp["GJR-skewt"] = dict(
            omega=res.params["omega"], alpha=res.params["alpha[1]"],
            gamma=res.params.get("gamma[1]", 0), beta=res.params["beta[1]"],
            mu=res.params["mu"],
            h=cv[-1] ** 2,
            eta=res.params["eta"], lam=res.params["lambda"],
        )
        gp["GJR-VIX"] = dict(gp["GJR-skewt"])
    except Exception:
        pass

    # ── Daily forecasts until next refit ──
    t_end = min(t_start + REFIT_EVERY, n_total)
    for t in range(t_start, t_end):
        idx = t - LOOKBACK

        for a in ALPHAS:
            # Historical: empirical quantile
            var_forecasts[a]["Historical"][idx] = np.percentile(window, a * 100)

            # Normal: parametric Gaussian
            var_forecasts[a]["Normal"][idx] = w_mean + w_std * norm.ppf(a)

            # Student-t: parametric with fitted df
            var_forecasts[a]["Student-t"][idx] = t_dist.ppf(
                a, nu_t, loc=loc_t, scale=scale_t)

            # GARCH-Normal
            if "GARCH-Normal" in gp:
                g = gp["GARCH-Normal"]
                vol = np.sqrt(g["h"]) / SCALE
                var_forecasts[a]["GARCH-Normal"][idx] = (
                    g["mu"] / SCALE + vol * norm.ppf(a))

            # GJR-skewt
            if "GJR-skewt" in gp:
                g = gp["GJR-skewt"]
                vol = np.sqrt(g["h"]) / SCALE
                q = ss.ppf(a, parameters=np.array([g["eta"], g["lam"]]))
                var_forecasts[a]["GJR-skewt"][idx] = g["mu"] / SCALE + vol * q

            # GJR-VIX (blend with VIX)
            if "GJR-VIX" in gp:
                g = gp["GJR-VIX"]
                garch_vol = np.sqrt(g["h"]) / SCALE
                v_daily = float(vix_daily_vol.iloc[t])
                blended = VIX_GARCH_W * garch_vol + (1 - VIX_GARCH_W) * v_daily
                q = ss.ppf(a, parameters=np.array([g["eta"], g["lam"]]))
                var_forecasts[a]["GJR-VIX"][idx] = g["mu"] / SCALE + blended * q

        # Update GARCH conditional variance using realised return
        r_t = ret_arr[t]
        for mname in ["GARCH-Normal", "GJR-skewt", "GJR-VIX"]:
            if mname in gp:
                g = gp[mname]
                eps = r_t * SCALE - g["mu"]
                g["h"] = (g["omega"]
                          + (g["alpha"] + g["gamma"] * float(eps < 0)) * eps ** 2
                          + g["beta"] * g["h"])

    if (ri + 1) % 25 == 0:
        log(f"    ... {ri + 1}/{n_refits} refits complete")

elapsed = time.time() - t0
log(f"\n  Walk-forward complete in {elapsed:.0f}s")

# ── Results ──────────────────────────────────────────────────────────
for a in ALPHAS:
    expected = int(n_eval * a)
    log(f"\n{'=' * 80}")
    log(f"  MODEL COMPARISON — {a:.0%} VaR")
    log(f"  Expected violations: ~{expected} out of {n_eval} days ({a:.2%})")
    log("=" * 80)
    log(f"  {'Model':<16s} {'Violations':>10s} {'Rate':>8s} "
        f"{'Kupiec p':>10s} {'Christ. p':>10s} {'Basel':>8s} {'AvgSeverity':>12s}")

    for m in MODELS:
        var_arr = var_forecasts[a][m]
        valid = ~np.isnan(var_arr)
        if valid.sum() == 0:
            continue
        violations = actual_ret[valid] < var_arr[valid]
        n_v = int(violations.sum())
        n_obs = int(valid.sum())
        rate = n_v / n_obs

        _, kp = kupiec_test(violations, n_obs, a)
        _, cp = christoffersen_test(violations)
        bz = basel_zone(n_v, n_obs, a)

        # Average severity: when violated, how bad? (actual / VaR)
        if n_v > 0:
            breach_actual = actual_ret[valid][violations]
            breach_var = var_arr[valid][violations]
            avg_sev = np.mean(breach_actual / breach_var)
            sev_str = f"{avg_sev:+.2f}×VaR"
        else:
            sev_str = "N/A"

        log(f"  {m:<16s} {n_v:>10d} {rate:>7.2%} "
            f"{kp:>10.3f} {cp:>10.3f} {bz:>8s} {sev_str:>12s}")

# ── Violation clustering analysis ────────────────────────────────────
log(f"\n{'=' * 80}")
log("  VIOLATION CLUSTERING ANALYSIS")
log("=" * 80)
log(f"  {'Model':<16s} {'MaxConsec':>10s} {'AvgGap':>10s} {'ClusterRatio':>13s}")

for m in MODELS:
    var_arr = var_forecasts[0.01][m]
    valid = ~np.isnan(var_arr)
    if valid.sum() == 0:
        continue
    violations = (actual_ret[valid] < var_arr[valid]).astype(int)

    # Max consecutive violations
    max_consec = 0
    current = 0
    for v in violations:
        if v:
            current += 1
            max_consec = max(max_consec, current)
        else:
            current = 0

    # Average gap between violations
    violation_idx = np.where(violations)[0]
    if len(violation_idx) > 1:
        gaps = np.diff(violation_idx)
        avg_gap = np.mean(gaps)
    else:
        avg_gap = float("nan")

    # Cluster ratio: fraction of violations that follow another violation
    if len(violation_idx) > 1:
        follows = np.sum(np.diff(violation_idx) == 1)
        cluster_ratio = follows / max(len(violation_idx) - 1, 1)
    else:
        cluster_ratio = 0.0

    log(f"  {m:<16s} {max_consec:>10d} {avg_gap:>10.1f} {cluster_ratio:>12.1%}")

# ── Crisis period analysis ───────────────────────────────────────────
log(f"\n{'=' * 80}")
log("  CRISIS PERIOD ANALYSIS (1% VaR)")
log("=" * 80)

crisis_periods = [
    ("COVID (Feb-Mar 2020)", "2020-02-19", "2020-03-23"),
    ("2022 Bear (Jan-Oct)", "2022-01-03", "2022-10-12"),
    ("Aug 2024 Selloff",    "2024-07-15", "2024-08-15"),
]

for name, start, end in crisis_periods:
    mask = (returns.index[LOOKBACK:] >= start) & (returns.index[LOOKBACK:] <= end)
    n_days = mask.sum()
    if n_days == 0:
        continue

    log(f"\n  {name}  ({n_days} days)")
    log(f"  {'Model':<16s} {'Violations':>10s} {'Rate':>8s} "
        f"{'Worst Loss':>11s} {'VaR@Worst':>11s}")

    for m in MODELS:
        var_arr = var_forecasts[0.01][m]
        valid = ~np.isnan(var_arr)
        crisis = mask & valid
        if crisis.sum() == 0:
            continue
        violations = actual_ret[crisis] < var_arr[crisis]
        n_v = int(violations.sum())
        n_c = int(crisis.sum())
        rate = n_v / n_c

        worst_idx = np.argmin(actual_ret[crisis])
        worst_loss = actual_ret[crisis][worst_idx]
        var_at_worst = var_arr[crisis][worst_idx]

        log(f"  {m:<16s} {n_v:>10d} {rate:>7.2%}  "
            f"{worst_loss:>10.2%}  {var_at_worst:>10.2%}")

# ── VaR accuracy by volatility regime ────────────────────────────────
log(f"\n{'=' * 80}")
log("  VaR ACCURACY BY VOLATILITY REGIME (1% VaR)")
log("=" * 80)

trailing_vol = pd.Series(ret_arr, index=returns.index).rolling(63).std().iloc[LOOKBACK:].values
vol_median = np.nanmedian(trailing_vol)

for regime, label in [(trailing_vol <= vol_median, "Low-vol"),
                      (trailing_vol > vol_median, "High-vol")]:
    n_days = regime.sum()
    if n_days == 0:
        continue
    log(f"\n  {label} regime ({n_days} days)")
    log(f"  {'Model':<16s} {'Violations':>10s} {'Rate':>8s}  {'Expected':>8s}")
    for m in MODELS:
        var_arr = var_forecasts[0.01][m]
        valid = ~np.isnan(var_arr) & regime
        if valid.sum() == 0:
            continue
        violations = actual_ret[valid] < var_arr[valid]
        n_v = int(violations.sum())
        n_obs = int(valid.sum())
        rate = n_v / n_obs
        log(f"  {m:<16s} {n_v:>10d} {rate:>7.2%}   {0.01:>7.2%}")

# ── Interpretation ───────────────────────────────────────────────────
log(f"\n{'=' * 80}")
log("  INTERPRETATION")
log("=" * 80)
log("""
  Value-at-Risk answers: "What is the worst daily loss we expect
  with α% probability?"  A 1% VaR of -2.3% means: "On 99 out of
  100 days, the loss should be less than 2.3%."

  The backtest checks two things:
  1. UNCONDITIONAL COVERAGE (Kupiec): Does the violation rate match
     the target?  If 1% VaR is violated 3% of the time, the model
     under-estimates risk.  Kupiec p > 0.05 → pass.

  2. INDEPENDENCE (Christoffersen): Are violations randomly scattered,
     or do they cluster?  Clustered violations mean the model is slow
     to adapt to regime changes.  Christoffersen p > 0.05 → pass.

  3. BASEL TRAFFIC LIGHT: The regulator's view.  Green = acceptable.
     Yellow = warning.  Red = capital surcharge.

  What the models reveal:
  - Normal/Historical: cheap but miss fat tails -- too many violations
  - Student-t: captures fat tails -- fewer violations
  - GARCH: adapts vol to recent data -- violations spread out, not clustered
  - GJR-skewt: captures both fat tails AND asymmetry -- best coverage
  - GJR-VIX: adds market-implied vol -- fastest regime adaptation

  The progression from Normal -> GJR-VIX shows exactly the evolution
  a risk quant goes through: from textbook to production.
""")
total = time.time() - t0
log(f"  Total runtime: {total:.0f}s")

with open("_var_backtest_output.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(output_lines))
log(f"  Output saved to _var_backtest_output.txt")
log("=" * 80)
