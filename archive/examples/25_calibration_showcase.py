"""Example 25 — Probabilistic Calibration Showcase
===================================================

The value of a probabilistic forecast isn't "beating the market" — it's
producing TRUSTWORTHY risk numbers.  When the model says there's a 5%
chance of losing more than X, losses should exceed X roughly 5% of the
time.  This is **calibration**.

This example runs a walk-forward backtest across 5 assets and measures:

1. **VaR coverage**: When the model says VaR(α) = X, do exceedances
   happen α% of the time?  Tested at 1%, 5%, 10%, 25%.
2. **Confidence interval coverage**: Do 50%, 80%, 90%, 95% forecast
   intervals contain the realised return at the nominal rate?
3. **PIT uniformity**: The Probability Integral Transform of a
   well-calibrated forecast should be Uniform(0,1).
4. **CRPS**: Continuous Ranked Probability Score — the gold standard
   for scoring probabilistic forecasts (lower = better).
5. **Vol calibration**: Is forecast vol close to realised vol?

Seven models compared:
  1. Gaussian:      trailing mean + trailing vol, normal distribution.
  2. Student-t:     same trailing statistics, but with MLE-fitted df.
  3. GARCH-t:       GARCH(1,1) conditional vol + Student-t innovations.
  4. GJR-skewt:     GJR-GARCH(1,1) + Hansen's skewed-t innovations.
  5. GJR-VIX:       GJR-skewt + 65/35 blend of GARCH vol with VIX implied vol.
  6. GJR-VIX+SKEW:  GJR-VIX + CBOE SKEW index adjusts skewness parameter.
  7. Ensemble:      HMM + Heston + Merton JD + RS-GBM Monte Carlo.
"""

from __future__ import annotations

import sys
import time
import warnings

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

warnings.filterwarnings("ignore")

# ── Output capture ──────────────────────────────────────────────────────
_OUT = "_calibration_output.txt"
_fh = open(_OUT, "w", encoding="utf-8")  # noqa: SIM115
_print = print


def print(*a, **kw):  # noqa: A001
    _print(*a, **kw)
    kw.pop("file", None)
    _print(*a, file=_fh, **kw)
    _fh.flush()


# ── Config ──────────────────────────────────────────────────────────────
TICKERS  = ["SPY", "QQQ", "IWM", "TLT", "GLD"]
START    = "2010-01-01"
END      = "2026-12-31"
LOOKBACK = 504
HORIZON  = 21
STEP     = 21
EMBARGO  = 21
N_PATHS  = 2000         # balance precision vs speed
N_T_SAMPLES = 10_000    # samples for Student-t CRPS

from arch.univariate.distribution import SkewStudent as _ArchSkewT  # noqa: E402

from synthquant.data.ingest import DataIngestor  # noqa: E402
from synthquant.ensemble_forecast import EnsembleForecaster  # noqa: E402
from synthquant.models.garch import GARCHModel  # noqa: E402

SKEW_SCALE = 750.0   # CBOE SKEW → lambda adjustment (tuned)
VIX_GARCH_W = 0.65   # GARCH weight in VIX blend (tuned from 0.50)

# ── Model registry (name, storage prefix) ───────────────────────────────
MODEL_INFO = [
    ("Ensemble",      "ens"),
    ("Student-t",     "t"),
    ("GARCH-t",       "gt"),
    ("GJR-skewt",     "gjr"),
    ("GJR-VIX",       "gvix"),
    ("GJR-VIX+SKEW",  "gvs"),
    ("Gaussian",      "gauss"),
]
MODEL_ABBREV = {
    "Ensemble": "E", "Student-t": "T", "GARCH-t": "Gt",
    "GJR-skewt": "GJR", "GJR-VIX": "VIX", "GJR-VIX+SKEW": "V+S",
    "Gaussian": "G",
}


# ════════════════════════════════════════════════════════════════════════
def main() -> int:  # noqa: C901, PLR0912, PLR0915
    t0 = time.perf_counter()
    rng = np.random.default_rng(42)

    print("=" * 80)
    print("  PROBABILISTIC CALIBRATION SHOWCASE")
    print(f"  {', '.join(TICKERS)}  |  {START} → {END}")
    print(f"  Lookback={LOOKBACK}d  Horizon={HORIZON}d  Embargo={EMBARGO}d  "
          f"Paths={N_PATHS}")
    print(f"  Walk-forward step={STEP}d")
    print("=" * 80)

    # ── 1. Data ─────────────────────────────────────────────────────────
    ingestor = DataIngestor()
    data: dict[str, dict] = {}
    for tk in TICKERS:
        df = ingestor.fetch(tk, start=START, end=END)
        close = df["close"].to_numpy(dtype=float)
        dates = df.index  # DatetimeIndex for VIX alignment
        rets = np.diff(np.log(close))
        data[tk] = {"close": close, "returns": rets, "dates": dates}
        print(f"  {tk}: {len(rets)} daily returns")

    # VIX (CBOE Volatility Index) — forward-looking implied vol for S&P 500
    vix_df = ingestor.fetch("^VIX", start=START, end=END)
    vix_close = vix_df["close"]  # pd.Series with DatetimeIndex
    print(f"  VIX: {len(vix_close)} observations")

    # SPY rolling 63-day vol (for cross-asset implied vol scaling)
    _spy_rets_s = pd.Series(
        data["SPY"]["returns"],
        index=data["SPY"]["dates"][1:],
    )
    spy_rolling_vol = _spy_rets_s.rolling(63, min_periods=21).std() * np.sqrt(252)

    # SKEW Index (CBOE tail risk — higher = heavier left tail)
    try:
        skew_df = ingestor.fetch("^SKEW", start=START, end=END)
        skew_close = skew_df["close"]
        print(f"  SKEW: {len(skew_close)} observations")
    except Exception:
        skew_close = None
        print("  SKEW: not available — skewness adjustment disabled")

    # ── 2. Walk-forward calibration test ────────────────────────────────
    VAR_ALPHAS = [0.01, 0.05, 0.10, 0.25]
    CI_LEVELS  = [0.50, 0.80, 0.90, 0.95]

    # Storage per ticker — one set of arrays per model
    results: dict[str, dict] = {}
    for tk in TICKERS:
        results[tk] = {"actual_ret": [], "actual_vol": []}
        for _, pfx in MODEL_INFO:
            results[tk][f"{pfx}_terminal_quantiles"] = []
            results[tk][f"{pfx}_ci_bounds"] = []
            results[tk][f"{pfx}_pit"] = []
            results[tk][f"{pfx}_crps"] = []
            results[tk][f"{pfx}_fvol"] = []

    total_windows = 0
    for tk in TICKERS:
        rets = data[tk]["returns"]
        close = data[tk]["close"]
        n = len(rets)
        nw = 0

        t = LOOKBACK
        while t + EMBARGO + HORIZON <= n:
            train = rets[t - LOOKBACK : t]
            actual_fwd = float(np.sum(rets[t + EMBARGO : t + EMBARGO + HORIZON]))
            actual_vol = float(
                np.std(rets[t + EMBARGO : t + EMBARGO + HORIZON]) * np.sqrt(252)
            )

            # ── Ensemble forecast ───────────────────────────────────
            try:
                fc = EnsembleForecaster(
                    detector="hmm", n_regimes=2, n_paths=N_PATHS,
                    random_state=42, use_filtered=True,
                )
                fc.fit(train)
                res = fc.forecast(
                    horizon=HORIZON,
                    current_price=float(close[t]),
                )
                sim_ret = res.terminal_returns  # (N_PATHS,)
                fvol = res.forecast_realized_vol
            except Exception:
                mu = float(np.mean(train)) * HORIZON
                sig = float(np.std(train)) * np.sqrt(HORIZON)
                sim_ret = rng.normal(mu, sig, N_PATHS)
                fvol = float(np.std(train) * np.sqrt(252))

            # VaR quantiles (left tail)
            ens_q = {}
            for a in VAR_ALPHAS:
                ens_q[a] = float(np.quantile(sim_ret, a))
            results[tk]["ens_terminal_quantiles"].append(ens_q)

            # Confidence intervals
            ens_ci = {}
            for lv in CI_LEVELS:
                lo = float(np.quantile(sim_ret, (1 - lv) / 2))
                hi = float(np.quantile(sim_ret, 1 - (1 - lv) / 2))
                ens_ci[lv] = (lo, hi)
            results[tk]["ens_ci_bounds"].append(ens_ci)

            # PIT: fraction of ensemble ≤ actual
            pit = float(np.mean(sim_ret <= actual_fwd))
            results[tk]["ens_pit"].append(pit)

            # CRPS (efficient sort-based O(n log n) formula)
            sorted_ens = np.sort(sim_ret)
            m = len(sorted_ens)
            mae_term = float(np.mean(np.abs(sorted_ens - actual_fwd)))
            spread_term = float(
                2 * np.sum(sorted_ens * np.arange(m)) / m**2
                - np.mean(sorted_ens)
            )
            crps = mae_term - spread_term
            results[tk]["ens_crps"].append(crps)
            results[tk]["ens_fvol"].append(fvol)

            # ── Gaussian benchmark ──────────────────────────────────
            trailing_mu = float(np.mean(train)) * HORIZON
            trailing_sig = float(np.std(train)) * np.sqrt(HORIZON)
            trailing_vol = float(np.std(train[-63:]) * np.sqrt(252))

            gauss_q = {}
            for a in VAR_ALPHAS:
                gauss_q[a] = float(sp_stats.norm.ppf(a, trailing_mu, trailing_sig))
            results[tk]["gauss_terminal_quantiles"].append(gauss_q)

            gauss_ci = {}
            for lv in CI_LEVELS:
                lo = float(sp_stats.norm.ppf((1 - lv) / 2, trailing_mu, trailing_sig))
                hi = float(sp_stats.norm.ppf(1 - (1 - lv) / 2, trailing_mu, trailing_sig))
                gauss_ci[lv] = (lo, hi)
            results[tk]["gauss_ci_bounds"].append(gauss_ci)

            gauss_pit = float(sp_stats.norm.cdf(actual_fwd, trailing_mu, trailing_sig))
            results[tk]["gauss_pit"].append(gauss_pit)

            # CRPS for Gaussian (closed-form)
            z = (actual_fwd - trailing_mu) / trailing_sig if trailing_sig > 0 else 0.0
            gauss_crps = float(
                trailing_sig * (z * (2 * sp_stats.norm.cdf(z) - 1)
                                + 2 * sp_stats.norm.pdf(z) - 1 / np.sqrt(np.pi))
            )
            results[tk]["gauss_crps"].append(gauss_crps)
            results[tk]["gauss_fvol"].append(trailing_vol)

            # ── Student-t benchmark ─────────────────────────────────
            # Fit df from daily training returns (MLE with free scale)
            df_t, _, _ = sp_stats.t.fit(train)
            df_t = max(df_t, 2.1)  # ensure finite variance

            # Same mean as Gaussian; adjust scale so total std matches
            # For t(df, loc, scale): std = scale * sqrt(df / (df - 2))
            # We want std = trailing_sig → scale = trailing_sig * sqrt((df-2)/df)
            t_scale = trailing_sig * np.sqrt((df_t - 2) / df_t)

            t_q = {}
            for a in VAR_ALPHAS:
                t_q[a] = float(sp_stats.t.ppf(a, df_t, loc=trailing_mu, scale=t_scale))
            results[tk]["t_terminal_quantiles"].append(t_q)

            t_ci = {}
            for lv in CI_LEVELS:
                lo = float(sp_stats.t.ppf((1 - lv) / 2, df_t, loc=trailing_mu, scale=t_scale))
                hi = float(sp_stats.t.ppf(1 - (1 - lv) / 2, df_t, loc=trailing_mu, scale=t_scale))
                t_ci[lv] = (lo, hi)
            results[tk]["t_ci_bounds"].append(t_ci)

            t_pit = float(sp_stats.t.cdf(actual_fwd, df_t, loc=trailing_mu, scale=t_scale))
            results[tk]["t_pit"].append(t_pit)

            # CRPS via sampling (10k samples — negligible cost, consistent method)
            t_samples = sp_stats.t.rvs(df_t, loc=trailing_mu, scale=t_scale,
                                       size=N_T_SAMPLES, random_state=rng)
            sorted_t = np.sort(t_samples)
            mt = len(sorted_t)
            mae_t = float(np.mean(np.abs(sorted_t - actual_fwd)))
            spread_t = float(
                2 * np.sum(sorted_t * np.arange(mt)) / mt**2
                - np.mean(sorted_t)
            )
            results[tk]["t_crps"].append(mae_t - spread_t)
            results[tk]["t_fvol"].append(trailing_vol)  # same vol as Gaussian

            # ── GARCH-t benchmark ───────────────────────────────────
            # Fit GARCH(1,1) with t-innovations to get conditional vol
            try:
                gm = GARCHModel(vol_model="GARCH", p=1, q=1, dist="t")
                gm.fit(train, scale=100.0)
                # 1-step conditional vol forecast, annualised
                garch_vol_1 = float(gm.forecast(horizon=1)[0])
                # Scale to horizon: sigma_H = sigma_1 * sqrt(H)
                gt_sig = garch_vol_1 / np.sqrt(252) * np.sqrt(HORIZON)
                # Get df from the fitted model's parameters
                gt_df = float(gm._result.params.get("nu", 5.0))
                gt_df = max(gt_df, 2.1)
                gt_fvol = garch_vol_1  # annualised conditional vol
            except Exception:
                # Fallback to trailing vol + default df
                gt_sig = trailing_sig
                gt_df = 5.0
                gt_fvol = trailing_vol

            gt_scale = gt_sig * np.sqrt((gt_df - 2) / gt_df)

            gt_q = {}
            for a in VAR_ALPHAS:
                gt_q[a] = float(sp_stats.t.ppf(a, gt_df, loc=trailing_mu, scale=gt_scale))
            results[tk]["gt_terminal_quantiles"].append(gt_q)

            gt_ci = {}
            for lv in CI_LEVELS:
                lo = float(sp_stats.t.ppf((1 - lv) / 2, gt_df, loc=trailing_mu, scale=gt_scale))
                hi = float(sp_stats.t.ppf(1 - (1 - lv) / 2, gt_df, loc=trailing_mu, scale=gt_scale))
                gt_ci[lv] = (lo, hi)
            results[tk]["gt_ci_bounds"].append(gt_ci)

            gt_pit = float(sp_stats.t.cdf(actual_fwd, gt_df, loc=trailing_mu, scale=gt_scale))
            results[tk]["gt_pit"].append(gt_pit)

            gt_samples = sp_stats.t.rvs(gt_df, loc=trailing_mu, scale=gt_scale,
                                        size=N_T_SAMPLES, random_state=rng)
            sorted_gt = np.sort(gt_samples)
            mgt = len(sorted_gt)
            mae_gt = float(np.mean(np.abs(sorted_gt - actual_fwd)))
            spread_gt = float(
                2 * np.sum(sorted_gt * np.arange(mgt)) / mgt**2
                - np.mean(sorted_gt)
            )
            results[tk]["gt_crps"].append(mae_gt - spread_gt)
            results[tk]["gt_fvol"].append(gt_fvol)

            # ── GJR-GARCH + skewed-t ────────────────────────────────
            # GJR-GARCH adds leverage (neg returns → higher vol)
            # Skewed-t adds asymmetric tails (fatter left tail)
            try:
                gjr_m = GARCHModel(vol_model="GJRGARCH", p=1, q=1, dist="skewt")
                gjr_m.fit(train, scale=100.0)
                gjr_vol_1 = float(gjr_m.forecast(horizon=1)[0])
                gjr_sig = gjr_vol_1 / np.sqrt(252) * np.sqrt(HORIZON)
                gjr_eta = float(gjr_m._result.params.get("eta", 5.0))
                gjr_eta = max(gjr_eta, 2.1)
                gjr_lam = float(gjr_m._result.params.get("lambda", 0.0))
                gjr_fvol = gjr_vol_1
            except Exception:
                gjr_sig = trailing_sig
                gjr_eta = 5.0
                gjr_lam = 0.0
                gjr_fvol = trailing_vol

            _skt = _ArchSkewT()
            _skt_params = np.array([gjr_eta, gjr_lam])

            gjr_q = {}
            for a in VAR_ALPHAS:
                gjr_q[a] = float(
                    trailing_mu + gjr_sig * _skt.ppf(a, parameters=_skt_params)
                )
            results[tk]["gjr_terminal_quantiles"].append(gjr_q)

            gjr_ci = {}
            for lv in CI_LEVELS:
                lo = float(
                    trailing_mu
                    + gjr_sig * _skt.ppf((1 - lv) / 2, parameters=_skt_params)
                )
                hi = float(
                    trailing_mu
                    + gjr_sig * _skt.ppf(1 - (1 - lv) / 2, parameters=_skt_params)
                )
                gjr_ci[lv] = (lo, hi)
            results[tk]["gjr_ci_bounds"].append(gjr_ci)

            gjr_z = (actual_fwd - trailing_mu) / gjr_sig if gjr_sig > 0 else 0.0
            gjr_pit = float(_skt.cdf(gjr_z, parameters=_skt_params))
            results[tk]["gjr_pit"].append(gjr_pit)

            # CRPS via inverse-transform sampling
            u_gjr = rng.uniform(0, 1, N_T_SAMPLES)
            gjr_z_samples = _skt.ppf(u_gjr, parameters=_skt_params)
            gjr_samples = trailing_mu + gjr_sig * gjr_z_samples
            sorted_gjr = np.sort(gjr_samples)
            mgjr = len(sorted_gjr)
            mae_gjr = float(np.mean(np.abs(sorted_gjr - actual_fwd)))
            spread_gjr = float(
                2 * np.sum(sorted_gjr * np.arange(mgjr)) / mgjr**2
                - np.mean(sorted_gjr)
            )
            results[tk]["gjr_crps"].append(mae_gjr - spread_gjr)
            results[tk]["gjr_fvol"].append(gjr_fvol)

            # ── GJR-VIX (VIX-blended vol) ───────────────────────────
            # Blend GARCH conditional vol with VIX implied vol (50/50).
            # VIX is forward-looking — jumps immediately in stress
            # while GARCH lags behind. Direct blend for SPY; for other
            # assets, scale VIX by (asset trailing vol / SPY trailing vol).
            current_date = data[tk]["dates"][t]
            vix_level = float(vix_close.asof(current_date))
            if np.isnan(vix_level) or vix_level <= 0:
                vix_level = 20.0
            vix_ann = vix_level / 100.0

            if tk == "SPY":
                implied_vol = vix_ann
            else:
                spy_tv = float(spy_rolling_vol.asof(current_date))
                if np.isnan(spy_tv) or spy_tv <= 0:
                    spy_tv = trailing_vol
                implied_vol = vix_ann * (trailing_vol / spy_tv) if spy_tv > 0 else vix_ann

            blended_vol = VIX_GARCH_W * gjr_fvol + (1 - VIX_GARCH_W) * implied_vol
            blended_sig = blended_vol / np.sqrt(252) * np.sqrt(HORIZON)

            gvix_q = {}
            for a in VAR_ALPHAS:
                gvix_q[a] = float(
                    trailing_mu + blended_sig * _skt.ppf(a, parameters=_skt_params)
                )
            results[tk]["gvix_terminal_quantiles"].append(gvix_q)

            gvix_ci = {}
            for lv in CI_LEVELS:
                lo = float(
                    trailing_mu
                    + blended_sig * _skt.ppf((1 - lv) / 2, parameters=_skt_params)
                )
                hi = float(
                    trailing_mu
                    + blended_sig * _skt.ppf(1 - (1 - lv) / 2, parameters=_skt_params)
                )
                gvix_ci[lv] = (lo, hi)
            results[tk]["gvix_ci_bounds"].append(gvix_ci)

            gvix_z = (actual_fwd - trailing_mu) / blended_sig if blended_sig > 0 else 0.0
            gvix_pit = float(_skt.cdf(gvix_z, parameters=_skt_params))
            results[tk]["gvix_pit"].append(gvix_pit)

            gvix_u = rng.uniform(0, 1, N_T_SAMPLES)
            gvix_z_samp = _skt.ppf(gvix_u, parameters=_skt_params)
            gvix_samp = trailing_mu + blended_sig * gvix_z_samp
            gvix_sorted = np.sort(gvix_samp)
            mgvix = len(gvix_sorted)
            mae_gvix = float(np.mean(np.abs(gvix_sorted - actual_fwd)))
            spread_gvix = float(
                2 * np.sum(gvix_sorted * np.arange(mgvix)) / mgvix**2
                - np.mean(gvix_sorted)
            )
            results[tk]["gvix_crps"].append(mae_gvix - spread_gvix)
            results[tk]["gvix_fvol"].append(blended_vol)

            # ── GJR-VIX+SKEW (VIX blend + SKEW-adjusted skewness) ───
            # Same vol as GJR-VIX, but adjusts skewness parameter
            # using CBOE SKEW index: higher SKEW → heavier left tail.
            if skew_close is not None:
                skew_lvl = float(skew_close.asof(current_date))
                if not np.isnan(skew_lvl) and skew_lvl > 0:
                    skew_adj = -(skew_lvl - 100) / SKEW_SCALE
                    gvs_lam = float(np.clip(gjr_lam + skew_adj, -0.9, 0.9))
                else:
                    gvs_lam = gjr_lam
            else:
                gvs_lam = gjr_lam
            _gvs_params = np.array([gjr_eta, gvs_lam])

            gvs_q = {}
            for a in VAR_ALPHAS:
                gvs_q[a] = float(
                    trailing_mu + blended_sig * _skt.ppf(a, parameters=_gvs_params)
                )
            results[tk]["gvs_terminal_quantiles"].append(gvs_q)

            gvs_ci = {}
            for lv in CI_LEVELS:
                lo = float(
                    trailing_mu
                    + blended_sig * _skt.ppf((1 - lv) / 2, parameters=_gvs_params)
                )
                hi = float(
                    trailing_mu
                    + blended_sig * _skt.ppf(1 - (1 - lv) / 2, parameters=_gvs_params)
                )
                gvs_ci[lv] = (lo, hi)
            results[tk]["gvs_ci_bounds"].append(gvs_ci)

            gvs_z = (actual_fwd - trailing_mu) / blended_sig if blended_sig > 0 else 0.0
            gvs_pit = float(_skt.cdf(gvs_z, parameters=_gvs_params))
            results[tk]["gvs_pit"].append(gvs_pit)

            gvs_u = rng.uniform(0, 1, N_T_SAMPLES)
            gvs_z_samp = _skt.ppf(gvs_u, parameters=_gvs_params)
            gvs_samp = trailing_mu + blended_sig * gvs_z_samp
            gvs_sorted = np.sort(gvs_samp)
            mgvs = len(gvs_sorted)
            mae_gvs = float(np.mean(np.abs(gvs_sorted - actual_fwd)))
            spread_gvs = float(
                2 * np.sum(gvs_sorted * np.arange(mgvs)) / mgvs**2
                - np.mean(gvs_sorted)
            )
            results[tk]["gvs_crps"].append(mae_gvs - spread_gvs)
            results[tk]["gvs_fvol"].append(blended_vol)

            results[tk]["actual_ret"].append(actual_fwd)
            results[tk]["actual_vol"].append(actual_vol)

            nw += 1
            t += STEP

        total_windows += nw
        print(f"  {tk}: {nw} walk-forward windows "
              f"({time.perf_counter() - t0:.0f}s)")

    elapsed_wf = time.perf_counter() - t0
    print(f"\n  Total: {total_windows} windows in {elapsed_wf:.0f}s")

    # ════════════════════════════════════════════════════════════════════
    # 3. RESULTS
    # ════════════════════════════════════════════════════════════════════

    # ── 3a. VaR Coverage ────────────────────────────────────────────────
    print(f"\n{'=' * 80}")
    print("  VAR EXCEEDANCE RATES  (nominal vs actual)")
    print(f"{'=' * 80}")
    print("\n  A well-calibrated VaR(α) should be breached exactly α% of the time.")
    print("  Over-coverage (too few breaches) = conservative.  Under-coverage = dangerous.\n")

    hdr = f"  {'':>6}"
    for a in VAR_ALPHAS:
        hdr += f"  VaR{int(a*100):>2}% nom={a:.0%}"
    print(hdr)

    breaches_all = {name: dict.fromkeys(VAR_ALPHAS, 0) for name, _ in MODEL_INFO}
    total_all = 0

    for tk in TICKERS:
        actual = np.array(results[tk]["actual_ret"])
        nw = len(actual)
        total_all += nw

        for name, pfx in MODEL_INFO:
            line = f"  {tk if name == MODEL_INFO[0][0] else '':>6}"
            for a in VAR_ALPHAS:
                qs = [results[tk][f"{pfx}_terminal_quantiles"][i][a] for i in range(nw)]
                breach = float(np.mean(actual < np.array(qs)))
                breaches_all[name][a] += int(np.sum(actual < np.array(qs)))
                line += f"       {breach:>5.1%}      "
            print(f"{line}  ← {name}")
        print()

    # Aggregate
    print(f"  {'AGG':>6}", end="")
    for a in VAR_ALPHAS:
        parts = " ".join(
            f"{MODEL_ABBREV[n]}={breaches_all[n][a] / total_all:.1%}"
            for n, _ in MODEL_INFO
        )
        print(f"  {parts}", end="")
    print(f"   (target: {' / '.join(f'{a:.0%}' for a in VAR_ALPHAS)})")

    # ── 3b. Confidence Interval Coverage ────────────────────────────────
    print(f"\n{'=' * 80}")
    print("  CONFIDENCE INTERVAL COVERAGE")
    print(f"{'=' * 80}")
    print("\n  Nominal coverage vs empirical coverage.  Perfect = diagonal.\n")

    print(f"  {'Ticker':>6}  {'Model':>10}", end="")
    for lv in CI_LEVELS:
        print(f"  {lv:.0%} CI", end="")
    print()

    cov_all = {name: dict.fromkeys(CI_LEVELS, 0) for name, _ in MODEL_INFO}

    for tk in TICKERS:
        actual = np.array(results[tk]["actual_ret"])
        nw = len(actual)

        for name, pfx in MODEL_INFO:
            line = f"  {tk:>6}  {name:>10}"
            for lv in CI_LEVELS:
                covered = 0
                for i in range(nw):
                    lo, hi = results[tk][f"{pfx}_ci_bounds"][i][lv]
                    if lo <= actual[i] <= hi:
                        covered += 1
                rate = covered / nw
                cov_all[name][lv] += covered
                line += f"  {rate:>5.1%}"
            print(line)
        print()

    for idx, (name, _) in enumerate(MODEL_INFO):
        label_left = "AGG" if idx == 0 else ""
        print(f"  {label_left:>6}  {name:>10}", end="")
        for lv in CI_LEVELS:
            print(f"  {cov_all[name][lv] / total_all:>5.1%}", end="")
        print()
    print(f"  {'':>6}  {'Nominal':>10}", end="")
    for lv in CI_LEVELS:
        print(f"  {lv:>5.0%}", end="")
    print()

    # Calibration error
    print("\n  Calibration error (|empirical - nominal|):")
    for name, _ in MODEL_INFO:
        errs = [abs(cov_all[name][lv] / total_all - lv) for lv in CI_LEVELS]
        print(f"    {name:>10}: mean={np.mean(errs):.3f}  max={np.max(errs):.3f}")

    # ── 3c. PIT Uniformity ──────────────────────────────────────────────
    print(f"\n{'=' * 80}")
    print("  PIT HISTOGRAM  (Probability Integral Transform)")
    print(f"{'=' * 80}")
    print("\n  A calibrated forecast produces a uniform PIT distribution.")
    print("  KS test p-value > 0.05 means we cannot reject uniformity.\n")

    n_bins = 10
    for name, pfx in MODEL_INFO:
        all_pit = []
        for tk in TICKERS:
            all_pit.extend(results[tk][f"{pfx}_pit"])
        pit_arr = np.array(all_pit)

        counts, edges = np.histogram(pit_arr, bins=n_bins, range=(0, 1))
        expected = len(pit_arr) / n_bins

        ks_stat, ks_p = sp_stats.kstest(pit_arr, "uniform")

        print(f"  {name}:")
        bar_max = 40
        for i in range(n_bins):
            lo, hi = edges[i], edges[i + 1]
            bar_len = int(counts[i] / max(counts) * bar_max) if max(counts) > 0 else 0
            marker = " *" if abs(counts[i] - expected) > 2 * np.sqrt(expected) else ""
            print(f"    [{lo:.1f}-{hi:.1f}] "
                  f"{'█' * bar_len}{'░' * (bar_max - bar_len)} "
                  f"{counts[i]:>3} (exp={expected:.0f}){marker}")
        print(f"    KS test vs Uniform: stat={ks_stat:.3f}, p={ks_p:.3f}"
              f"  {'✓ UNIFORM' if ks_p > 0.05 else '✗ NOT UNIFORM'}")
        print()

    # ── 3d. CRPS Comparison ─────────────────────────────────────────────
    print(f"{'=' * 80}")
    print("  CRPS COMPARISON  (lower = better)")
    print(f"{'=' * 80}")
    print("\n  CRPS measures the full-distribution forecast accuracy.\n")

    total_crps: dict[str, list] = {name: [] for name, _ in MODEL_INFO}

    print(f"  {'Ticker':>6}", end="")
    for name, _ in MODEL_INFO:
        print(f"  {name:>10}", end="")
    print(f"  {'Winner':>10}")

    for tk in TICKERS:
        vals: dict[str, float] = {}
        for name, pfx in MODEL_INFO:
            v = float(np.mean(results[tk][f"{pfx}_crps"]))
            vals[name] = v
            total_crps[name].extend(results[tk][f"{pfx}_crps"])
        winner = min(vals, key=vals.get)
        line = f"  {tk:>6}"
        for name, _ in MODEL_INFO:
            line += f"  {vals[name]:>10.5f}"
        print(f"{line}  {winner:>10}")

    agg_vals = {name: float(np.mean(total_crps[name])) for name, _ in MODEL_INFO}
    agg_winner = min(agg_vals, key=agg_vals.get)
    line = f"  {'AGG':>6}"
    for name, _ in MODEL_INFO:
        line += f"  {agg_vals[name]:>10.5f}"
    print(f"{line}  {agg_winner:>10}")

    for name, _ in MODEL_INFO:
        if name == agg_winner:
            continue
        pct = (agg_vals[name] / agg_vals[agg_winner] - 1) * 100
        print(f"\n  {name} CRPS is {pct:.1f}% worse than {agg_winner}")

    # ── 3e. Volatility Calibration ──────────────────────────────────────
    print(f"\n{'=' * 80}")
    print("  VOLATILITY CALIBRATION  (forecast vol vs realised vol)")
    print(f"{'=' * 80}")

    from scipy.stats import spearmanr

    # Student-t and Gaussian use the same trailing vol.
    # GARCH-t has its own conditional vol estimate.
    vol_models = [("Ensemble", "ens"), ("GARCH-t", "gt"), ("GJR-skewt", "gjr"),
                   ("GJR-VIX", "gvix"), ("GJR-VIX+SKEW", "gvs"),
                   ("Gaussian", "gauss")]

    print(f"\n  {'Ticker':>6}  {'Model':>10}  {'Spearman ρ':>12}  {'MAE':>8}  {'Bias':>8}")
    for tk in TICKERS:
        av = np.array(results[tk]["actual_vol"])
        for model, pfx in vol_models:
            fv = np.array(results[tk][f"{pfx}_fvol"])
            rho, _ = spearmanr(fv, av)
            mae = float(np.mean(np.abs(fv - av)))
            bias = float(np.mean(fv - av))
            print(f"  {tk:>6}  {model:>10}  {rho:>+12.3f}  {mae:>8.4f}  {bias:>+8.4f}")
        print()

    # Aggregate
    all_fv: dict[str, list] = {name: [] for name, _ in vol_models}
    all_av: list = []
    for tk in TICKERS:
        all_av.extend(results[tk]["actual_vol"])
        for name, pfx in vol_models:
            all_fv[name].extend(results[tk][f"{pfx}_fvol"])
    all_av_arr = np.array(all_av)

    print("  Aggregate vol rank correlation:")
    for name, _ in vol_models:
        rho, _ = spearmanr(np.array(all_fv[name]), all_av_arr)
        print(f"    {name:>10}:  ρ = {rho:+.3f}")
    print("\n  (Student-t vol is identical to Gaussian — same trailing estimate,")
    print("   differs only in tail shape.  GARCH-t and GJR use conditional vol.")
    print("   GJR-VIX blends GARCH conditional vol 65/35 with VIX-implied vol.")
    print("   GJR-VIX+SKEW has same vol but adjusts skewness via CBOE SKEW index.)")

    # ── 3f. Conditional Calibration ─────────────────────────────────────
    print(f"\n{'=' * 80}")
    print("  CONDITIONAL CALIBRATION  (does calibration hold in all regimes?)")
    print(f"{'=' * 80}")
    print("\n  Splitting windows by realised vol tertile: "
          "calm / normal / stressed\n")

    all_actual_vol = np.array(all_av)
    q33 = np.quantile(all_actual_vol, 1 / 3)
    q67 = np.quantile(all_actual_vol, 2 / 3)

    for regime_name, lo_v, hi_v in [
        ("Calm", 0.0, q33),
        ("Normal", q33, q67),
        ("Stressed", q67, 999.0),
    ]:
        mask_indices = []
        for tk in TICKERS:
            for i, av in enumerate(results[tk]["actual_vol"]):
                if lo_v <= av < hi_v:
                    mask_indices.append((tk, i))

        if not mask_indices:
            continue

        print(f"  {regime_name} ({len(mask_indices)} windows):")

        for name, pfx in MODEL_INFO:
            line = f"    {name:>10}:"
            for lv in CI_LEVELS:
                covered = 0
                for tk_i, i in mask_indices:
                    lo, hi = results[tk_i][f"{pfx}_ci_bounds"][i][lv]
                    if lo <= results[tk_i]["actual_ret"][i] <= hi:
                        covered += 1
                rate = covered / len(mask_indices)
                err = rate - lv
                line += f"  {lv:.0%}CI={rate:.1%}({err:+.1%})"
            print(line)
        print()

    # ── 3g. Tail Behaviour ──────────────────────────────────────────────
    print(f"{'=' * 80}")
    print("  TAIL BEHAVIOUR  (worst-case accuracy)")
    print(f"{'=' * 80}")
    print("\n  How accurate are the tails when actual returns are in the worst 10%?\n")

    all_actual_ret: list = []
    all_q5: dict[str, list] = {name: [] for name, _ in MODEL_INFO}
    for tk in TICKERS:
        for i in range(len(results[tk]["actual_ret"])):
            all_actual_ret.append(results[tk]["actual_ret"][i])
            for name, pfx in MODEL_INFO:
                all_q5[name].append(results[tk][f"{pfx}_terminal_quantiles"][i][0.05])

    all_actual_ret_arr = np.array(all_actual_ret)
    for name in all_q5:
        all_q5[name] = np.array(all_q5[name])

    thresh = np.quantile(all_actual_ret_arr, 0.10)
    bad_mask = all_actual_ret_arr <= thresh
    n_bad = int(bad_mask.sum())

    print(f"  Worst 10% of outcomes: {n_bad} windows (actual ret ≤ {thresh:.3f})")
    print(f"  Mean actual return in these windows: {all_actual_ret_arr[bad_mask].mean():.3f}")
    print()
    print("  In these bad windows, did the VaR5% capture the loss?")
    for name, _ in MODEL_INFO:
        warn = float(np.mean(all_actual_ret_arr[bad_mask] < all_q5[name][bad_mask]))
        print(f"    {name:>10}:  {1 - warn:.1%} of worst outcomes were within VaR5%")

    print("\n  Average loss the model expected (VaR5%) vs actual loss:")
    for name, _ in MODEL_INFO:
        print(f"    {name:>10} expected:  {all_q5[name][bad_mask].mean():+.3f}")
    print(f"    {'Actual':>10}:             {all_actual_ret_arr[bad_mask].mean():+.3f}")

    # ── Interpretation ──────────────────────────────────────────────────
    print(f"\n{'=' * 80}")
    print("  INTERPRETATION")
    print(f"{'=' * 80}")
    print("""
  Seven models compete:

    1. Gaussian:      trailing vol, normal distribution.
    2. Student-t:     trailing vol, MLE-fitted df.
    3. GARCH-t:       GARCH(1,1) conditional vol + Student-t.
    4. GJR-skewt:     GJR-GARCH(1,1) + Hansen's skewed-t.
    5. GJR-VIX:       GJR-skewt + 65/35 blend of GARCH vol
                      with VIX-implied vol.  VIX is forward-looking
                      and reacts instantly to stress; GARCH lags.
                      For non-SPY assets, VIX is scaled by the
                      asset/SPY trailing-vol ratio.
    6. GJR-VIX+SKEW:  Same vol as GJR-VIX, but CBOE SKEW index
                      adjusts the skewness parameter (lambda).
                      Higher SKEW → heavier left tail forecast.
    7. Ensemble:      HMM + Heston + Merton JD + RS-GBM Monte Carlo.

  Key metrics:
    - CRPS: comprehensive distribution accuracy (lower = better)
    - CI coverage: do intervals hit their nominal rates?
    - PIT uniformity: is the whole distribution calibrated?
    - VaR coverage: are tail risk estimates accurate?
    - Conditional calibration: does it hold in calm AND crisis?
""")

    elapsed = time.perf_counter() - t0
    print(f"  Total runtime: {elapsed:.0f}s")
    print(f"  Output saved to {_OUT}")
    print("=" * 80)

    _fh.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
