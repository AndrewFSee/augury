"""Example 26 — External Data Experiment
==========================================

Systematically tests five external data sources to improve GJR-GARCH
probabilistic forecasts.  Each source targets a specific weakness:

  1. **VRP weighting**:  Dynamic VIX blend weight based on the variance
     risk premium.  Reduces calm over-coverage by shrinking VIX weight
     when VIX persistently overstates future vol.
  2. **Asset-specific implied vol**:  MOVE index for TLT, GVZ for GLD
     instead of cross-asset VIX scaling.
  3. **SKEW index**:  Adjusts the skewness parameter (lambda) of the
     skewed-t distribution using CBOE SKEW as a tail-risk signal.
  4. **Credit spreads**:  HYG/LQD ratio change as a vol multiplier.
     Widens before equity stress — a leading indicator.
  5. **Full combined**:  VRP + asset-specific IV + SKEW + credit, all
     together.

Eight models compared:
  1. GJR-skewt:   Pure GJR-GARCH + skewed-t (baseline, no external data)
  2. GJR-VIX:     Fixed 50/50 blend of GARCH vol with VIX implied vol
  3. GJR-VRP:     Dynamic VIX weight via variance risk premium
  4. GJR-ImpVol:  MOVE for TLT, GVZ for GLD, VIX for equities
  5. GJR-SKEW:    SKEW index adjusts skewness parameter
  6. GJR-Credit:  Credit spread (HYG/LQD) vol multiplier
  7. GJR-Full:    VRP + ImpVol + SKEW + Credit combined
  8. Gaussian:    Reference baseline

Same walk-forward framework as Example 25.

Note: this predates ``augury.evaluation.walk_forward`` and example 27 tunes
its constants on the same data it scores. For a leak-free test of implied-vol
blending see example 32 and ``augury.forecasters.ImpliedVolForecaster``.
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
TICKERS   = ["SPY", "QQQ", "IWM", "TLT", "GLD"]
START     = "2010-01-01"
END       = "2026-12-31"
LOOKBACK  = 504
HORIZON   = 21
STEP      = 21
EMBARGO   = 21
N_T_SAMPLES = 10_000

# Tuning constants for external signals
TLT_DURATION = 17.0    # approx modified duration for 20+ yr treasuries
VRP_SCALE    = 2.5     # VRP → VIX weight sensitivity
SKEW_SCALE   = 300.0   # SKEW index → lambda adjustment denominator
CREDIT_SCALE = 2.0     # credit spread → vol multiplier scaling

from arch.univariate.distribution import SkewStudent as _ArchSkewT  # noqa: E402

from augury.data.ingest import DataIngestor  # noqa: E402
from augury.models.garch import GARCHModel  # noqa: E402

# ── Model registry (name, storage prefix) ───────────────────────────────
MODEL_INFO = [
    ("GJR-skewt",   "gjr"),
    ("GJR-VIX",     "gvix"),
    ("GJR-VRP",     "gvrp"),
    ("GJR-ImpVol",  "giv"),
    ("GJR-SKEW",    "gskw"),
    ("GJR-Credit",  "gcr"),
    ("GJR-Full",    "gful"),
    ("Gaussian",    "gauss"),
]
MODEL_ABBREV = {
    "GJR-skewt": "GJR", "GJR-VIX": "VIX", "GJR-VRP": "VRP",
    "GJR-ImpVol": "IV", "GJR-SKEW": "SKW", "GJR-Credit": "CR",
    "GJR-Full": "FULL", "Gaussian": "G",
}

_skt = _ArchSkewT()


# ── Helpers ─────────────────────────────────────────────────────────────
def _skewt_metrics(mu, sig, params, actual, rng, var_alphas, ci_levels):
    """Compute quantiles, CI bounds, PIT, and CRPS for a skewed-t model."""
    q = {}
    for a in var_alphas:
        q[a] = float(mu + sig * _skt.ppf(a, parameters=params))
    ci = {}
    for lv in ci_levels:
        lo = float(mu + sig * _skt.ppf((1 - lv) / 2, parameters=params))
        hi = float(mu + sig * _skt.ppf(1 - (1 - lv) / 2, parameters=params))
        ci[lv] = (lo, hi)
    z = (actual - mu) / sig if sig > 0 else 0.0
    pit = float(_skt.cdf(z, parameters=params))
    # CRPS via inverse-transform sampling
    u = rng.uniform(0, 1, N_T_SAMPLES)
    samp = np.sort(mu + sig * _skt.ppf(u, parameters=params))
    m = len(samp)
    mae = float(np.mean(np.abs(samp - actual)))
    spread = float(2 * np.sum(samp * np.arange(m)) / m**2 - np.mean(samp))
    return q, ci, pit, mae - spread


def _store(results, tk, pfx, q, ci, pit, crps, fvol):
    """Append one window's results for a given model."""
    results[tk][f"{pfx}_terminal_quantiles"].append(q)
    results[tk][f"{pfx}_ci_bounds"].append(ci)
    results[tk][f"{pfx}_pit"].append(pit)
    results[tk][f"{pfx}_crps"].append(crps)
    results[tk][f"{pfx}_fvol"].append(fvol)


# ════════════════════════════════════════════════════════════════════════
def main() -> int:  # noqa: C901, PLR0912, PLR0915
    t0 = time.perf_counter()
    rng = np.random.default_rng(42)

    print("=" * 80)
    print("  EXTERNAL DATA EXPERIMENT")
    print(f"  {', '.join(TICKERS)}  |  {START} → {END}")
    print(f"  Lookback={LOOKBACK}d  Horizon={HORIZON}d  Embargo={EMBARGO}d")
    print(f"  Walk-forward step={STEP}d")
    print(f"  VRP_SCALE={VRP_SCALE}  SKEW_SCALE={SKEW_SCALE}  "
          f"CREDIT_SCALE={CREDIT_SCALE}  TLT_DURATION={TLT_DURATION}")
    print("=" * 80)

    # ── 1. Data ─────────────────────────────────────────────────────────
    ingestor = DataIngestor()
    data: dict[str, dict] = {}
    for tk in TICKERS:
        df = ingestor.fetch(tk, start=START, end=END)
        close = df["close"].to_numpy(dtype=float)
        dates = df.index
        rets = np.diff(np.log(close))
        data[tk] = {"close": close, "returns": rets, "dates": dates}
        print(f"  {tk}: {len(rets)} daily returns")

    # ── External data sources ───────────────────────────────────────────
    # VIX (always available)
    vix_df = ingestor.fetch("^VIX", start=START, end=END)
    vix_close = vix_df["close"]
    print(f"  VIX: {len(vix_close)} observations")

    # SPY rolling 63-day vol (for cross-asset VIX scaling)
    _spy_rets_s = pd.Series(
        data["SPY"]["returns"],
        index=data["SPY"]["dates"][1:],
    )
    spy_rolling_vol = _spy_rets_s.rolling(63, min_periods=21).std() * np.sqrt(252)

    # MOVE Index (implied vol for Treasury bonds)
    try:
        move_df = ingestor.fetch("^MOVE", start=START, end=END)
        move_close = move_df["close"]
        print(f"  MOVE: {len(move_close)} observations")
    except Exception:
        move_close = None
        print("  MOVE: not available — TLT will use VIX scaling")

    # GVZ (CBOE Gold ETF Volatility Index)
    try:
        gvz_df = ingestor.fetch("^GVZ", start=START, end=END)
        gvz_close = gvz_df["close"]
        print(f"  GVZ: {len(gvz_close)} observations")
    except Exception:
        gvz_close = None
        print("  GVZ: not available — GLD will use VIX scaling")

    # SKEW Index (CBOE tail risk)
    try:
        skew_df = ingestor.fetch("^SKEW", start=START, end=END)
        skew_close = skew_df["close"]
        print(f"  SKEW: {len(skew_close)} observations")
    except Exception:
        skew_close = None
        print("  SKEW: not available — skewness adjustment disabled")

    # Credit spread signal: HYG/LQD ratio
    try:
        hyg_df = ingestor.fetch("HYG", start=START, end=END)
        lqd_df = ingestor.fetch("LQD", start=START, end=END)
        _cred = pd.DataFrame({
            "hyg": np.log(hyg_df["close"]),
            "lqd": np.log(lqd_df["close"]),
        }).ffill().dropna()
        credit_ratio = _cred["hyg"] - _cred["lqd"]
        credit_change_63 = credit_ratio - credit_ratio.shift(63)
        print(f"  Credit (HYG/LQD): {len(_cred)} observations")
    except Exception:
        credit_change_63 = None
        print("  Credit (HYG/LQD): not available — credit adjustment disabled")

    # ── 2. Walk-forward calibration test ────────────────────────────────
    VAR_ALPHAS = [0.01, 0.05, 0.10, 0.25]
    CI_LEVELS  = [0.50, 0.80, 0.90, 0.95]

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

            # ── Trailing statistics ─────────────────────────────────
            trailing_mu = float(np.mean(train)) * HORIZON
            trailing_sig = float(np.std(train)) * np.sqrt(HORIZON)
            trailing_vol = float(np.std(train[-63:]) * np.sqrt(252))

            # ── Gaussian baseline ───────────────────────────────────
            gauss_q = {}
            for a in VAR_ALPHAS:
                gauss_q[a] = float(sp_stats.norm.ppf(a, trailing_mu, trailing_sig))
            gauss_ci = {}
            for lv in CI_LEVELS:
                lo = float(sp_stats.norm.ppf((1 - lv) / 2, trailing_mu, trailing_sig))
                hi = float(sp_stats.norm.ppf(1 - (1 - lv) / 2, trailing_mu, trailing_sig))
                gauss_ci[lv] = (lo, hi)
            gauss_pit = float(sp_stats.norm.cdf(actual_fwd, trailing_mu, trailing_sig))
            z = (actual_fwd - trailing_mu) / trailing_sig if trailing_sig > 0 else 0.0
            gauss_crps = float(
                trailing_sig * (z * (2 * sp_stats.norm.cdf(z) - 1)
                                + 2 * sp_stats.norm.pdf(z) - 1 / np.sqrt(np.pi))
            )
            _store(results, tk, "gauss", gauss_q, gauss_ci, gauss_pit,
                   gauss_crps, trailing_vol)

            # ── Fit GJR-GARCH (shared across all GJR variants) ─────
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

            base_params = np.array([gjr_eta, gjr_lam])

            # ── 1. GJR-skewt (pure GARCH, no external data) ────────
            q, ci, pit, crps = _skewt_metrics(
                trailing_mu, gjr_sig, base_params, actual_fwd, rng,
                VAR_ALPHAS, CI_LEVELS)
            _store(results, tk, "gjr", q, ci, pit, crps, gjr_fvol)

            # ── External data signals for this window ───────────────
            current_date = data[tk]["dates"][t]

            # VIX implied vol (same as Example 25)
            vix_level = float(vix_close.asof(current_date))
            if np.isnan(vix_level) or vix_level <= 0:
                vix_level = 20.0
            vix_ann = vix_level / 100.0

            if tk == "SPY":
                vix_implied = vix_ann
            else:
                spy_tv = float(spy_rolling_vol.asof(current_date))
                if np.isnan(spy_tv) or spy_tv <= 0:
                    spy_tv = trailing_vol
                vix_implied = (
                    vix_ann * (trailing_vol / spy_tv) if spy_tv > 0 else vix_ann
                )

            # Asset-specific implied vol (MOVE for TLT, GVZ for GLD)
            if tk == "TLT" and move_close is not None:
                move_lvl = float(move_close.asof(current_date))
                if not np.isnan(move_lvl) and move_lvl > 0:
                    # MOVE is in bps of yield vol; price vol ≈ duration × yield vol
                    asset_implied = TLT_DURATION * move_lvl / 10000.0
                else:
                    asset_implied = vix_implied
            elif tk == "GLD" and gvz_close is not None:
                gvz_lvl = float(gvz_close.asof(current_date))
                if not np.isnan(gvz_lvl) and gvz_lvl > 0:
                    # GVZ is in % annualised vol, same scale as VIX
                    asset_implied = gvz_lvl / 100.0
                else:
                    asset_implied = vix_implied
            else:
                asset_implied = vix_implied

            # SKEW → skewness parameter adjustment
            if skew_close is not None:
                skew_lvl = float(skew_close.asof(current_date))
                if not np.isnan(skew_lvl) and skew_lvl > 0:
                    # SKEW=100 → symmetric; SKEW>100 → heavier left tail
                    skew_adj = -(skew_lvl - 100) / SKEW_SCALE
                else:
                    skew_adj = 0.0
            else:
                skew_adj = 0.0
            skew_lam = float(np.clip(gjr_lam + skew_adj, -0.9, 0.9))
            skew_params = np.array([gjr_eta, skew_lam])

            # Credit spread → vol multiplier
            if credit_change_63 is not None:
                cc = credit_change_63.asof(current_date)
                cc = float(cc) if not np.isnan(cc) else 0.0
                # Negative cc = HY underperforming IG = widening = stress
                credit_mult = min(2.0, max(1.0, 1.0 - CREDIT_SCALE * cc))
            else:
                credit_mult = 1.0

            # Variance risk premium
            vrp = vix_implied - trailing_vol

            # ── 2. GJR-VIX (fixed 50/50 blend) ─────────────────────
            vix_vol = 0.5 * gjr_fvol + 0.5 * vix_implied
            vix_sig = vix_vol / np.sqrt(252) * np.sqrt(HORIZON)
            q, ci, pit, crps = _skewt_metrics(
                trailing_mu, vix_sig, base_params, actual_fwd, rng,
                VAR_ALPHAS, CI_LEVELS)
            _store(results, tk, "gvix", q, ci, pit, crps, vix_vol)

            # ── 3. GJR-VRP (dynamic VIX weight via VRP) ────────────
            # High VRP (calm) → VIX overstates → lower VIX weight
            # Low/neg VRP (crisis) → trust VIX more → higher weight
            vix_w = float(np.clip(0.5 - VRP_SCALE * vrp, 0.15, 0.85))
            vrp_vol = (1 - vix_w) * gjr_fvol + vix_w * vix_implied
            vrp_sig = vrp_vol / np.sqrt(252) * np.sqrt(HORIZON)
            q, ci, pit, crps = _skewt_metrics(
                trailing_mu, vrp_sig, base_params, actual_fwd, rng,
                VAR_ALPHAS, CI_LEVELS)
            _store(results, tk, "gvrp", q, ci, pit, crps, vrp_vol)

            # ── 4. GJR-ImpVol (asset-specific implied vol) ─────────
            iv_vol = 0.5 * gjr_fvol + 0.5 * asset_implied
            iv_sig = iv_vol / np.sqrt(252) * np.sqrt(HORIZON)
            q, ci, pit, crps = _skewt_metrics(
                trailing_mu, iv_sig, base_params, actual_fwd, rng,
                VAR_ALPHAS, CI_LEVELS)
            _store(results, tk, "giv", q, ci, pit, crps, iv_vol)

            # ── 5. GJR-SKEW (SKEW-adjusted skewness) ───────────────
            # Uses GARCH vol (same as GJR-skewt) but SKEW-adjusted lambda
            q, ci, pit, crps = _skewt_metrics(
                trailing_mu, gjr_sig, skew_params, actual_fwd, rng,
                VAR_ALPHAS, CI_LEVELS)
            _store(results, tk, "gskw", q, ci, pit, crps, gjr_fvol)

            # ── 6. GJR-Credit (credit spread vol adj) ──────────────
            cr_vol = gjr_fvol * credit_mult
            cr_sig = cr_vol / np.sqrt(252) * np.sqrt(HORIZON)
            q, ci, pit, crps = _skewt_metrics(
                trailing_mu, cr_sig, base_params, actual_fwd, rng,
                VAR_ALPHAS, CI_LEVELS)
            _store(results, tk, "gcr", q, ci, pit, crps, cr_vol)

            # ── 7. GJR-Full (VRP + ImpVol + SKEW + Credit) ─────────
            full_vrp = asset_implied - trailing_vol
            full_vix_w = float(np.clip(0.5 - VRP_SCALE * full_vrp, 0.15, 0.85))
            full_vol = (
                (1 - full_vix_w) * gjr_fvol + full_vix_w * asset_implied
            ) * credit_mult
            full_sig = full_vol / np.sqrt(252) * np.sqrt(HORIZON)
            q, ci, pit, crps = _skewt_metrics(
                trailing_mu, full_sig, skew_params, actual_fwd, rng,
                VAR_ALPHAS, CI_LEVELS)
            _store(results, tk, "gful", q, ci, pit, crps, full_vol)

            # ── Store actuals ───────────────────────────────────────
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

    # All models except GJR-SKEW (same vol as GJR-skewt)
    vol_models = [
        ("GJR-skewt", "gjr"), ("GJR-VIX", "gvix"), ("GJR-VRP", "gvrp"),
        ("GJR-ImpVol", "giv"), ("GJR-Credit", "gcr"), ("GJR-Full", "gful"),
        ("Gaussian", "gauss"),
    ]

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
    print("\n  (GJR-SKEW has identical vol to GJR-skewt — differs only in tail shape.)")

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

    # ── 3h. External Data Impact Summary ────────────────────────────────
    print(f"\n{'=' * 80}")
    print("  EXTERNAL DATA IMPACT SUMMARY")
    print(f"{'=' * 80}")
    print("\n  Comparing each enhancement against the relevant baseline.\n")

    # CRPS improvement vs baselines
    gjr_crps = agg_vals["GJR-skewt"]
    vix_crps = agg_vals["GJR-VIX"]

    comparisons = [
        ("VIX blend (50/50)",      "GJR-VIX",     gjr_crps),
        ("VRP weighting",          "GJR-VRP",     vix_crps),
        ("Asset-specific IV",     "GJR-ImpVol",  vix_crps),
        ("SKEW adjustment",        "GJR-SKEW",    gjr_crps),
        ("Credit spread",          "GJR-Credit",  gjr_crps),
        ("Full combined",          "GJR-Full",    gjr_crps),
    ]

    print(f"  {'Enhancement':>20}  {'Model':>10}  {'CRPS':>10}  {'vs baseline':>12}  {'Verdict':>10}")
    for label, model, baseline in comparisons:
        model_crps = agg_vals[model]
        diff_pct = (model_crps / baseline - 1) * 100
        verdict = "✓ HELPS" if diff_pct < -0.1 else ("≈ NEUTRAL" if abs(diff_pct) < 0.5 else "✗ HURTS")
        print(f"  {label:>20}  {model:>10}  {model_crps:>10.5f}  {diff_pct:>+11.1f}%  {verdict:>10}")

    # Stressed-regime improvement
    print("\n  Stressed-regime 50% CI coverage (target: 50%):")
    stressed_indices = []
    for tk in TICKERS:
        for i, av in enumerate(results[tk]["actual_vol"]):
            if av >= q67:
                stressed_indices.append((tk, i))
    if stressed_indices:
        for name, pfx in MODEL_INFO:
            covered = 0
            for tk_i, i in stressed_indices:
                lo, hi = results[tk_i][f"{pfx}_ci_bounds"][i][0.50]
                if lo <= results[tk_i]["actual_ret"][i] <= hi:
                    covered += 1
            rate = covered / len(stressed_indices)
            err = rate - 0.50
            print(f"    {name:>10}: {rate:.1%} ({err:+.1%})")

    # ── Interpretation ──────────────────────────────────────────────────
    print(f"\n{'=' * 80}")
    print("  INTERPRETATION")
    print(f"{'=' * 80}")
    print("""
  Eight models tested — all share the same GJR-GARCH(1,1) + skewed-t
  fit.  Only the vol scale and/or distribution shape differ:

    1. GJR-skewt:   Pure GARCH vol (baseline).
    2. GJR-VIX:     50/50 blend of GARCH vol with VIX implied vol.
    3. GJR-VRP:     Dynamic blend weight based on variance risk premium.
                    Less VIX in calm (VIX overstates), more in crisis.
    4. GJR-ImpVol:  MOVE index for TLT, GVZ for GLD, VIX for equities.
                    Asset-specific implied vol instead of cross-VIX proxy.
    5. GJR-SKEW:    CBOE SKEW index adjusts the skewness parameter.
                    Higher SKEW → heavier left tail in the forecast.
    6. GJR-Credit:  HYG/LQD credit spread change as vol multiplier.
                    Widening spreads → higher forecast vol.
    7. GJR-Full:    VRP + ImpVol + SKEW + Credit, all combined.
    8. Gaussian:    Trailing vol, normal distribution (reference).

  Key questions answered:
    - Does VRP-dynamic weighting reduce calm over-coverage?
    - Do MOVE/GVZ give better implied vol for bonds/gold?
    - Does SKEW improve tail forecasts?
    - Does credit spread add value beyond VIX?
    - Does combining everything help or overfit?
""")

    elapsed = time.perf_counter() - t0
    print(f"  Total runtime: {elapsed:.0f}s")
    print(f"  Output saved to {_OUT}")
    print("=" * 80)

    _fh.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
