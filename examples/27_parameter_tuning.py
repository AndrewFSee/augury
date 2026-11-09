"""Example 27 — Parameter Tuning for External Data Models
==========================================================

Grid-searches key parameters for each external data enhancement,
using the same walk-forward framework as Example 26 but evaluating
only CRPS and stressed-regime 50%CI coverage.

Sweeps run independently for each enhancement so the search space
stays manageable.  The script reports best parameters per model.
"""

from __future__ import annotations

import sys
import time
import warnings
from itertools import product

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Output capture ──────────────────────────────────────────────────────
_OUT = "_tuning_output.txt"
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

from arch.univariate.distribution import SkewStudent as _ArchSkewT  # noqa: E402

from augury.data.ingest import DataIngestor  # noqa: E402
from augury.models.garch import GARCHModel  # noqa: E402

_skt = _ArchSkewT()


# ── Helpers ─────────────────────────────────────────────────────────────
def _skewt_crps(mu, sig, params, actual, rng):
    """CRPS for skewed-t forecast."""
    u = rng.uniform(0, 1, N_T_SAMPLES)
    samp = np.sort(mu + sig * _skt.ppf(u, parameters=params))
    m = len(samp)
    mae = float(np.mean(np.abs(samp - actual)))
    spread = float(2 * np.sum(samp * np.arange(m)) / m**2 - np.mean(samp))
    return mae - spread


def _skewt_ci50(mu, sig, params, actual):
    """Check if actual falls within 50% CI."""
    lo = float(mu + sig * _skt.ppf(0.25, parameters=params))
    hi = float(mu + sig * _skt.ppf(0.75, parameters=params))
    return lo <= actual <= hi


# ════════════════════════════════════════════════════════════════════════
def main() -> int:  # noqa: C901, PLR0912, PLR0915
    t0 = time.perf_counter()
    rng = np.random.default_rng(42)

    print("=" * 80)
    print("  PARAMETER TUNING SWEEP")
    print(f"  {', '.join(TICKERS)}  |  {START} → {END}")
    print("=" * 80)

    # ── 1. Load all data ────────────────────────────────────────────────
    ingestor = DataIngestor()
    data: dict[str, dict] = {}
    for tk in TICKERS:
        df = ingestor.fetch(tk, start=START, end=END)
        close = df["close"].to_numpy(dtype=float)
        dates = df.index
        rets = np.diff(np.log(close))
        data[tk] = {"close": close, "returns": rets, "dates": dates}
        print(f"  {tk}: {len(rets)} returns")

    # External data
    vix_df = ingestor.fetch("^VIX", start=START, end=END)
    vix_close = vix_df["close"]

    _spy_rets_s = pd.Series(data["SPY"]["returns"], index=data["SPY"]["dates"][1:])
    spy_rolling_vol = _spy_rets_s.rolling(63, min_periods=21).std() * np.sqrt(252)

    try:
        move_close = ingestor.fetch("^MOVE", start=START, end=END)["close"]
    except Exception:
        move_close = None

    try:
        gvz_close = ingestor.fetch("^GVZ", start=START, end=END)["close"]
    except Exception:
        gvz_close = None

    try:
        skew_close = ingestor.fetch("^SKEW", start=START, end=END)["close"]
    except Exception:
        skew_close = None

    try:
        hyg_df = ingestor.fetch("HYG", start=START, end=END)
        lqd_df = ingestor.fetch("LQD", start=START, end=END)
        _cred = pd.DataFrame({
            "hyg": np.log(hyg_df["close"]),
            "lqd": np.log(lqd_df["close"]),
        }).ffill().dropna()
        credit_ratio = _cred["hyg"] - _cred["lqd"]
        credit_change_63 = credit_ratio - credit_ratio.shift(63)
    except Exception:
        credit_change_63 = None

    print("  Data loaded.\n")

    # ── 2. Pre-compute all per-window GARCH fits + signals ──────────────
    # This avoids re-fitting GARCH for every parameter combo.
    print("  Pre-computing GARCH fits for all windows...")

    WindowData = dict  # type alias for clarity

    all_windows: list[WindowData] = []

    for tk in TICKERS:
        rets = data[tk]["returns"]
        close = data[tk]["close"]
        n = len(rets)

        t = LOOKBACK
        while t + EMBARGO + HORIZON <= n:
            train = rets[t - LOOKBACK : t]
            actual_fwd = float(np.sum(rets[t + EMBARGO : t + EMBARGO + HORIZON]))
            actual_vol = float(
                np.std(rets[t + EMBARGO : t + EMBARGO + HORIZON]) * np.sqrt(252)
            )

            trailing_mu = float(np.mean(train)) * HORIZON
            trailing_sig = float(np.std(train)) * np.sqrt(HORIZON)
            trailing_vol = float(np.std(train[-63:]) * np.sqrt(252))

            # GJR-GARCH fit
            try:
                gjr_m = GARCHModel(vol_model="GJRGARCH", p=1, q=1, dist="skewt")
                gjr_m.fit(train, scale=100.0)
                gjr_vol_1 = float(gjr_m.forecast(horizon=1)[0])
                gjr_sig = gjr_vol_1 / np.sqrt(252) * np.sqrt(HORIZON)
                gjr_eta = max(float(gjr_m._result.params.get("eta", 5.0)), 2.1)
                gjr_lam = float(gjr_m._result.params.get("lambda", 0.0))
                gjr_fvol = gjr_vol_1
            except Exception:
                gjr_sig = trailing_sig
                gjr_eta = 5.0
                gjr_lam = 0.0
                gjr_fvol = trailing_vol

            # External signals
            current_date = data[tk]["dates"][t]

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
                vix_implied = vix_ann * (trailing_vol / spy_tv) if spy_tv > 0 else vix_ann

            # Asset-specific implied vol
            if tk == "TLT" and move_close is not None:
                move_lvl = float(move_close.asof(current_date))
                move_valid = not np.isnan(move_lvl) and move_lvl > 0
            else:
                move_lvl = 0.0
                move_valid = False

            if tk == "GLD" and gvz_close is not None:
                gvz_lvl = float(gvz_close.asof(current_date))
                gvz_valid = not np.isnan(gvz_lvl) and gvz_lvl > 0
            else:
                gvz_lvl = 0.0
                gvz_valid = False

            # SKEW
            if skew_close is not None:
                skew_lvl = float(skew_close.asof(current_date))
                skew_valid = not np.isnan(skew_lvl) and skew_lvl > 0
            else:
                skew_lvl = 100.0
                skew_valid = False

            # Credit
            if credit_change_63 is not None:
                cc = credit_change_63.asof(current_date)
                cc = float(cc) if not np.isnan(cc) else 0.0
            else:
                cc = 0.0

            all_windows.append({
                "tk": tk,
                "actual_fwd": actual_fwd,
                "actual_vol": actual_vol,
                "trailing_mu": trailing_mu,
                "trailing_sig": trailing_sig,
                "trailing_vol": trailing_vol,
                "gjr_sig": gjr_sig,
                "gjr_fvol": gjr_fvol,
                "gjr_eta": gjr_eta,
                "gjr_lam": gjr_lam,
                "vix_implied": vix_implied,
                "move_lvl": move_lvl,
                "move_valid": move_valid,
                "gvz_lvl": gvz_lvl,
                "gvz_valid": gvz_valid,
                "skew_lvl": skew_lvl,
                "skew_valid": skew_valid,
                "cc": cc,
            })
            t += STEP

    n_windows = len(all_windows)
    print(f"  {n_windows} windows pre-computed ({time.perf_counter() - t0:.0f}s)\n")

    # Split into vol tertiles for conditional eval
    all_vols = np.array([w["actual_vol"] for w in all_windows])
    vol_33 = np.percentile(all_vols, 33.3)
    vol_67 = np.percentile(all_vols, 66.7)
    for w in all_windows:
        if w["actual_vol"] <= vol_33:
            w["regime"] = "calm"
        elif w["actual_vol"] <= vol_67:
            w["regime"] = "normal"
        else:
            w["regime"] = "stressed"

    # ── Helper to evaluate a model config across all windows ────────────
    def eval_model(vol_fn, params_fn=None):
        """Evaluate a model.
        vol_fn(w) -> annualised vol
        params_fn(w) -> np.array([eta, lam]) or None for base params
        Returns (avg_crps, stressed_50ci_coverage, calm_50ci_overage).
        """
        crps_list = []
        stressed_hits = 0
        stressed_total = 0
        calm_hits = 0
        calm_total = 0

        for w in all_windows:
            vol = vol_fn(w)
            sig = vol / np.sqrt(252) * np.sqrt(HORIZON)
            params = params_fn(w) if params_fn else np.array([w["gjr_eta"], w["gjr_lam"]])
            mu = w["trailing_mu"]
            actual = w["actual_fwd"]

            crps = _skewt_crps(mu, sig, params, actual, rng)
            crps_list.append(crps)

            hit = _skewt_ci50(mu, sig, params, actual)
            if w["regime"] == "stressed":
                stressed_hits += hit
                stressed_total += 1
            elif w["regime"] == "calm":
                calm_hits += hit
                calm_total += 1

        avg_crps = float(np.mean(crps_list))
        stressed_cov = stressed_hits / stressed_total if stressed_total > 0 else 0.5
        calm_cov = calm_hits / calm_total if calm_total > 0 else 0.5
        return avg_crps, stressed_cov, calm_cov

    # ── 3. Sweep: VIX blend weight ─────────────────────────────────────
    print("=" * 80)
    print("  SWEEP 1: VIX Blend Weight (GJR-VIX)")
    print("  Testing fixed blend: vol = w*GARCH + (1-w)*VIX_implied")
    print("=" * 80)

    vix_weights = [0.20, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.80]
    print(f"\n  {'GARCH_w':>8s}  {'CRPS':>8s}  {'Stressed50':>11s}  {'Calm50':>8s}")
    best_vix = {"crps": 1.0, "w": 0.5}
    for gw in vix_weights:
        def _vfn(w, _gw=gw):
            return _gw * w["gjr_fvol"] + (1 - _gw) * w["vix_implied"]
        crps, s50, c50 = eval_model(_vfn)
        tag = ""
        if crps < best_vix["crps"]:
            best_vix = {"crps": crps, "w": gw, "s50": s50, "c50": c50}
            tag = " ←"
        print(f"  {gw:>8.2f}  {crps:>8.5f}  {s50:>10.1%}  {c50:>7.1%}{tag}")
    print(f"\n  Best GARCH weight: {best_vix['w']:.2f} "
          f"(CRPS={best_vix['crps']:.5f}, stressed50={best_vix['s50']:.1%}, "
          f"calm50={best_vix['c50']:.1%})")

    # ── 4. Sweep: VRP scale + center ───────────────────────────────────
    print("\n" + "=" * 80)
    print("  SWEEP 2: VRP Dynamic Weighting")
    print("  vix_w = clip(center - VRP_SCALE * vrp, lo, hi)")
    print("=" * 80)

    vrp_scales = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
    vrp_centers = [0.3, 0.4, 0.5, 0.6, 0.7]
    vrp_bounds = [(0.10, 0.90), (0.15, 0.85), (0.20, 0.80)]

    print(f"\n  {'Scale':>6s}  {'Center':>7s}  {'Bounds':>10s}  {'CRPS':>8s}  {'Stressed50':>11s}  {'Calm50':>8s}")
    best_vrp = {"crps": 1.0}
    for scale, center, (lo, hi) in product(vrp_scales, vrp_centers, vrp_bounds):
        def _vfn(w, _s=scale, _c=center, _lo=lo, _hi=hi):
            vrp = w["vix_implied"] - w["trailing_vol"]
            vix_w = float(np.clip(_c - _s * vrp, _lo, _hi))
            return (1 - vix_w) * w["gjr_fvol"] + vix_w * w["vix_implied"]
        crps, s50, c50 = eval_model(_vfn)
        if crps < best_vrp["crps"]:
            best_vrp = {"crps": crps, "scale": scale, "center": center,
                        "lo": lo, "hi": hi, "s50": s50, "c50": c50}
    # Print top result and nearby
    print(f"\n  Best: scale={best_vrp['scale']}, center={best_vrp['center']}, "
          f"bounds=({best_vrp['lo']},{best_vrp['hi']})")
    print(f"  CRPS={best_vrp['crps']:.5f}, stressed50={best_vrp['s50']:.1%}, "
          f"calm50={best_vrp['c50']:.1%}")

    # Print full grid for best bounds
    blo, bhi = best_vrp["lo"], best_vrp["hi"]
    print(f"\n  Full grid for bounds=({blo},{bhi}):")
    print(f"  {'Scale':>6s}  {'Center':>7s}  {'CRPS':>8s}  {'Stressed50':>11s}  {'Calm50':>8s}")
    for scale, center in product(vrp_scales, vrp_centers):
        def _vfn(w, _s=scale, _c=center, _lo=blo, _hi=bhi):
            vrp = w["vix_implied"] - w["trailing_vol"]
            vix_w = float(np.clip(_c - _s * vrp, _lo, _hi))
            return (1 - vix_w) * w["gjr_fvol"] + vix_w * w["vix_implied"]
        crps, s50, c50 = eval_model(_vfn)
        tag = " ←" if (scale == best_vrp["scale"] and center == best_vrp["center"]) else ""
        print(f"  {scale:>6.1f}  {center:>7.1f}  {crps:>8.5f}  {s50:>10.1%}  {c50:>7.1%}{tag}")

    # ── 5. Sweep: ImpVol blend weight + TLT_DURATION ───────────────────
    print("\n" + "=" * 80)
    print("  SWEEP 3: Asset-Specific Implied Vol (ImpVol)")
    print("  vol = garch_w * GARCH + (1-garch_w) * asset_implied")
    print("=" * 80)

    iv_garch_weights = [0.30, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70]
    tlt_durations = [12.0, 15.0, 17.0, 19.0, 22.0]

    print(f"\n  {'GARCH_w':>8s}  {'TLT_dur':>8s}  {'CRPS':>8s}  {'Stressed50':>11s}  {'Calm50':>8s}")
    best_iv = {"crps": 1.0}
    for gw, dur in product(iv_garch_weights, tlt_durations):
        def _vfn(w, _gw=gw, _dur=dur):
            if w["tk"] == "TLT" and w["move_valid"]:
                ai = _dur * w["move_lvl"] / 10000.0
            elif w["tk"] == "GLD" and w["gvz_valid"]:
                ai = w["gvz_lvl"] / 100.0
            else:
                ai = w["vix_implied"]
            return _gw * w["gjr_fvol"] + (1 - _gw) * ai
        crps, s50, c50 = eval_model(_vfn)
        if crps < best_iv["crps"]:
            best_iv = {"crps": crps, "gw": gw, "dur": dur, "s50": s50, "c50": c50}
    print(f"\n  Best: GARCH_w={best_iv['gw']}, TLT_dur={best_iv['dur']}")
    print(f"  CRPS={best_iv['crps']:.5f}, stressed50={best_iv['s50']:.1%}, "
          f"calm50={best_iv['c50']:.1%}")

    # Print full grid
    print("\n  Full grid:")
    print(f"  {'GARCH_w':>8s}  {'TLT_dur':>8s}  {'CRPS':>8s}  {'Stressed50':>11s}  {'Calm50':>8s}")
    for gw, dur in product(iv_garch_weights, tlt_durations):
        def _vfn(w, _gw=gw, _dur=dur):
            if w["tk"] == "TLT" and w["move_valid"]:
                ai = _dur * w["move_lvl"] / 10000.0
            elif w["tk"] == "GLD" and w["gvz_valid"]:
                ai = w["gvz_lvl"] / 100.0
            else:
                ai = w["vix_implied"]
            return _gw * w["gjr_fvol"] + (1 - _gw) * ai
        crps, s50, c50 = eval_model(_vfn)
        tag = " ←" if (gw == best_iv["gw"] and dur == best_iv["dur"]) else ""
        print(f"  {gw:>8.2f}  {dur:>8.1f}  {crps:>8.5f}  {s50:>10.1%}  {c50:>7.1%}{tag}")

    # ── 6. Sweep: SKEW scale ───────────────────────────────────────────
    print("\n" + "=" * 80)
    print("  SWEEP 4: SKEW Index → Skewness Adjustment")
    print("  skew_adj = -(SKEW - 100) / SKEW_SCALE")
    print("  Uses GARCH vol (no implied vol blend)")
    print("=" * 80)

    skew_scales = [100.0, 150.0, 200.0, 250.0, 300.0, 400.0, 500.0, 750.0, 1000.0]
    print(f"\n  {'SKEW_SCALE':>11s}  {'CRPS':>8s}  {'Stressed50':>11s}  {'Calm50':>8s}")
    best_skew = {"crps": 1.0}
    for ss in skew_scales:
        def _pfn(w, _ss=ss):
            if w["skew_valid"]:
                adj = -(w["skew_lvl"] - 100) / _ss
                lam = float(np.clip(w["gjr_lam"] + adj, -0.9, 0.9))
            else:
                lam = w["gjr_lam"]
            return np.array([w["gjr_eta"], lam])
        def _vfn(w):
            return w["gjr_fvol"]
        crps, s50, c50 = eval_model(_vfn, _pfn)
        tag = ""
        if crps < best_skew["crps"]:
            best_skew = {"crps": crps, "scale": ss, "s50": s50, "c50": c50}
            tag = " ←"
        print(f"  {ss:>11.0f}  {crps:>8.5f}  {s50:>10.1%}  {c50:>7.1%}{tag}")
    print(f"\n  Best SKEW_SCALE: {best_skew['scale']:.0f} "
          f"(CRPS={best_skew['crps']:.5f})")

    # ── 7. Sweep: Credit scale ─────────────────────────────────────────
    print("\n" + "=" * 80)
    print("  SWEEP 5: Credit Spread Vol Multiplier")
    print("  credit_mult = clip(1.0 - CREDIT_SCALE * cc_63, 1.0, max_mult)")
    print("=" * 80)

    credit_scales = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]
    credit_maxes = [1.5, 2.0, 3.0]
    print(f"\n  {'CR_SCALE':>9s}  {'Max':>5s}  {'CRPS':>8s}  {'Stressed50':>11s}  {'Calm50':>8s}")
    best_cr = {"crps": 1.0}
    for cs, cm in product(credit_scales, credit_maxes):
        def _vfn(w, _cs=cs, _cm=cm):
            mult = min(_cm, max(1.0, 1.0 - _cs * w["cc"]))
            return w["gjr_fvol"] * mult
        crps, s50, c50 = eval_model(_vfn)
        tag = ""
        if crps < best_cr["crps"]:
            best_cr = {"crps": crps, "scale": cs, "max": cm, "s50": s50, "c50": c50}
            tag = " ←"
        print(f"  {cs:>9.1f}  {cm:>5.1f}  {crps:>8.5f}  {s50:>10.1%}  {c50:>7.1%}{tag}")
    print(f"\n  Best: scale={best_cr['scale']}, max={best_cr['max']} "
          f"(CRPS={best_cr['crps']:.5f})")

    # ── 8. Baseline comparison ─────────────────────────────────────────
    print("\n" + "=" * 80)
    print("  BASELINE COMPARISON")
    print("=" * 80)

    # Pure GJR-skewt
    crps_gjr, s50_gjr, c50_gjr = eval_model(lambda w: w["gjr_fvol"])
    print(f"\n  GJR-skewt (no external):  CRPS={crps_gjr:.5f}  "
          f"stressed50={s50_gjr:.1%}  calm50={c50_gjr:.1%}")

    # Best VIX blend
    def _best_vix(w):
        gw = best_vix["w"]
        return gw * w["gjr_fvol"] + (1 - gw) * w["vix_implied"]
    crps_v, s50_v, c50_v = eval_model(_best_vix)
    print(f"  Best VIX (w={best_vix['w']:.2f}):  CRPS={crps_v:.5f}  "
          f"stressed50={s50_v:.1%}  calm50={c50_v:.1%}")

    # Best VRP
    def _best_vrp(w):
        vrp = w["vix_implied"] - w["trailing_vol"]
        vix_w = float(np.clip(best_vrp["center"] - best_vrp["scale"] * vrp,
                              best_vrp["lo"], best_vrp["hi"]))
        return (1 - vix_w) * w["gjr_fvol"] + vix_w * w["vix_implied"]
    crps_vr, s50_vr, c50_vr = eval_model(_best_vrp)
    print(f"  Best VRP (s={best_vrp['scale']}, c={best_vrp['center']}, "
          f"b={best_vrp['lo']}-{best_vrp['hi']}):  CRPS={crps_vr:.5f}  "
          f"stressed50={s50_vr:.1%}  calm50={c50_vr:.1%}")

    # Best ImpVol
    def _best_iv(w):
        gw = best_iv["gw"]
        dur = best_iv["dur"]
        if w["tk"] == "TLT" and w["move_valid"]:
            ai = dur * w["move_lvl"] / 10000.0
        elif w["tk"] == "GLD" and w["gvz_valid"]:
            ai = w["gvz_lvl"] / 100.0
        else:
            ai = w["vix_implied"]
        return gw * w["gjr_fvol"] + (1 - gw) * ai
    crps_iv, s50_iv, c50_iv = eval_model(_best_iv)
    print(f"  Best ImpVol (w={best_iv['gw']:.2f}, dur={best_iv['dur']:.0f}):  "
          f"CRPS={crps_iv:.5f}  stressed50={s50_iv:.1%}  calm50={c50_iv:.1%}")

    # ── 9. Combined: best ImpVol + VRP + SKEW ──────────────────────────
    print("\n" + "=" * 80)
    print("  COMBINED: Best ImpVol + VRP + SKEW")
    print("=" * 80)

    # ImpVol + VRP: use asset-specific IV with VRP-dynamic weighting
    def _combo_vol(w):
        dur = best_iv["dur"]
        if w["tk"] == "TLT" and w["move_valid"]:
            ai = dur * w["move_lvl"] / 10000.0
        elif w["tk"] == "GLD" and w["gvz_valid"]:
            ai = w["gvz_lvl"] / 100.0
        else:
            ai = w["vix_implied"]
        vrp = ai - w["trailing_vol"]
        vix_w = float(np.clip(best_vrp["center"] - best_vrp["scale"] * vrp,
                              best_vrp["lo"], best_vrp["hi"]))
        return (1 - vix_w) * w["gjr_fvol"] + vix_w * ai

    def _combo_skew_params(w):
        if w["skew_valid"]:
            adj = -(w["skew_lvl"] - 100) / best_skew["scale"]
            lam = float(np.clip(w["gjr_lam"] + adj, -0.9, 0.9))
        else:
            lam = w["gjr_lam"]
        return np.array([w["gjr_eta"], lam])

    # ImpVol+VRP only
    crps_ivr, s50_ivr, c50_ivr = eval_model(_combo_vol)
    print(f"\n  ImpVol+VRP:       CRPS={crps_ivr:.5f}  "
          f"stressed50={s50_ivr:.1%}  calm50={c50_ivr:.1%}")

    # ImpVol+VRP+SKEW
    crps_all, s50_all, c50_all = eval_model(_combo_vol, _combo_skew_params)
    print(f"  ImpVol+VRP+SKEW:  CRPS={crps_all:.5f}  "
          f"stressed50={s50_all:.1%}  calm50={c50_all:.1%}")

    # ImpVol+SKEW (no VRP)
    crps_is, s50_is, c50_is = eval_model(_best_iv, _combo_skew_params)
    print(f"  ImpVol+SKEW:      CRPS={crps_is:.5f}  "
          f"stressed50={s50_is:.1%}  calm50={c50_is:.1%}")

    # VIX+SKEW
    def _vix_skew_params(w):
        return _combo_skew_params(w)
    crps_vs, s50_vs, c50_vs = eval_model(_best_vix, _vix_skew_params)
    print(f"  VIX+SKEW:         CRPS={crps_vs:.5f}  "
          f"stressed50={s50_vs:.1%}  calm50={c50_vs:.1%}")

    print(f"\n  Total runtime: {time.perf_counter() - t0:.0f}s")
    print(f"  Output saved to {_OUT}")
    print("=" * 80)

    _fh.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
