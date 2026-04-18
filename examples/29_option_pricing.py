"""Example 29 — Monte Carlo Option Pricing
=========================================

Price European, Asian, and barrier options using calibrated
simulation engines and compare to Black-Scholes.

The key insight: Black-Scholes assumes constant volatility, so it
produces a flat implied vol across strikes.  Real markets show a
"volatility smile" (OTM puts are more expensive than ATM options).
Stochastic vol and jump-diffusion models reproduce this smile.

This example demonstrates:
  1. GJR-GARCH skewt MC → vol smile from asymmetric GARCH + fat tails
  2. Heston stochastic vol MC → vol smile from mean-reverting vol
  3. Black-Scholes (flat vol baseline)
  4. Implied vol surface extraction
  5. Greeks via bump-and-reprice
  6. Exotic option pricing (Asian, barrier)

Use case:
  Derivatives pricing and vol surface construction are core skills
  for sell-side quant roles.  This example shows the full pipeline
  from calibration to pricing to risk sensitivities.
"""

import time
import numpy as np
import yfinance as yf
from scipy.stats import norm
from arch import arch_model

from synthquant.analytics.options import MCOptionPricer, ImpliedVolSurface, _bs_call
from synthquant.simulation.engines.heston import HestonEngine

# ── Configuration ────────────────────────────────────────────────────
TICKER = "SPY"
START = "2024-01-01"
END = "2026-12-31"
N_PATHS = 10_000
SCALE = 100
R = 0.043              # risk-free rate (~current 10Y)
MATURITIES_DAYS = [21, 63, 126]   # 1m, 3m, 6m
MONEYNESS = np.arange(0.85, 1.16, 0.025)  # 85% to 115%

# ── Main ─────────────────────────────────────────────────────────────
t0 = time.time()
output_lines = []


def log(msg=""):
    print(msg)
    output_lines.append(msg)


log("=" * 80)
log("  MONTE CARLO OPTION PRICING")
log(f"  {TICKER}  |  {N_PATHS:,} paths  |  r={R:.1%}")
log(f"  Maturities: {[f'{d}d' for d in MATURITIES_DAYS]}")
log(f"  Moneyness: {MONEYNESS[0]:.0%} – {MONEYNESS[-1]:.0%}")
log("=" * 80)

# ── Download & prepare data ──────────────────────────────────────────
df = yf.download(TICKER, start=START, end=END, progress=False)
close = df["Close"].squeeze().dropna()
returns = close.pct_change().dropna()
S0 = float(close.iloc[-1])
trailing_vol = float(returns.iloc[-252:].std() * np.sqrt(252))

log(f"\n  {TICKER}: S0=${S0:.2f}  trailing vol={trailing_vol:.1%}")
log(f"  Using last {len(returns)} daily returns for calibration")

# ── Fit GJR-GARCH skewt ─────────────────────────────────────────────
log("\n  Fitting GJR-GARCH(1,1) skewt...")
am = arch_model(returns.values * SCALE, vol="GARCH", p=1, o=1, q=1,
                dist="skewt", mean="Constant")
res = am.fit(disp="off", show_warning=False)

omega = res.params["omega"]
alpha = res.params["alpha[1]"]
gamma = res.params.get("gamma[1]", 0)
beta = res.params["beta[1]"]
mu = res.params["mu"]
eta = res.params["eta"]
lam = res.params["lambda"]
cv = np.asarray(res.conditional_volatility)
h_last = cv[-1] ** 2

log(f"    omega={omega:.6f}  alpha={alpha:.4f}  gamma={gamma:.4f}  "
    f"beta={beta:.4f}")
log(f"    eta={eta:.2f} (df)  lambda={lam:.3f} (skew)")
log(f"    Current cond vol: {np.sqrt(h_last)/SCALE*np.sqrt(252):.1%} annualised")

# ── Calibrate Heston parameters from GARCH ───────────────────────────
# GARCH → Heston mapping (Barone-Adesi et al., 2008)
persistence = alpha + 0.5 * gamma + beta
v0 = h_last / SCALE ** 2 * 252                  # annualised variance
theta_h = (omega / (1 - persistence)) / SCALE ** 2 * 252
kappa_h = (1 - persistence) * 252
sigma_v_h = alpha * np.sqrt(2) * np.sqrt(252)
rho_h = -0.70  # typical equity leverage

log(f"\n  Heston parameters (from GARCH mapping):")
log(f"    v0={v0:.4f}  theta={theta_h:.4f}  kappa={kappa_h:.2f}  "
    f"sigma_v={sigma_v_h:.2f}  rho={rho_h:.2f}")

# ── Generate MC price paths ─────────────────────────────────────────
max_steps = max(MATURITIES_DAYS)
pricer = MCOptionPricer()
iv_calc = ImpliedVolSurface()

# --- GJR-GARCH paths ---
log(f"\n  Simulating {N_PATHS:,} GJR-GARCH paths ({max_steps} steps)...")
t_sim = time.time()

# Simulate using the fitted GARCH model (includes skewt innovations)
garch_log_rets = np.empty((N_PATHS, max_steps))
for i in range(N_PATHS):
    sim = res.model.simulate(res.params, nobs=max_steps)
    garch_log_rets[i] = sim["data"].values / SCALE

# Convert to price paths
garch_price_paths = np.hstack([
    np.full((N_PATHS, 1), S0),
    S0 * np.exp(np.cumsum(garch_log_rets, axis=1))
])
log(f"    GARCH simulation: {time.time()-t_sim:.1f}s")

# --- Heston paths ---
log(f"  Simulating {N_PATHS:,} Heston paths ({max_steps} steps)...")
t_sim = time.time()
heston = HestonEngine()
dt = 1.0 / 252
heston_price_paths = heston.simulate(
    n_paths=N_PATHS, n_steps=max_steps, dt=dt,
    S0=S0, v0=v0, mu=R, kappa=kappa_h, theta=theta_h,
    sigma_v=sigma_v_h, rho=rho_h, random_state=42,
)
log(f"    Heston simulation: {time.time()-t_sim:.1f}s")

# ── Price options across strikes and maturities ──────────────────────
strikes = S0 * MONEYNESS
n_strikes = len(strikes)
n_mats = len(MATURITIES_DAYS)

# Storage: mc_prices[model][maturity_idx, strike_idx]
mc_prices = {
    "GARCH": np.zeros((n_mats, n_strikes)),
    "Heston": np.zeros((n_mats, n_strikes)),
    "BS": np.zeros((n_mats, n_strikes)),
}

log(f"\n  Pricing European calls across {n_strikes} strikes × "
    f"{n_mats} maturities...")

for mi, T_days in enumerate(MATURITIES_DAYS):
    T = T_days / 252.0
    # Slice paths to this maturity
    garch_paths_T = garch_price_paths[:, :T_days + 1]
    heston_paths_T = heston_price_paths[:, :T_days + 1]

    for si, K in enumerate(strikes):
        mc_prices["GARCH"][mi, si] = pricer.price_european(
            garch_paths_T, K, T, R, "call")
        mc_prices["Heston"][mi, si] = pricer.price_european(
            heston_paths_T, K, T, R, "call")
        mc_prices["BS"][mi, si] = _bs_call(S0, K, T, R, trailing_vol)

# ── Extract implied volatilities ─────────────────────────────────────
log("\n  Extracting implied vol surface...")
maturities_years = np.array(MATURITIES_DAYS) / 252.0

iv_garch = iv_calc.compute(S0, strikes, maturities_years,
                           mc_prices["GARCH"], R, "call")
iv_heston = iv_calc.compute(S0, strikes, maturities_years,
                            mc_prices["Heston"], R, "call")
iv_bs = iv_calc.compute(S0, strikes, maturities_years,
                        mc_prices["BS"], R, "call")

# ── Print option price comparison ────────────────────────────────────
log(f"\n{'=' * 80}")
log("  EUROPEAN CALL PRICES")
log("=" * 80)

for mi, T_days in enumerate(MATURITIES_DAYS):
    T = T_days / 252.0
    log(f"\n  Maturity: {T_days}d ({T:.2f}yr)")
    log(f"  {'Strike':>8s} {'Money':>7s} {'BS':>8s} {'GARCH':>8s} "
        f"{'Heston':>8s} {'IV_BS':>7s} {'IV_GCH':>7s} {'IV_Hst':>7s}")

    for si, K in enumerate(strikes):
        m = K / S0
        bs_p = mc_prices["BS"][mi, si]
        gc_p = mc_prices["GARCH"][mi, si]
        ht_p = mc_prices["Heston"][mi, si]
        iv_b = iv_bs[mi, si]
        iv_g = iv_garch[mi, si]
        iv_h = iv_heston[mi, si]

        bs_str = f"{iv_b:>6.1%}" if not np.isnan(iv_b) else f"{'N/A':>6s}"
        gc_str = f"{iv_g:>6.1%}" if not np.isnan(iv_g) else f"{'N/A':>6s}"
        ht_str = f"{iv_h:>6.1%}" if not np.isnan(iv_h) else f"{'N/A':>6s}"

        log(f"  {K:>8.1f} {m:>6.0%} {bs_p:>8.2f} {gc_p:>8.2f} "
            f"{ht_p:>8.2f} {bs_str:>7s} {gc_str:>7s} {ht_str:>7s}")

# Cleaner table output
log(f"\n{'=' * 80}")
log("  IMPLIED VOLATILITY SMILE (by maturity)")
log("=" * 80)

for mi, T_days in enumerate(MATURITIES_DAYS):
    log(f"\n  T = {T_days} days")
    log(f"  {'Moneyness':>10s} {'BS':>8s} {'GARCH':>8s} {'Heston':>8s} "
        f"{'GCH-BS':>8s} {'Hst-BS':>8s}")

    for si, K in enumerate(strikes):
        m = K / S0
        iv_b = iv_bs[mi, si]
        iv_g = iv_garch[mi, si]
        iv_h = iv_heston[mi, si]

        bs_str = f"{iv_b:.1%}" if not np.isnan(iv_b) else "  N/A"
        gc_str = f"{iv_g:.1%}" if not np.isnan(iv_g) else "  N/A"
        ht_str = f"{iv_h:.1%}" if not np.isnan(iv_h) else "  N/A"

        if not np.isnan(iv_g) and not np.isnan(iv_b):
            diff_g = f"{(iv_g - iv_b)*100:+.1f}pp"
        else:
            diff_g = "  N/A"
        if not np.isnan(iv_h) and not np.isnan(iv_b):
            diff_h = f"{(iv_h - iv_b)*100:+.1f}pp"
        else:
            diff_h = "  N/A"

        log(f"  {m:>9.0%} {bs_str:>8s} {gc_str:>8s} {ht_str:>8s} "
            f"{diff_g:>8s} {diff_h:>8s}")

# ── Vol smile summary ────────────────────────────────────────────────
log(f"\n{'=' * 80}")
log("  VOL SMILE METRICS")
log("=" * 80)

for mi, T_days in enumerate(MATURITIES_DAYS):
    log(f"\n  T = {T_days} days:")
    for model_name, iv_surface in [("GARCH", iv_garch), ("Heston", iv_heston),
                                    ("BS", iv_bs)]:
        row = iv_surface[mi]
        valid = ~np.isnan(row)
        if valid.sum() < 3:
            continue
        atm_idx = np.argmin(np.abs(MONEYNESS - 1.0))
        atm_iv = row[atm_idx] if not np.isnan(row[atm_idx]) else np.nanmean(row)

        # Skew: IV(90%) - IV(110%)
        idx_90 = np.argmin(np.abs(MONEYNESS - 0.90))
        idx_110 = np.argmin(np.abs(MONEYNESS - 1.10))
        skew = row[idx_90] - row[idx_110] if (
            not np.isnan(row[idx_90]) and not np.isnan(row[idx_110])) else float("nan")

        # Curvature: (IV(90%) + IV(110%))/2 - IV(100%)
        curv = ((row[idx_90] + row[idx_110]) / 2 - atm_iv) if (
            not np.isnan(row[idx_90]) and not np.isnan(row[idx_110])
            and not np.isnan(atm_iv)) else float("nan")

        log(f"    {model_name:<8s}  ATM IV={atm_iv:.1%}  "
            f"Skew(90-110)={skew:+.1%}  Curvature={curv:+.2%}")

# ── Greeks ───────────────────────────────────────────────────────────
log(f"\n{'=' * 80}")
log("  GREEKS (3-month ATM call)")
log("=" * 80)

T_greeks = 63 / 252.0
K_atm = S0  # ATM
garch_paths_3m = garch_price_paths[:, :64]
heston_paths_3m = heston_price_paths[:, :64]

greeks_garch = pricer.compute_greeks(garch_paths_3m, K_atm, T_greeks, R,
                                      bump_size=0.01, option_type="call")
greeks_heston = pricer.compute_greeks(heston_paths_3m, K_atm, T_greeks, R,
                                       bump_size=0.01, option_type="call")

# BS Greeks (analytical)
d1 = (np.log(S0 / K_atm) + (R + 0.5 * trailing_vol**2) * T_greeks) / (
    trailing_vol * np.sqrt(T_greeks))
d2 = d1 - trailing_vol * np.sqrt(T_greeks)
bs_delta = float(norm.cdf(d1))
bs_gamma = float(norm.pdf(d1) / (S0 * trailing_vol * np.sqrt(T_greeks)))
bs_vega = float(S0 * norm.pdf(d1) * np.sqrt(T_greeks))

log(f"  {'':>12s} {'BS':>10s} {'GARCH':>10s} {'Heston':>10s}")
log(f"  {'Delta':>12s} {bs_delta:>10.4f} {greeks_garch['delta']:>10.4f} "
    f"{greeks_heston['delta']:>10.4f}")
log(f"  {'Gamma':>12s} {bs_gamma:>10.6f} {greeks_garch['gamma']:>10.6f} "
    f"{greeks_heston['gamma']:>10.6f}")
log(f"  {'Vega':>12s} {bs_vega:>10.2f} {greeks_garch['vega']:>10.2f} "
    f"{greeks_heston['vega']:>10.2f}")

# ── Exotic options ───────────────────────────────────────────────────
log(f"\n{'=' * 80}")
log("  EXOTIC OPTION PRICING (3-month, GARCH model)")
log("=" * 80)

T_exotic = 63 / 252.0
K_exotic = S0  # ATM

# European (baseline)
euro_call = pricer.price_european(garch_paths_3m, K_exotic, T_exotic, R, "call")
euro_put = pricer.price_european(garch_paths_3m, K_exotic, T_exotic, R, "put")

# Asian
asian_call = pricer.price_asian(garch_paths_3m, K_exotic, T_exotic, R, "call")
asian_put = pricer.price_asian(garch_paths_3m, K_exotic, T_exotic, R, "put")

# Barrier: down-and-out put (protection disappears if crash too severe)
barrier_95 = S0 * 0.90  # 10% below spot
dao_put = pricer.price_barrier(garch_paths_3m, K_exotic, barrier_95,
                                T_exotic, R, "knock_out", "put")

# Barrier: up-and-out call (capped upside)
barrier_110 = S0 * 1.10
uao_call = pricer.price_barrier(garch_paths_3m, K_exotic, barrier_110,
                                 T_exotic, R, "knock_out", "call")

log(f"  {'Type':<28s} {'Price':>8s} {'vs European':>12s}")
log(f"  {'European Call':<28s} {euro_call:>8.2f} {'baseline':>12s}")
log(f"  {'European Put':<28s} {euro_put:>8.2f} {'baseline':>12s}")
log(f"  {'Asian Call (avg price)':<28s} {asian_call:>8.2f} "
    f"{asian_call/euro_call-1:>+11.1%}")
log(f"  {'Asian Put (avg price)':<28s} {asian_put:>8.2f} "
    f"{asian_put/euro_put-1:>+11.1%}")
dao_label = f"Down-Out Put (B={barrier_95:.0f})"
uao_label = f"Up-Out Call (B={barrier_110:.0f})"
log(f"  {dao_label:<28s} {dao_put:>8.2f} "
    f"{dao_put/euro_put-1:>+11.1%}")
log(f"  {uao_label:<28s} {uao_call:>8.2f} "
    f"{uao_call/euro_call-1:>+11.1%}")

# ── Interpretation ───────────────────────────────────────────────────
log(f"\n{'=' * 80}")
log("  INTERPRETATION")
log("=" * 80)
log("""
  Black-Scholes assumes constant volatility, producing a flat
  implied vol surface.  Real markets show:
  - SKEW: OTM puts trade at higher IV than OTM calls (crash fear)
  - SMILE: both deep OTM puts and calls have higher IV than ATM
  - TERM STRUCTURE: short-term vol reacts more to recent events

  The GJR-GARCH skewt model reproduces these features because:
  - Asymmetric GARCH (gamma > 0): negative returns increase vol
    more than positive returns → creates skew
  - Skewed-t innovations: fat left tail → higher OTM put prices
  - Time-varying vol: naturally creates a vol term structure

  The Heston model produces a smile through:
  - Stochastic variance with mean-reversion → vol of vol → smile
  - Negative correlation (rho < 0) → leverage effect → skew

  For exotic options:
  - Asian options are cheaper (averaging smooths out extremes)
  - Barrier options are cheaper (path constraint reduces expected payoff)
  - These differences grow with vol → stochastic vol matters more for exotics

  The vol smile has direct P&L implications: if you delta-hedge with
  BS (flat vol), you'll systematically lose money on OTM puts because
  you're underestimating their true cost.
""")
total = time.time() - t0
log(f"  Total runtime: {total:.0f}s")

with open("_option_pricing_output.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(output_lines))
log(f"  Output saved to _option_pricing_output.txt")
log("=" * 80)
