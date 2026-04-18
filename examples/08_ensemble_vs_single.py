"""Walk-forward comparison: ProbabilisticForecaster vs EnsembleForecaster."""

import contextlib
import io
import warnings

import numpy as np
from scipy.stats import kurtosis, skew

from synthquant.ensemble_forecast import EnsembleForecaster
from synthquant.forecast import ProbabilisticForecaster


@contextlib.contextmanager
def suppress_optimizer_noise():
    """Suppress GARCH convergence warnings and optimizer stderr noise."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with contextlib.redirect_stderr(io.StringIO()):
            yield


def generate_realistic_market_data(n_days: int = 1000, seed: int = 42) -> tuple:
    """Generate market returns with regime switching, GARCH vol, and jumps.

    Produces data that exhibits:
    - Vol clustering (GARCH dynamics per regime)
    - Regime shifts (Markov switching)
    - Jump events (compound Poisson, more frequent in crisis)
    - Leverage effect (negative returns increase next-period vol)

    Returns (returns, regimes) arrays.
    """
    rng = np.random.default_rng(seed)

    # Markov transition matrix
    # Calm -> Crisis: ~1.5% daily, Crisis -> Calm: ~4% daily
    P = np.array([[0.985, 0.015],
                  [0.04,  0.96]])

    regimes = np.zeros(n_days, dtype=int)
    for t in range(1, n_days):
        regimes[t] = rng.choice(2, p=P[regimes[t - 1]])

    # Per-regime GARCH(1,1) parameters
    # Calm: moderate persistence, ~12% annualised vol
    # Crisis: high persistence, ~35% annualised vol
    params = {
        0: {"omega": 2.0e-6, "alpha": 0.06, "beta": 0.90, "mu": 0.0003},
        1: {"omega": 1.5e-5, "alpha": 0.12, "beta": 0.84, "mu": -0.0008},
    }

    # Jump parameters (compound Poisson per regime)
    jump_params = {
        0: {"lam": 0.004, "mu_j": -0.008, "sigma_j": 0.010},   # rare, small
        1: {"lam": 0.030, "mu_j": -0.030, "sigma_j": 0.020},   # frequent, large
    }

    returns = np.zeros(n_days)
    h = np.zeros(n_days)
    h[0] = 1.5e-4  # ~1.2% daily vol initial

    for t in range(n_days):
        r = regimes[t]
        p = params[r]

        if t > 0:
            leverage = 0.03 if returns[t - 1] < 0 else 0.0
            h[t] = p["omega"] + (p["alpha"] + leverage) * returns[t - 1] ** 2 + p["beta"] * h[t - 1]

        vol = np.sqrt(max(h[t], 1e-10))
        returns[t] = p["mu"] + vol * rng.standard_normal()

        jp = jump_params[r]
        n_jumps = rng.poisson(jp["lam"])
        if n_jumps > 0:
            returns[t] += rng.normal(jp["mu_j"], jp["sigma_j"], size=n_jumps).sum()

    return returns, regimes


def walk_forward(forecaster_cls, returns, label, window=500, step=21, n_paths=5000):
    in_90 = 0
    in_50 = 0
    n_windows = 0
    crps_list = []
    vol_errors = []

    for start in range(0, len(returns) - window - step, step):
        train = returns[start : start + window]
        actual = returns[start + window : start + window + step]
        if len(actual) < step:
            break

        with suppress_optimizer_noise():
            fc = forecaster_cls(n_paths=n_paths, random_state=start)
            fc.fit(train)
            res = fc.forecast(horizon=step, current_price=100.0)
            sc = fc.score(res, actual)

        if sc["actual_in_90pct_interval"]:
            in_90 += 1
        if sc["actual_in_50pct_interval"]:
            in_50 += 1
        crps_list.append(sc["crps"])
        vol_errors.append(abs(sc["realised_annual_vol"] - sc["forecast_annual_vol"]))
        n_windows += 1

    return {
        "label": label,
        "windows": n_windows,
        "coverage_90": in_90 / n_windows if n_windows else 0,
        "coverage_50": in_50 / n_windows if n_windows else 0,
        "mean_crps": np.mean(crps_list) if crps_list else 0,
        "mean_vol_err": np.mean(vol_errors) if vol_errors else 0,
    }


def main():
    warnings.filterwarnings("ignore")
    print("=" * 70)
    print("  HEAD-TO-HEAD: ProbabilisticForecaster vs EnsembleForecaster")
    print("=" * 70)

    returns, regimes = generate_realistic_market_data()
    n_calm = int(np.sum(regimes == 0))
    n_crisis = int(np.sum(regimes == 1))
    ann_vol = float(np.std(returns) * np.sqrt(252))
    ex_kurt = float(kurtosis(returns))
    sk = float(skew(returns))
    print(f"\nRealistic synthetic data: {len(returns)} days")
    print(f"  Regime split:    calm={n_calm}, crisis={n_crisis}")
    print(f"  Annualised vol:  {ann_vol:.1%}")
    print(f"  Excess kurtosis: {ex_kurt:.2f}")
    print(f"  Skewness:        {sk:.3f}")

    # --- Single forecast comparison (training on first 750 days) ---
    train = returns[:750]

    print("\n--- Single 63-day Forecast (750-day training window) ---")
    for name, Cls in [("Single-engine", ProbabilisticForecaster), ("Ensemble", EnsembleForecaster)]:
        with suppress_optimizer_noise():
            fc = Cls(n_paths=10_000, random_state=42)
            fc.fit(train)
            r = fc.forecast(horizon=63, current_price=100.0)
        r = fc.forecast(horizon=63, current_price=100.0)
        k = kurtosis(r.terminal_returns)
        s = skew(r.terminal_returns)
        print(f"\n  {name}:")
        print(f"    E[return]:      {r.expected_return:+.2%}")
        print(f"    VaR 95%:        {r.var_95:+.2%}")
        print(f"    CVaR 95%:       {r.cvar_95:+.2%}")
        print(f"    Return std:     {r.return_std:.2%}")
        print(f"    Kurtosis:       {k:.2f}")
        print(f"    Skew:           {s:.3f}")
        print(f"    P(loss>10%):    {r.prob_loss_gt_10pct:.1%}")
        print(f"    P(loss>20%):    {r.prob_loss_gt_20pct:.1%}")
        if hasattr(r.metadata, "get") and "engine_weights" in r.metadata:
            w = r.metadata["engine_weights"]
            print(f"    Engine weights: Heston={w['heston']:.1%} MertonJD={w['merton_jd']:.1%} RS-GBM={w['regime_switching']:.1%}")
        if isinstance(fc, EnsembleForecaster):
            # Show calibrated params for current regime
            cr = r.current_regime
            cp = fc._calibrated[cr]
            h = cp.heston
            j = cp.jumps
            print(f"    Heston: v0={h['v0']:.4f} theta={h['theta']:.4f} kappa={h['kappa']:.1f} sigma_v={h['sigma_v']:.3f} rho={h['rho']:.2f}")
            print(f"    Jumps:  lambda={j['lambda_j']:.1f}/yr mu_j={j['mu_j']:.4f} sigma_j={j['sigma_j']:.4f}")

    # --- Walk-forward comparison ---
    print("\n--- Walk-Forward Calibration Test (21-day windows) ---")
    for name, Cls in [("Single-engine", ProbabilisticForecaster), ("Ensemble", EnsembleForecaster)]:
        res = walk_forward(Cls, returns, name)
        print(f"\n  {res['label']}:")
        print(f"    Windows:        {res['windows']}")
        print(f"    90% coverage:   {res['coverage_90']:.1%}  (target ~90%)")
        print(f"    50% coverage:   {res['coverage_50']:.1%}  (target ~50%)")
        print(f"    Mean CRPS:      {res['mean_crps']:.6f}  (lower = better)")
        print(f"    Mean vol error: {res['mean_vol_err']:.2%}")

    print("\n" + "=" * 70)
    print("  DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
