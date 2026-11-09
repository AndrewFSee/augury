"""07 - Probabilistic Forecasting

Demonstrates end-to-end probabilistic forecasting with the
ProbabilisticForecaster: fit on historical data, generate forecast
distributions, and validate calibration via walk-forward scoring.
"""

import numpy as np

from synthquant.forecast import ProbabilisticForecaster


def generate_market_data(
    n_calm: int = 500,
    n_volatile: int = 250,
    n_calm2: int = 250,
    seed: int = 42,
) -> np.ndarray:
    """Generate synthetic market returns with regime shifts."""
    rng = np.random.default_rng(seed)
    calm1 = rng.normal(0.0003, 0.008, n_calm)       # Bull / low vol
    crisis = rng.normal(-0.002, 0.028, n_volatile)   # Bear / high vol
    calm2 = rng.normal(0.0004, 0.009, n_calm2)       # Recovery
    return np.concatenate([calm1, crisis, calm2])


def main() -> None:
    print("=" * 65)
    print("  PROBABILISTIC FORECASTING DEMO")
    print("=" * 65)

    # ------------------------------------------------------------------
    # 1. Generate synthetic market data
    # ------------------------------------------------------------------
    returns = generate_market_data()
    train_size = 750
    train = returns[:train_size]
    holdout = returns[train_size:]

    print(f"\nData: {len(returns)} days total")
    print(f"  Training:  {train_size} days")
    print(f"  Holdout:   {len(holdout)} days")

    # ------------------------------------------------------------------
    # 2. Fit the forecaster
    # ------------------------------------------------------------------
    print("\n--- Fitting ProbabilisticForecaster (HMM + GARCH) ---")
    fc = ProbabilisticForecaster(
        detector="hmm",
        n_regimes=2,
        n_paths=10_000,
        vol_model="GARCH",
        random_state=42,
    )
    fc.fit(train)

    # ------------------------------------------------------------------
    # 3. Generate a 1-quarter (63 day) forecast
    # ------------------------------------------------------------------
    result = fc.forecast(horizon=63, current_price=100.0)
    s = result.summary()

    print(f"\nCurrent regime:           {s['current_regime']}")
    proba_str = ", ".join(f"{p:.1%}" for p in s["regime_probabilities"])
    print(f"Regime probabilities:     [{proba_str}]")

    print("\n--- 63-Day Forecast Distribution (10,000 paths) ---")
    print(f"  Expected return:        {s['expected_return']:+.2%}")
    print(f"  Median return:          {s['median_return']:+.2%}")
    print(f"  Return std dev:         {s['return_std']:.2%}")
    print(f"  P(positive return):     {s['prob_positive']:.1%}")
    print(f"  P(loss > 10%):          {s['prob_loss_gt_10pct']:.1%}")
    print(f"  P(loss > 20%):          {s['prob_loss_gt_20pct']:.1%}")

    print("\n--- Risk Metrics ---")
    print(f"  VaR 95%:                {s['var_95']:+.2%}")
    print(f"  CVaR 95%:               {s['cvar_95']:+.2%}")
    print(f"  VaR 99%:                {s['var_99']:+.2%}")
    print(f"  CVaR 99%:               {s['cvar_99']:+.2%}")
    print(f"  Median max drawdown:    {s['median_max_drawdown']:+.2%}")

    print("\n--- Price Fan (at S0 = $100) ---")
    print(f"  1st percentile:         ${s['price_p5']:.2f}")
    print(f"  25th percentile:        ${s['price_p25']:.2f}")
    print(f"  Median:                 ${s['price_p50']:.2f}")
    print(f"  75th percentile:        ${s['price_p75']:.2f}")
    print(f"  99th percentile:        ${s['price_p95']:.2f}")

    # ------------------------------------------------------------------
    # 4. Score against holdout
    # ------------------------------------------------------------------
    actual_holdout = holdout[:63]
    scores = fc.score(result, actual_holdout)

    print(f"\n--- Forecast Validation (vs {len(actual_holdout)}-day holdout) ---")
    print(f"  Actual terminal return: {scores['actual_terminal_return']:+.2%}")
    print(f"  Forecast mean return:   {scores['forecast_mean_return']:+.2%}")
    print(f"  CRPS:                   {scores['crps']:.6f}")
    print(f"  KS statistic:           {scores['ks_test']['statistic']:.4f}")
    print(f"  KS p-value:             {scores['ks_test']['p_value']:.4f}")
    print(f"  Realised annual vol:    {scores['realised_annual_vol']:.2%}")
    print(f"  Forecast annual vol:    {scores['forecast_annual_vol']:.2%}")
    print(f"  In 90% interval:        {scores['actual_in_90pct_interval']}")
    print(f"  In 50% interval:        {scores['actual_in_50pct_interval']}")

    # ------------------------------------------------------------------
    # 5. Walk-forward calibration test
    # ------------------------------------------------------------------
    print("\n--- Walk-Forward Calibration (21-day windows) ---")
    window = 500
    step = 21
    in_90 = 0
    in_50 = 0
    n_windows = 0
    crps_list = []

    for start in range(0, len(returns) - window - step, step):
        train_w = returns[start : start + window]
        actual_w = returns[start + window : start + window + step]
        if len(actual_w) < step:
            break

        fc_w = ProbabilisticForecaster(
            detector="hmm", n_regimes=2, n_paths=5_000,
            vol_model="GARCH", random_state=start,
        )
        fc_w.fit(train_w)
        res_w = fc_w.forecast(horizon=step, current_price=100.0)
        sc_w = fc_w.score(res_w, actual_w)

        if sc_w["actual_in_90pct_interval"]:
            in_90 += 1
        if sc_w["actual_in_50pct_interval"]:
            in_50 += 1
        crps_list.append(sc_w["crps"])
        n_windows += 1

    if n_windows > 0:
        print(f"  Windows tested:         {n_windows}")
        print(f"  90% coverage:           {in_90 / n_windows:.1%}  (target: ~90%)")
        print(f"  50% coverage:           {in_50 / n_windows:.1%}  (target: ~50%)")
        print(f"  Mean CRPS:              {np.mean(crps_list):.6f}")

    # ------------------------------------------------------------------
    # 6. Compare detectors
    # ------------------------------------------------------------------
    print("\n--- Detector Comparison (63-day forecast) ---")
    for det_name in ("hmm", "ms_garch", "ensemble"):
        fc_d = ProbabilisticForecaster(
            detector=det_name, n_regimes=2, n_paths=5_000,
            vol_model="GARCH", random_state=42,
        )
        fc_d.fit(train)
        r_d = fc_d.forecast(horizon=63, current_price=100.0)
        sc_d = fc_d.score(r_d, actual_holdout)
        print(
            f"  {det_name:12s}  E[r]={r_d.expected_return:+.2%}  "
            f"VaR95={r_d.var_95:+.2%}  "
            f"CRPS={sc_d['crps']:.6f}  "
            f"in_90%={sc_d['actual_in_90pct_interval']}"
        )

    print("\n" + "=" * 65)
    print("  DONE")
    print("=" * 65)


if __name__ == "__main__":
    main()
