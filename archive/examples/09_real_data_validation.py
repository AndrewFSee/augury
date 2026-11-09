"""Walk-forward validation on real market data (SPY, QQQ, and volatile tickers).

Uses the SynthQuant data layer (yfinance) to fetch real prices, then runs
walk-forward forecasting with both ProbabilisticForecaster and EnsembleForecaster.

Install requirement:  pip install "synthquant[data]"
"""

import contextlib
import io
import sys
import warnings

import numpy as np
from scipy.stats import kurtosis, skew

from synthquant.data.ingest import DataIngestor
from synthquant.ensemble_forecast import EnsembleForecaster
from synthquant.forecast import ProbabilisticForecaster


@contextlib.contextmanager
def suppress():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with contextlib.redirect_stderr(io.StringIO()):
            yield


def fetch_returns(symbol: str, start: str, end: str) -> np.ndarray:
    """Fetch daily log returns for a ticker."""
    ingestor = DataIngestor()
    df = ingestor.fetch(symbol, start=start, end=end, interval="1d")
    prices = df["close"].values
    returns = np.diff(np.log(prices))
    # Remove any NaN/inf
    returns = returns[np.isfinite(returns)]
    return returns


def walk_forward(forecaster_cls, returns, window=500, step=21, horizon=21, n_paths=5000):
    """Walk-forward evaluation returning per-window metrics."""
    results = []
    n_windows = 0

    for start in range(0, len(returns) - window - horizon, step):
        train = returns[start : start + window]
        actual = returns[start + window : start + window + horizon]
        if len(actual) < horizon:
            break

        try:
            with suppress():
                fc = forecaster_cls(n_paths=n_paths, random_state=start)
                fc.fit(train)
                res = fc.forecast(horizon=horizon, current_price=100.0)
                sc = fc.score(res, actual)

            results.append({
                "in_90": sc["actual_in_90pct_interval"],
                "in_50": sc["actual_in_50pct_interval"],
                "crps": sc["crps"],
                "vol_err": abs(sc["realised_annual_vol"] - sc["forecast_annual_vol"]),
                "forecast_vol": sc["forecast_annual_vol"],
                "realised_vol": sc["realised_annual_vol"],
            })
            n_windows += 1
        except Exception as e:
            print(f"    [Window {n_windows} failed: {type(e).__name__}: {e}]", flush=True)
            continue

    if not results:
        return None

    return {
        "windows": n_windows,
        "coverage_90": np.mean([r["in_90"] for r in results]),
        "coverage_50": np.mean([r["in_50"] for r in results]),
        "mean_crps": np.mean([r["crps"] for r in results]),
        "mean_vol_err": np.mean([r["vol_err"] for r in results]),
        "mean_forecast_vol": np.mean([r["forecast_vol"] for r in results]),
        "mean_realised_vol": np.mean([r["realised_vol"] for r in results]),
    }


def single_forecast_comparison(returns, symbol):
    """Run a single 63-day forecast from the most recent data."""
    train = returns[-750:]  # last 750 trading days for training
    print(f"\n  --- Single 63-day Forecast ({symbol}, last 750 days training) ---")

    for name, Cls in [("Single-engine", ProbabilisticForecaster), ("Ensemble", EnsembleForecaster)]:
        with suppress():
            fc = Cls(n_paths=10_000, random_state=42)
            fc.fit(train)
            r = fc.forecast(horizon=63, current_price=100.0)

        k = kurtosis(r.terminal_returns)
        s = skew(r.terminal_returns)
        print(f"\n    {name}:")
        print(f"      E[return]:      {r.expected_return:+.2%}")
        print(f"      VaR 95%:        {r.var_95:+.2%}")
        print(f"      CVaR 95%:       {r.cvar_95:+.2%}")
        print(f"      Return std:     {r.return_std:.2%}")
        print(f"      Kurtosis:       {k:.2f}")
        print(f"      Skew:           {s:.3f}")
        print(f"      P(loss>10%):    {r.prob_loss_gt_10pct:.1%}")
        print(f"      P(loss>20%):    {r.prob_loss_gt_20pct:.1%}")
        print(f"      Regime:         {r.current_regime}")

        if "engine_weights" in r.metadata:
            w = r.metadata["engine_weights"]
            print(f"      Engine weights: H={w['heston']:.0%} M={w['merton_jd']:.0%} R={w['regime_switching']:.0%}")

        if isinstance(fc, EnsembleForecaster):
            cr = r.current_regime
            cp = fc._calibrated[cr]
            h = cp.heston
            j = cp.jumps
            print(f"      Heston: v0={h['v0']:.4f} theta={h['theta']:.4f} kappa={h['kappa']:.1f} sv={h['sigma_v']:.2f} rho={h['rho']:.2f}")
            print(f"      Jumps:  lam={j['lambda_j']:.1f}/yr mu_j={j['mu_j']:.4f} sigma_j={j['sigma_j']:.4f}")


def main():
    # Tickers: broad market + higher volatility
    tickers = {
        "SPY":  ("S&P 500 ETF",       "2019-01-01", "2025-12-31"),
        "QQQ":  ("Nasdaq 100 ETF",     "2019-01-01", "2025-12-31"),
        "IWM":  ("Russell 2000 ETF",   "2019-01-01", "2025-12-31"),
        "XLE":  ("Energy Sector ETF",  "2019-01-01", "2025-12-31"),
        "GLD":  ("Gold ETF",           "2019-01-01", "2025-12-31"),
        "EEM":  ("Emerging Markets",   "2019-01-01", "2025-12-31"),
    }

    print("=" * 74)
    print("  REAL-DATA WALK-FORWARD VALIDATION")
    print("  ProbabilisticForecaster vs EnsembleForecaster")
    print("=" * 74)

    # Summary table data
    summary_rows = []

    for symbol, (desc, start, end) in tickers.items():
        print(f"\n{'='*74}")
        print(f"  {symbol} - {desc}")
        print(f"{'='*74}")

        try:
            returns = fetch_returns(symbol, start, end)
        except Exception as e:
            print(f"  SKIPPED: {e}")
            continue

        ann_vol = float(np.std(returns) * np.sqrt(252))
        ex_kurt = float(kurtosis(returns))
        sk = float(skew(returns))
        print(f"  Data: {len(returns)} trading days")
        print(f"  Ann. vol: {ann_vol:.1%}  Kurtosis: {ex_kurt:.1f}  Skew: {sk:.2f}")

        # Single forecast comparison
        if len(returns) >= 750:
            single_forecast_comparison(returns, symbol)

        # Walk-forward
        print("\n  --- Walk-Forward (500d window, 21d step/horizon, 5000 paths) ---")
        for name, Cls in [("Single-engine", ProbabilisticForecaster), ("Ensemble", EnsembleForecaster)]:
            sys.stdout.write(f"    Running {name}...")
            sys.stdout.flush()
            res = walk_forward(Cls, returns, window=500, step=21, horizon=21, n_paths=5000)
            if res is None:
                print(" FAILED (not enough data)")
                continue
            print(f" done ({res['windows']} windows)")
            print(f"      90% coverage:   {res['coverage_90']:.1%}  (target ~90%)")
            print(f"      50% coverage:   {res['coverage_50']:.1%}  (target ~50%)")
            print(f"      Mean CRPS:      {res['mean_crps']:.6f}")
            print(f"      Mean vol err:   {res['mean_vol_err']:.1%}")
            print(f"      Avg fcst vol:   {res['mean_forecast_vol']:.1%}")
            print(f"      Avg real vol:   {res['mean_realised_vol']:.1%}")

            summary_rows.append({
                "Symbol": symbol,
                "Model": name,
                "Windows": res["windows"],
                "Cov90": f"{res['coverage_90']:.0%}",
                "Cov50": f"{res['coverage_50']:.0%}",
                "CRPS": f"{res['mean_crps']:.4f}",
                "VolErr": f"{res['mean_vol_err']:.1%}",
            })

    # Summary table
    if summary_rows:
        print(f"\n\n{'='*74}")
        print("  SUMMARY TABLE")
        print(f"{'='*74}")
        print(f"  {'Symbol':<6} {'Model':<14} {'Win':>4} {'Cov90':>6} {'Cov50':>6} {'CRPS':>8} {'VolErr':>7}")
        print(f"  {'-'*6} {'-'*14} {'-'*4} {'-'*6} {'-'*6} {'-'*8} {'-'*7}")
        for r in summary_rows:
            print(f"  {r['Symbol']:<6} {r['Model']:<14} {r['Windows']:>4} {r['Cov90']:>6} {r['Cov50']:>6} {r['CRPS']:>8} {r['VolErr']:>7}")

    print(f"\n{'='*74}")
    print("  DONE")
    print(f"{'='*74}")


if __name__ == "__main__":
    main()
