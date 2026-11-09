"""Demo: ForecastSignal on real market data.

Shows the composite signal for each ticker using EnsembleForecaster.
"""
import contextlib
import io
import warnings

import numpy as np
import yfinance as yf
from scipy.stats import kurtosis, skew

from synthquant.ensemble_forecast import EnsembleForecaster
from synthquant.strategy.forecast_signals import ForecastSignal


def suppress():
    return contextlib.ExitStack()  # noop — let output through


tickers = ["SPY", "QQQ", "IWM", "XLE", "GLD", "EEM"]

print("=" * 78)
print("  FORECAST SIGNAL DEMO — EnsembleForecaster + ForecastSignal")
print("  Horizon: 21 trading days  |  Paths: 10,000")
print("  Now with momentum component (short=21d, long=63d)")
print("=" * 78)

signal_gen = ForecastSignal()
signal_gen_short = ForecastSignal(allow_short=True)

results = []

for symbol in tickers:
    print(f"\n{'─'*78}")
    print(f"  {symbol}")
    print(f"{'─'*78}")

    hist = yf.Ticker(symbol).history(start="2020-01-01", end="2026-12-31")
    prices = hist["Close"].values
    returns = np.diff(np.log(prices))
    returns = returns[np.isfinite(returns)]
    last_price = float(prices[-1])

    print(f"  Data:  {len(returns)} days  |  Last price: ${last_price:.2f}")
    print(f"  Vol:   {np.std(returns)*np.sqrt(252):.1%}  |  Kurt: {kurtosis(returns):.1f}  |  Skew: {skew(returns):.2f}")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with contextlib.redirect_stderr(io.StringIO()):
            fc = EnsembleForecaster(n_paths=10_000, random_state=42)
            fc.fit(returns[-750:])
            forecast = fc.forecast(horizon=21, current_price=last_price)

    sig = signal_gen.generate(forecast, recent_returns=returns[-63:])
    sig_s = signal_gen_short.generate(forecast, recent_returns=returns[-63:])

    signal_label = {1: "LONG", 0: "FLAT", -1: "SHORT"}
    print("\n  Forecast:")
    print(f"    E[return]:     {forecast.expected_return:+.2%}")
    print(f"    VaR 95%:       {forecast.var_95:+.2%}")
    print(f"    CVaR 95%:      {forecast.cvar_95:+.2%}")
    print(f"    P(profit):     {forecast.prob_positive:.1%}")
    print(f"    P(loss>10%):   {forecast.prob_loss_gt_10pct:.1%}")
    print(f"    Skew:          {skew(forecast.terminal_returns):.3f}")
    print(f"    Regime:        {forecast.current_regime} (conf={float(forecast.regime_probabilities[forecast.current_regime]):.1%})")

    if "engine_weights" in forecast.metadata:
        w = forecast.metadata["engine_weights"]
        print(f"    Engine wts:    H={w['heston']:.0%} M={w['merton_jd']:.0%} R={w['regime_switching']:.0%}")

    print("\n  Signal (long-only):")
    print(f"    Direction:     {signal_label[sig.signal]}")
    print(f"    Conviction:    {sig.conviction:.2f}")
    print("    Components:")
    for name, val in sig.components.items():
        bar_len = int(abs(val) * 20)
        bar = ("+" if val >= 0 else "-") * bar_len
        print(f"      {name:<12} {val:+.3f}  |{bar}|")

    if sig_s.signal == -1:
        print(f"  Signal (short OK): SHORT  conviction={sig_s.conviction:.2f}")

    results.append({
        "symbol": symbol,
        "signal": signal_label[sig.signal],
        "conviction": sig.conviction,
        "regime": forecast.current_regime,
        "e_ret": forecast.expected_return,
        "p_profit": forecast.prob_positive,
        "skew": skew(forecast.terminal_returns),
    })

# Summary
print(f"\n\n{'='*78}")
print("  SUMMARY")
print(f"{'='*78}")
print(f"  {'Symbol':<6} {'Signal':<6} {'Conv':>5} {'Rgm':>4} {'E[ret]':>8} {'P(+)':>6} {'Skew':>7}")
print(f"  {'─'*6} {'─'*6} {'─'*5} {'─'*4} {'─'*8} {'─'*6} {'─'*7}")
for r in results:
    print(f"  {r['symbol']:<6} {r['signal']:<6} {r['conviction']:>5.2f} {r['regime']:>4} {r['e_ret']:>+8.2%} {r['p_profit']:>5.1%} {r['skew']:>+7.3f}")

print(f"\n{'='*78}")
print("  DONE")
print(f"{'='*78}")
