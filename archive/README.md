# Archive

Frozen code retired during the November 2026 audit. Nothing here is packaged,
linted or tested, and the imports still use the old `synthquant` package name.

## Why it was archived

In a walk-forward comparison (daily, 2012–2026, SPY/QQQ/IWM/TLT/GLD) the
regime-switching `ProbabilisticForecaster` and the four-engine
`EnsembleForecaster` (Heston + Merton + regime-switching + Rough Bergomi) were
beaten at every horizon by GJR-GARCH with skew-t innovations, now the default
(`augury.forecast_prices`). On SPY at 21 days the ensemble was 7.5% *worse*
than a rolling-volatility Gaussian, and the HMM forecaster no better than it.

| Path | What it was |
|------|-------------|
| `synthquant/forecast.py` | `ProbabilisticForecaster`: HMM regimes + GARCH scaling + regime-switching Monte Carlo |
| `synthquant/ensemble_forecast.py` | `EnsembleForecaster`: four-engine weighted Monte Carlo, intraday paths |
| `synthquant/simulation/engines/rough_bergomi.py`, `synthquant/models/rough_vol.py` | Rough Bergomi engine and its duplicate model class |
| `synthquant/regime/market_adjuster.py` | VIX/SKEW Bayesian adjustment of regime probabilities |
| `synthquant/strategy/` | Signals, allocators and the signal tuner built on `ForecastResult` |
| `tests/`, `examples/` | Their tests and the examples (06–25) that used them |

## Running it

Everything here runs, with its tests passing, at the git tag
`pre-augury-archive`:

```bash
git checkout pre-augury-archive
```

The archived modules include the correctness fixes from the audit (Heston
`rho`, drift convexity, GARCH conditioning, optimized engine weights, Rough
Bergomi units), so the comparison above is against working code.
