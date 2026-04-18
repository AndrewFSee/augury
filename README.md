# SynthQuant Engine

![CI](https://github.com/AndrewFSee/synthquant/actions/workflows/ci.yml/badge.svg?branch=main)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![PyPI](https://img.shields.io/pypi/v/synthquant)

**SynthQuant** is a production-grade synthetic financial data generation and probabilistic forecasting platform. It combines regime detection, stochastic volatility models, Monte Carlo simulation, and risk analytics into a unified, extensible Python framework.

<p align="center">
  <img src="docs/fan_chart.png" alt="SPY 24-hour probabilistic forecast fan chart" width="100%">
</p>

<p align="center"><em>24-hour probabilistic price forecast for SPY — 500 MC paths at 5-minute intervals via EnsembleForecaster (Heston + Merton JD + Regime-Switching + Rough Bergomi). Percentile bands show the full predicted distribution with fat tails and negative skew from GJR-GARCH calibration.</em></p>

---

## Showcase

### VaR Backtesting — Model Progression

Walk-forward VaR backtest on SPY (2014–2026, 3,088 days). Each model is re-fit every 21 days on a rolling 504-day window.

> **GJR-skewt** is the only model that passes all regulatory and statistical tests.

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  1% VaR MODEL COMPARISON  ·  Expected violations ≈ 31 / 3,088 days        │
├──────────────────┬────────────┬──────────┬───────────┬───────────┬──────────┤
│  Model           │ Violations │   Rate   │ Kupiec p  │ Christ. p │  Basel   │
├──────────────────┼────────────┼──────────┼───────────┼───────────┼──────────┤
│  Historical      │     53     │  1.72%   │   0.000   │   0.000   │ 🟡 Yellow│
│  Normal          │     91     │  2.95%   │   0.000   │   0.000   │ 🔴 Red   │
│  Student-t       │     62     │  2.01%   │   0.000   │   0.000   │ 🔴 Red   │
│  GARCH-Normal    │     68     │  2.20%   │   0.000   │   0.689   │ 🔴 Red   │
│  GJR-skewt       │  ✅ 40     │  1.30%   │ ✅ 0.115  │ ✅ 0.547  │ 🟢 Green │
│  GJR-VIX         │     16     │  0.52%   │   0.003   │   1.000   │ 🟢 Green │
└──────────────────┴────────────┴──────────┴───────────┴───────────┴──────────┘
```

<details>
<summary><strong>Crisis Period Performance (COVID, 2022 Bear, Aug 2024)</strong></summary>

```
COVID Crash (Feb–Mar 2020, 24 days)
  Model            Violations   Rate    Worst Loss   VaR@Worst
  Normal                   12  50.0%      -10.94%      -2.49%    ← missed half the days
  GJR-skewt                 1   4.2%      -10.94%     -19.41%    ← caught the crash
  GJR-VIX                   1   4.2%      -10.94%     -17.79%

2022 Bear Market (Jan–Oct, 196 days)
  Normal                   12   6.1%       -4.35%      -2.59%
  GJR-skewt                 3   1.5%       -4.35%      -3.51%
  GJR-VIX                   1   0.5%       -4.35%      -3.89%
```

</details>

### Monte Carlo Option Pricing

European call prices and implied volatility surface across 13 strikes × 3 maturities, comparing GJR-GARCH, Heston stochastic vol, and Black-Scholes.

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  IMPLIED VOL SMILE  ·  63-day maturity  ·  SPY @ $710.67                   │
├────────────┬─────────┬─────────┬─────────┬──────────┬──────────────────────┤
│  Moneyness │   BS    │  GARCH  │ Heston  │  GCH-BS  │  Observation         │
├────────────┼─────────┼─────────┼─────────┼──────────┼──────────────────────┤
│     85%    │  13.3%  │  48.1%  │  15.9%  │ +34.8pp  │  Deep OTM put skew  │
│     90%    │  13.3%  │  39.7%  │  15.4%  │ +26.4pp  │                      │
│     95%    │  13.3%  │  32.1%  │  14.4%  │ +18.8pp  │                      │
│    100%    │  13.3%  │  25.7%  │  13.4%  │ +12.4pp  │  ATM                 │
│    105%    │  13.3%  │  20.7%  │  13.9%  │  +7.4pp  │                      │
│    110%    │  13.3%  │  17.7%  │  15.0%  │  +4.4pp  │  OTM call wing       │
│    115%    │  13.3%  │  17.1%  │  15.9%  │  +3.8pp  │  Smile uptick        │
└────────────┴─────────┴─────────┴─────────┴──────────┴──────────────────────┘
```

> **Key insight**: GJR-GARCH naturally produces the volatility skew observed in real options markets — OTM puts are expensive because asymmetric GARCH makes negative returns increase vol more than positive returns. Black-Scholes (flat vol) misses this entirely.

<details>
<summary><strong>Greeks & Exotic Options</strong></summary>

```
Greeks (3-month ATM call)
                       BS      GARCH     Heston
         Delta     0.577     0.784     0.599
         Gamma   0.00828   0.00512   0.01042
          Vega     139.1     210.4     167.1

Exotic Options (3-month, GARCH model)
  Type                            Price  vs European
  European Call                   40.10     baseline
  Asian Call (avg price)          21.11      -47.4%
  Down-Out Put (B=640)             3.19      -75.7%
  Up-Out Call (B=782)             15.90      -60.4%
```

</details>

### Stress Testing & Scenario Analysis

Portfolio stress test (SPY 60% / TLT 30% / GLD 10%) using GARCH-calibrated scenarios with decaying vol shocks over a 63-day horizon.

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  PORTFOLIO RISK UNDER STRESS  ·  63-day horizon  ·  10,000 MC paths        │
├──────────────────────┬─────────┬──────────┬──────────┬──────────────────────┤
│  Metric              │ Normal  │ Vol 2×   │ COVID 3× │ GFC 4×             │
├──────────────────────┼─────────┼──────────┼──────────┼──────────────────────┤
│  VaR 1%              │ -18.9%  │  -25.6%  │  -29.5%  │  -32.0%            │
│  CVaR 1%             │ -28.8%  │  -35.4%  │  -37.4%  │  -39.8%            │
│  Median Max DD       │  -3.9%  │   -5.5%  │   -7.2%  │   -8.8%            │
│  Worst Max DD (1pct) │ -29.7%  │  -37.0%  │  -43.1%  │  -45.8%            │
│  P(loss > 10%)       │   3.5%  │    5.9%  │    7.5%  │    9.2%            │
│  P(loss > 20%)       │   0.9%  │    1.8%  │    2.5%  │    3.5%            │
└──────────────────────┴─────────┴──────────┴──────────┴──────────────────────┘

  Historical validation:
    GFC actual (49 days):   -22.6% portfolio loss → exceeds 2× VaR scenario  ✓
    COVID actual (24 days): -17.5% portfolio loss → exceeds Normal VaR        ✓
    Diversification benefit held at ~30% even under 4× stress                 ✓
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        SynthQuant Engine                        │
├──────────────┬──────────────┬───────────────┬───────────────────┤
│  Data Layer  │ Regime Layer │  Model Layer  │ Simulation Layer  │
│              │              │               │                   │
│ • Ingest     │ • HMM        │ • GARCH       │ • GBM Engine      │
│ • Features   │ • MS-GARCH   │ • Heston      │ • Heston Engine   │
│ • Storage    │ • Clustering │ • Merton JD   │ • Merton JD Eng.  │
│   (Parquet)  │ • Ensemble   │ • Rough Vol   │ • Regime Switch.  │
│ • Market     │ • Features   │ • HAR-RV      │ • Rough Bergomi   │
│   Data API   │ • Market Adj │ • Calibration │ • Variance Reduc. │
│ • Realized   │              │               │ • Copula Corr.    │
│   Measures   │              │               │ • GPU Engine      │
├──────────────┴──────────────┴───────────────┴───────────────────┤
│                     Forecasting Layer                           │
│  • ProbabilisticForecaster  • EnsembleForecaster               │
│  • GJR-GARCH skewt          • VIX/SKEW blending               │
│  • Walk-forward scoring      • CRPS / PIT / KS diagnostics    │
│  • Intraday 5-min paths      • CRPS-optimal engine weights    │
├─────────────────────────────────────────────────────────────────┤
│                       Analytics Layer                           │
│  • VaR / CVaR / ES   • Drawdown Distribution  • Tail Ratio     │
│  • MC Option Pricing • Implied Vol Surface    • Greeks          │
│  • CRPS / PIT / KS   • Rolling Moments        • Jarque-Bera    │
│  • Synth Multi-Scale CRPS    • Percentile Capping & EWMA      │
├──────────────────────────────┬──────────────────────────────────┤
│        Strategy Layer        │          API / Dashboard         │
│  • Kelly Criterion           │  • FastAPI REST + WebSocket      │
│  • Risk Parity               │  • Streamlit Dashboard           │
│  • CVaR-Optimal Sizing       │  • Plotly Visualizations         │
│  • Regime Signals            │                                  │
│  • Forecast Signals          │                                  │
│  • Predictive Allocator      │                                  │
│  • Signal Tuner              │                                  │
│  • Delta/Gamma Hedging       │                                  │
│  • Walk-Forward Backtest     │                                  │
└──────────────────────────────┴──────────────────────────────────┘
```

---

## Installation

```bash
# Core installation
pip install synthquant

# With data fetching (yfinance)
pip install "synthquant[data]"

# With Bayesian calibration (PyMC)
pip install "synthquant[bayesian]"

# With REST API (FastAPI)
pip install "synthquant[api]"

# With dashboard (Streamlit + Plotly)
pip install "synthquant[dashboard]"

# Everything
pip install "synthquant[all]"

# Development
pip install "synthquant[dev]"
```

---

## Quickstart

```python
import numpy as np
from synthquant.simulation.engines.gbm import GBMEngine
from synthquant.regime.hmm import HMMRegimeDetector
from synthquant.analytics.risk_metrics import value_at_risk, expected_shortfall

# 1. Simulate 10,000 GBM paths over 1 year (252 steps)
engine = GBMEngine()
paths = engine.simulate(
    n_paths=10_000,
    n_steps=252,
    dt=1/252,
    S0=100.0,
    mu=0.07,
    sigma=0.20,
    random_state=42,
)
print(f"Paths shape: {paths.shape}")  # (10000, 253)

# 2. Detect regimes on historical returns
returns = np.random.normal(0, 0.01, 500)  # replace with real data
detector = HMMRegimeDetector(n_components=2, random_state=42)
detector.fit(returns)
regimes = detector.predict(returns)
params = detector.get_regime_params()
print(f"Regime 0 vol: {params[0]['sigma']:.4f}")

# 3. Compute risk metrics on terminal prices
var_95 = value_at_risk(paths, alpha=0.05)
es_95 = expected_shortfall(paths, alpha=0.05)
print(f"95% VaR: {var_95:.4f}")
print(f"95% ES:  {es_95:.4f}")
```

---

## Module Overview

| Module | Description |
|--------|-------------|
| `synthquant.data` | Market data ingestion (yfinance), feature engineering, Parquet storage, realized measures (RV, BPV, semi-variance, Parkinson, Garman-Klass) |
| `synthquant.regime` | HMM, Markov-Switching GARCH, GMM clustering, ensemble detectors, regime features, market adjustment |
| `synthquant.models` | GARCH, Heston, Merton Jump-Diffusion, Rough Bergomi, HAR-RV (standard/jump/CJ), Bayesian calibration |
| `synthquant.simulation` | GBM, Heston, Merton JD, Regime-Switching, Rough Bergomi engines; variance reduction; copulas; GPU |
| `synthquant.forecast` | ProbabilisticForecaster, EnsembleForecaster — walk-forward probabilistic density forecasting, intraday 5-min paths, CRPS-optimal engine weights |
| `synthquant.analytics` | VaR, CVaR, drawdown, MC option pricing, implied vol surface, Greeks, forecast scoring (CRPS/PIT/KS), Synth multi-scale CRPS scoring |
| `synthquant.strategy` | Kelly criterion, risk parity, CVaR sizing, regime signals, forecast signals, predictive allocator, signal tuner, hedging, walk-forward backtest |
| `synthquant.api` | FastAPI REST + WebSocket server |
| `dashboard` | Streamlit interactive dashboard with Plotly charts |

---

## Examples

### Core Pipeline

| # | Example | Description |
|---|---------|-------------|
| 01 | `data_ingestion` | Fetch SPY data, compute features, store to Parquet |
| 02 | `regime_detection` | HMM, GMM, and ensemble regime detection |
| 03 | `volatility_modeling` | Fit GARCH and Heston, compare vol forecasts |
| 04 | `monte_carlo_simulation` | Regime-switching MC simulation with fan charts |
| 05 | `risk_analytics` | VaR, CVaR, drawdown, Greeks from simulated paths |
| 06 | `full_pipeline` | End-to-end: data → regime → model → strategy → backtest |

### Forecasting & Evaluation

| # | Example | Description |
|---|---------|-------------|
| 07 | `probabilistic_forecasting` | ProbabilisticForecaster fit, forecast, walk-forward scoring |
| 08 | `ensemble_vs_single` | Walk-forward comparison: single vs ensemble forecaster |
| 09 | `real_data_validation` | Walk-forward validation on SPY, QQQ, and volatile tickers |

### Signal Construction & Tuning

| # | Example | Description |
|---|---------|-------------|
| 10 | `forecast_signals` | Composite forecast signal demo on real data |
| 11 | `signal_tuning` | Walk-forward signal weight tuning (differential evolution) |
| 12 | `default_robustness` | Default signal weight robustness across tickers/periods |

### Regime Analysis & Allocation

| # | Example | Description |
|---|---------|-------------|
| 13 | `regime_stability` | Regime detection stability: smoothed vs filtered HMM |
| 14 | `allocator_comparison` | Regime allocator vs composite signal vs buy-and-hold |
| 15 | `momentum_reentry` | Isolate momentum re-entry feature impact |
| 16 | `multivariate_hmm` | Multivariate HMM feature comparison |

### Advanced Signals & Market Data

| # | Example | Description |
|---|---------|-------------|
| 17 | `feature_selection` | HMM feature subset selection via walk-forward |
| 18 | `predictive_signals` | Predictive signals vs buy-and-hold across tickers |
| 19 | `market_implied_signals` | OHLCV vs OHLCV + options/credit data |
| 20 | `market_regime_adjustment` | Market-implied regime adjustment for forecast calibration |

### Diagnostics & Evaluation

| # | Example | Description |
|---|---------|-------------|
| 21 | `hmm_diagnostics` | Comprehensive HMM accuracy and behavior evaluation |
| 22 | `honest_evaluation` | Honest walk-forward evaluation of momentum × vol-sizing |

### Strategy Backtests

| # | Example | Description |
|---|---------|-------------|
| 23 | `forecast_as_signal` | Does MC forecast distribution add value beyond trailing indicators? |
| 24 | `vrp_strategy` | Variance risk premium strategy using probabilistic vol forecast |
| 25 | `calibration_showcase` | Probabilistic calibration showcase with 7 models |
| 25 | `multi_asset_allocation` | Multi-asset forecast-driven allocation |

### Calibration & Model Improvement

| # | Example | Description |
|---|---------|-------------|
| 26 | `external_data_experiment` | External data sources (VIX, SKEW, credit) for GJR-GARCH |
| 27 | `parameter_tuning` | Grid-search optimal parameters for external data models |

### Industry Use Cases

| # | Example | Description |
|---|---------|-------------|
| 28 | `var_backtesting` | **VaR backtesting report** — Basel III regulatory compliance, Kupiec/Christoffersen tests |
| 29 | `option_pricing` | **MC option pricing** — European/Asian/barrier options, implied vol surface, Greeks |
| 30 | `stress_testing` | **Stress testing** — Regime-conditional scenario analysis, historical validation |

```bash
# Run any example
python examples/01_data_ingestion.py
python examples/28_var_backtesting.py
python examples/30_stress_testing.py
```

---

## Configuration

All settings can be overridden via environment variables prefixed with `SYNTHQUANT_`:

```bash
SYNTHQUANT_N_PATHS=50000 \
SYNTHQUANT_RANDOM_SEED=42 \
SYNTHQUANT_LOG_LEVEL=DEBUG \
python my_script.py
```

Or via a `.env` file in the project root.

---

## Recent Changes

### Forecasting Improvements

| Feature | Description |
|---------|-------------|
| **HAR-RV Model** | Multi-scale realized volatility forecasting (standard, jump, CJ variants) based on Corsi (2009). Captures heterogeneous trader horizons at daily/weekly/monthly scales. |
| **Intraday Simulation** | `forecast_intraday()` produces 100 paths × 289 points at 5-minute intervals, matching Synth subnet's format for direct CRPS comparison. |
| **Synth CRPS Scoring** | Multi-scale CRPS at 5-min / 30-min / 3-hour / 24-hour, with percentile capping, EWMA smoothing, and softmax weighting — mirrors the Synth subnet scoring pipeline. |
| **Rough Bergomi Engine** | 4th simulation engine in the ensemble with variogram-based Hurst exponent calibration for rough volatility dynamics. |
| **CRPS-Optimal Weights** | Walk-forward CV + Nelder-Mead optimization replaces hardcoded heuristic engine weights. |
| **Realized Measures** | Full suite: RV, bipower variation, semi-variance, jump variation, realized kernel, Parkinson, and Garman-Klass estimators. |

### Bug Fixes

| Fix | Description |
|-----|-------------|
| Heston calibration type error | `.values` on `ExtensionArray` caused crash during calibration |
| `None` transition matrix | `None` passed to `RegimeSwitchingEngine` when transition matrix unavailable |
| Ensemble label alignment | `None` `best_perm` crash in ensemble label alignment |
| MS-GARCH regime params | `None` `_label_order` crash in MS-GARCH regime parameter extraction |
| Risk parity division by zero | `RiskParityAllocator` blew up on zero-variance assets |
| Broad `except Exception` | Calibration silently swallowed real errors — now catches specific exceptions |
| Black-Scholes near-zero sigma | Formula blowup when sigma approaches zero — now returns intrinsic value |

### Performance & Architecture

| Improvement | Description |
|-------------|-------------|
| **Vectorized rolling windows** | `cumsum` + `stride_tricks` replaces O(n²) Python loops in regime features |
| **Hungarian algorithm** | O(k³) label alignment for k > 5 regimes replaces O(k!) brute-force permutation |
| **`CalibrationBounds` dataclass** | Replaces 12+ scattered magic numbers with a single configurable bounds object |

### New Files

- `synthquant/models/har_rv.py` — HAR-RV model (standard, jump, CJ variants)
- `synthquant/analytics/synth_scoring.py` — Synth-compatible multi-scale CRPS scoring
- `synthquant/data/realized_measures.py` — Realized volatility estimators (RV, BPV, semi-variance, Parkinson, Garman-Klass)

---

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Install dev dependencies: `pip install -e ".[dev]"`
4. Run tests: `pytest -m "not slow and not integration"`
5. Lint: `ruff check synthquant/ tests/`
6. Submit a pull request

---

## License

MIT License — see [LICENSE](LICENSE) for details.
