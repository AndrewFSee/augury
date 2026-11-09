# SynthQuant Engine

![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-3776ab?logo=python&logoColor=white)
![License: MIT](https://img.shields.io/badge/license-MIT-22c55e)
![NumPy](https://img.shields.io/badge/numpy-%E2%89%A51.24-013243?logo=numpy&logoColor=white)
![SciPy](https://img.shields.io/badge/scipy-%E2%89%A51.10-8caae6?logo=scipy&logoColor=white)
![Code style: Ruff](https://img.shields.io/badge/code%20style-ruff-261230?logo=ruff&logoColor=d7ff64)

**SynthQuant** forecasts the full distribution of future returns on real market data: price bands, VaR and expected shortfall, and probabilities of reaching a level, at horizons from one day to several months. Every model is judged out of sample with a walk-forward harness, and the default (GJR-GARCH with skew-t innovations) is the one that won. Volatility models, Monte Carlo simulation, regime detection and risk analytics sit around that core.

<p align="center">
  <img src="docs/fan_chart.png" alt="SPY 21-day probabilistic forecast fan chart" width="100%">
</p>

<p align="center"><em>21-trading-day probabilistic price forecast for SPY from the default model, GJR-GARCH with skew-t innovations (<code>synthquant.forecast_prices</code>). Percentile bands show the full predicted distribution; the asymmetric fan comes from the leverage effect, which raises volatility after down days. Regenerate with <code>python scripts/generate_fan_chart.py</code>.</em></p>

---

## Showcase

All numbers below are produced by the examples with the code in this repository
(data through 2026-11-06). Re-run them with `python examples/<name>.py`.

### Which model forecasts best?

Every forecaster is scored out of sample with the same walk-forward harness: daily
forecasts from 2012 to 2026 on SPY, QQQ, IWM, TLT and GLD, each model re-fit every
21 days on a rolling 2-year window and evaluated on the same dates
([example 31](examples/31_model_comparison.py)). The score is CRPS, a proper scoring
rule for whole distributions; skill is the improvement over a rolling-volatility
Gaussian, averaged over the five assets.

<table>
<thead>
<tr><th>Model</th><th align="center">1 day</th><th align="center">5 days</th><th align="center">21 days</th></tr>
</thead>
<tbody>
<tr><td><b>GJR-GARCH skew-t</b> (default)</td><td align="center"><b>+3.4%</b></td><td align="center"><b>+3.1%</b></td><td align="center"><b>+3.4%</b></td></tr>
<tr><td>Filtered historical simulation</td><td align="center">+3.3%</td><td align="center">+2.9%</td><td align="center">+3.3%</td></tr>
<tr><td>GJR-GARCH normal</td><td align="center">+3.1%</td><td align="center">+2.7%</td><td align="center">+2.9%</td></tr>
<tr><td>HAR</td><td align="center">+2.6%</td><td align="center">+2.2%</td><td align="center">+2.1%</td></tr>
<tr><td>EWMA (RiskMetrics)</td><td align="center">+2.5%</td><td align="center">+1.8%</td><td align="center">+1.2%</td></tr>
<tr><td>Laplace (21-day MLE)</td><td align="center">+2.1%</td><td align="center">+1.4%</td><td align="center">+0.4%</td></tr>
<tr><td>Historical simulation</td><td align="center">+0.9%</td><td align="center">+0.8%</td><td align="center">+0.2%</td></tr>
</tbody>
</table>

The regime-switching and multi-engine Monte Carlo forecasters (now [archived](archive/README.md);
reproducible at git tag `pre-augury-archive`) are slower, so they were scored on monthly
forecast dates for SPY, alongside the default model on the same dates:

<table>
<thead>
<tr><th>Model (SPY, monthly origins)</th><th align="center">1 day</th><th align="center">5 days</th><th align="center">21 days</th></tr>
</thead>
<tbody>
<tr><td><b>GJR-GARCH skew-t</b></td><td align="center">+9.0%</td><td align="center">+5.7%</td><td align="center">+7.3%</td></tr>
<tr><td>Regime-switching HMM <i>(archived)</i></td><td align="center">+6.6%</td><td align="center">+3.6%</td><td align="center">+0.4%</td></tr>
<tr><td>4-engine ensemble <i>(archived)</i></td><td align="center">+7.6%</td><td align="center">+2.8%</td><td align="center">-7.5%</td></tr>
</tbody>
</table>

> GJR-GARCH with skew-t innovations is the best model at every horizon, which is why
> it is the default behind `forecast_prices()`. The four-engine ensemble does worse
> than the simple baseline at 21 days.

### VaR Backtesting

One-day 99% VaR on SPY, 3,229 trading days (2014–2026), from the same harness
([example 28](examples/28_var_backtesting.py)). Expected violations ≈ 32.

<table>
<thead>
<tr>
<th>Model</th>
<th align="center">Violations</th>
<th align="center">Rate</th>
<th align="center">Kupiec <i>p</i></th>
<th align="center">Christoffersen <i>p</i></th>
<th align="center">Basel zone</th>
</tr>
</thead>
<tbody>
<tr><td>Historical simulation</td><td align="center">44</td><td align="center">1.36%</td><td align="center">0.050 ✅</td><td align="center">0.000</td><td align="center"><img src="https://img.shields.io/badge/Yellow-eab308?style=flat-square" alt="Yellow"></td></tr>
<tr><td>Gaussian (rolling vol)</td><td align="center">86</td><td align="center">2.66%</td><td align="center">0.000</td><td align="center">0.000</td><td align="center"><img src="https://img.shields.io/badge/Red-ef4444?style=flat-square" alt="Red"></td></tr>
<tr><td>EWMA (RiskMetrics)</td><td align="center">78</td><td align="center">2.42%</td><td align="center">0.000</td><td align="center">0.013</td><td align="center"><img src="https://img.shields.io/badge/Red-ef4444?style=flat-square" alt="Red"></td></tr>
<tr><td>Laplace (21-day)</td><td align="center">53</td><td align="center">1.64%</td><td align="center">0.001</td><td align="center">0.002</td><td align="center"><img src="https://img.shields.io/badge/Yellow-eab308?style=flat-square" alt="Yellow"></td></tr>
<tr><td>HAR</td><td align="center">38</td><td align="center">1.18%</td><td align="center">0.326 ✅</td><td align="center">0.001</td><td align="center"><img src="https://img.shields.io/badge/Green-22c55e?style=flat-square" alt="Green"></td></tr>
<tr><td>GJR-GARCH normal</td><td align="center">61</td><td align="center">1.89%</td><td align="center">0.000</td><td align="center">0.465 ✅</td><td align="center"><img src="https://img.shields.io/badge/Red-ef4444?style=flat-square" alt="Red"></td></tr>
<tr><td>Filtered historical simulation</td><td align="center">37</td><td align="center">1.15%</td><td align="center">0.416 ✅</td><td align="center">0.445 ✅</td><td align="center"><img src="https://img.shields.io/badge/Green-22c55e?style=flat-square" alt="Green"></td></tr>
<tr><td><b>GJR-GARCH skew-t</b></td><td align="center"><b>34</b></td><td align="center">1.05%</td><td align="center"><b>0.764 ✅</b></td><td align="center"><b>0.373 ✅</b></td><td align="center"><img src="https://img.shields.io/badge/Green-22c55e?style=flat-square" alt="Green"></td></tr>
</tbody>
</table>

> Only GJR-GARCH skew-t and filtered historical simulation have the right number of
> violations *and* no clustering. Kupiec tests the violation rate, Christoffersen
> whether violations bunch together (the signature of a model that reacts too slowly).

<details>
<summary><b>Crisis windows (99% VaR violations)</b></summary>
<br>

| Window | Days | Gaussian | EWMA | GJR-GARCH skew-t |
|:-------|:----:|:--------:|:----:|:----------------:|
| COVID crash (Feb 19 – Mar 23, 2020) | 24 | 11 (45.8%) | 6 (25.0%) | **1 (4.2%)** |
| 2022 bear market (Jan – Oct) | 196 | 12 (6.1%) | 4 (2.0%) | **2 (1.0%)** |
| Aug 2024 selloff (Jul 15 – Aug 15) | 24 | 2 (8.3%) | 3 (12.5%) | 3 (12.5%) |

The August 2024 volatility spike came out of a calm market and caught every
volatility model; only the slow-moving historical and 21-day Laplace estimates had a
single violation.

</details>

### Monte Carlo Option Pricing

Implied volatility from Monte Carlo prices of European calls on SPY
([example 29](examples/29_option_pricing.py)). GJR-GARCH paths start from today's
conditional volatility (38% annualised after the 2026-11-05 selloff) and are made
risk-neutral with an empirical martingale correction. The Black-Scholes column uses a
flat volatility matched to the GARCH model's own expected variance at the maturity, so
the difference between the columns is the shape of the smile, not the vol level.

<table>
<thead>
<tr>
<th colspan="5">Implied volatility smile &nbsp;·&nbsp; 63-day maturity &nbsp;·&nbsp; SPY @ $735.29</th>
</tr>
<tr>
<th align="center">Moneyness</th>
<th align="center">Black-Scholes (matched)</th>
<th align="center">GJR-GARCH skew-t</th>
<th align="center">Heston</th>
<th>Observation</th>
</tr>
</thead>
<tbody>
<tr><td align="center">85%</td><td align="center">26.3%</td><td align="center">28.9%</td><td align="center">30.9%</td><td>OTM puts: crash protection priced up</td></tr>
<tr><td align="center">90%</td><td align="center">26.3%</td><td align="center">25.8%</td><td align="center">27.5%</td><td></td></tr>
<tr><td align="center">95%</td><td align="center">26.3%</td><td align="center">23.2%</td><td align="center">23.9%</td><td></td></tr>
<tr><td align="center"><b>100%</b></td><td align="center">26.3%</td><td align="center">20.9%</td><td align="center">20.2%</td><td><b>ATM</b></td></tr>
<tr><td align="center">105%</td><td align="center">26.3%</td><td align="center">19.2%</td><td align="center">16.9%</td><td></td></tr>
<tr><td align="center">110%</td><td align="center">26.3%</td><td align="center">18.0%</td><td align="center">15.3%</td><td></td></tr>
<tr><td align="center">115%</td><td align="center">26.3%</td><td align="center">17.3%</td><td align="center">15.4%</td><td>OTM calls: cheapest</td></tr>
</tbody>
</table>

> The leverage effect (volatility rises after down moves) and skewed innovations
> produce a downward-sloping skew: 11.6 vol points between 85% and 115% strikes at
> 63 days, and the skew flattens with maturity (21 days: 17.5 points; 126 days: 7.4).

<details>
<summary><b>Greeks & exotic options</b></summary>
<br>

**Greeks** (63-day ATM call)

| | Black-Scholes (matched) | GJR-GARCH | Heston |
|:-----|:----------:|:-----:|:------:|
| Delta | 0.559 | 0.637 | 0.707 |
| Gamma | 0.00409 | 0.00533 | 0.00495 |
| Vega | 145.1 | 132.2 | 140.5 |

**Exotic options** (63-day, GJR-GARCH)

| Type | Price | vs European |
|:-----|------:|:-----------:|
| European call | $34.59 | baseline |
| Asian call (average price) | $22.90 | −33.8% |
| Down-and-out put (B = 662) | $3.96 | −85.2% vs put |
| Up-and-out call (B = 809) | $9.67 | −72.0% |

</details>

### Stress Testing & Scenario Analysis

A 60% SPY / 30% TLT / 10% GLD portfolio over 63 trading days
([example 30](examples/30_stress_testing.py)). Each asset follows its own fitted
GJR-GARCH skew-t model; the assets are linked by a Gaussian copula estimated from
their standardized residuals. Stress scenarios multiply today's volatility, and the
shock then decays at each asset's fitted persistence.

<table>
<thead>
<tr>
<th colspan="5">Portfolio risk under stress &nbsp;·&nbsp; 63-day horizon &nbsp;·&nbsp; 10,000 correlated paths</th>
</tr>
<tr>
<th>Metric</th>
<th align="center">Current</th>
<th align="center">Vol 2×</th>
<th align="center">COVID 3×</th>
<th align="center">GFC 4×</th>
</tr>
</thead>
<tbody>
<tr><td>VaR 1%</td><td align="center">−28.1%</td><td align="center">−41.5%</td><td align="center">−50.9%</td><td align="center">−55.9%</td></tr>
<tr><td>CVaR 1%</td><td align="center">−38.0%</td><td align="center">−50.3%</td><td align="center">−57.2%</td><td align="center">−61.7%</td></tr>
<tr><td>VaR 5%</td><td align="center">−13.6%</td><td align="center">−21.5%</td><td align="center">−28.6%</td><td align="center">−34.6%</td></tr>
<tr><td>Median max drawdown</td><td align="center">−6.4%</td><td align="center">−10.3%</td><td align="center">−14.2%</td><td align="center">−18.0%</td></tr>
<tr><td>Worst max drawdown (1%)</td><td align="center">−35.6%</td><td align="center">−49.9%</td><td align="center">−57.9%</td><td align="center">−63.7%</td></tr>
<tr><td>P(loss > 10%)</td><td align="center">8.7%</td><td align="center">15.5%</td><td align="center">19.9%</td><td align="center">22.9%</td></tr>
<tr><td>P(loss > 20%)</td><td align="center">2.3%</td><td align="center">5.8%</td><td align="center">9.8%</td><td align="center">13.0%</td></tr>
<tr><td>Diversification benefit</td><td align="center">13.4%</td><td align="center">14.2%</td><td align="center">15.3%</td><td align="center">18.1%</td></tr>
</tbody>
</table>

"Current" is not calm: SPY's conditional volatility is 35.5% against a long-run 18.8%.

<blockquote>
<b>Where past crises land</b> (probability of a simulated loss at least as bad over the same number of days)<br>
<b>GFC</b> (Sep–Nov 2008, 49 days, −22.6%): 1.5% under current conditions, 10.8% under GFC 4×<br>
<b>COVID</b> (Feb–Mar 2020, 24 days, −17.5%): 1.5% current, 13.6% GFC 4×<br>
<b>2022 bear</b> (Mar–Jun 2022, 56 days, −16.0%): 3.5% current, 16.3% GFC 4×
</blockquote>

---

## Architecture

<table>
<tr>
<td colspan="4" align="center"><h3>SynthQuant Engine</h3></td>
</tr>

<tr>
<td valign="top" width="25%">

**Data Layer**
- Ingest (yfinance)
- Feature Engineering
- Parquet Storage
- Market Data API
- Realized Measures

</td>
<td valign="top" width="25%">

**Regime Layer**
- HMM
- MS-GARCH
- GMM Clustering
- Ensemble Detector
- Regime Features
- Market Adjustment

</td>
<td valign="top" width="25%">

**Model Layer**
- GARCH / GJR-GARCH
- Heston Stochastic Vol
- Merton Jump-Diffusion
- HAR-RV
- Bayesian Calibration

</td>
<td valign="top" width="25%">

**Simulation Layer**
- GBM Engine
- Heston Engine
- Merton JD Engine
- Regime-Switching
- Variance Reduction
- Copula Correlation
- GPU Engine

</td>
</tr>

<tr>
<td colspan="4">

**Forecasting Layer** &mdash; `forecast_prices` (default: GJR-GARCH skew-t) · FHS · HAR · Laplace · baselines · walk-forward harness · Kupiec / Christoffersen / Basel · Diebold-Mariano

</td>
</tr>

<tr>
<td colspan="4">

**Analytics Layer** &mdash; VaR / CVaR / ES · Drawdown Distribution · Tail Ratio · MC Option Pricing · Implied Vol Surface · Greeks · CRPS / PIT / KS · Rolling Moments · Jarque-Bera · Synth Multi-Scale CRPS · Percentile Capping & EWMA

</td>
</tr>

<tr>
<td colspan="2" valign="top">

**Strategy Layer**
- Kelly Criterion · Risk Parity · CVaR-Optimal Sizing
- Regime Signals · Momentum Signals
- Delta/Gamma Hedging · Walk-Forward Backtest

</td>
<td colspan="2" valign="top">

**API / Dashboard**
- FastAPI REST + WebSocket
- Streamlit Dashboard
- Plotly Visualizations

</td>
</tr>
</table>

---

## Installation

```bash
# Core installation
pip install synthquant

# With data fetching (yfinance)
pip install "synthquant[data]"

# With plotting (matplotlib, for the fan chart script and examples)
pip install "synthquant[plot]"

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
import yfinance as yf
from synthquant import forecast_prices

close = yf.download("SPY", period="5y", auto_adjust=True)["Close"].squeeze()

# Default model: GJR-GARCH(1,1) with skew-t innovations, fit on the last 2 years
fc = forecast_prices(close, horizon=21, random_state=0)

fc.summary()        # price bands, 95% VaR / ES and P(up) at 1, 5, 10, 21 days
fc.bands()          # daily price quantiles (5/25/50/75/95%) for a fan chart
fc.value_at_risk(0.01, horizon=5)   # 5-day 99% VaR as a simple return
fc.prob_above(750, horizon=21)      # probability SPY closes above $750 in 21 days

# The default makes no directional call (zero drift). To express a view,
# e.g. a 5% equity risk premium, pass it explicitly:
fc = forecast_prices(close, horizon=21, annual_drift=0.05)
```

**Why this default?** Every model in the library was scored out of sample with the
same walk-forward harness (daily, 2012–2026, SPY/QQQ/IWM/TLT/GLD;
[example 31](examples/31_model_comparison.py)). GJR-GARCH skew-t had the best CRPS at
1, 5 and 21 days and passed the 1% VaR backtests on most assets. The regime-switching
and four-engine ensemble forecasters did worse at every horizon and have been
archived.

Compare models yourself:

```python
import numpy as np
from synthquant.evaluation import walk_forward
from synthquant.forecasters import GaussianForecaster, GJRGarchForecaster, LaplaceForecaster

returns = np.log(close).diff().dropna()
res = walk_forward(
    returns,
    [GaussianForecaster(), LaplaceForecaster(), GJRGarchForecaster("skewt")],
    horizons=(1, 5, 21),
)
res.summary()       # CRPS, skill vs baseline, Diebold-Mariano p, PIT, coverage
res.var_backtest()  # Kupiec / Christoffersen / Basel zone
```

---

## Module Overview

| Module | Description |
|--------|-------------|
| `synthquant.data` | Market data ingestion (yfinance), feature engineering, Parquet storage, realized measures (RV, BPV, semi-variance, Parkinson, Garman-Klass) |
| `synthquant.regime` | HMM, Markov-Switching GARCH, GMM clustering, ensemble detectors, regime features |
| `synthquant.models` | GARCH, Heston, Merton Jump-Diffusion, HAR-RV (standard/jump/CJ), Bayesian calibration |
| `synthquant.simulation` | GBM, Heston, Merton JD, Regime-Switching engines; variance reduction; copulas; GPU |
| `synthquant.forecasters` | **Default entry point.** `forecast_prices`, GJR-GARCH (normal / t / skew-t, the default), filtered historical simulation, HAR, Laplace, Gaussian/EWMA and historical baselines — one `fit` / `predict` interface |
| `synthquant.evaluation` | Walk-forward harness, CRPS / PIT / coverage, Kupiec / Christoffersen / Basel backtests, Diebold-Mariano test |
| `synthquant.analytics` | VaR, CVaR, drawdown, MC option pricing, implied vol surface, Greeks, forecast scoring (CRPS/PIT/KS), Synth multi-scale CRPS scoring |
| `synthquant.strategy` | Kelly criterion, risk parity, CVaR sizing, regime and momentum signals, hedging, walk-forward backtest |
| `synthquant.api` | FastAPI REST + WebSocket server |
| `dashboard` | Streamlit interactive dashboard with Plotly charts |
| `archive/` | Retired code (regime/ensemble forecasters, Rough Bergomi, forecast-driven allocators) — see [archive/README.md](archive/README.md) |

---

## Examples

| # | Example | Description |
|---|---------|-------------|
| 01 | `data_ingestion` | Fetch SPY data, compute features, store locally |
| 02 | `regime_detection` | HMM, GMM and ensemble regime detection |
| 03 | `volatility_modeling` | Fit GARCH and Heston, compare forecasts |
| 04 | `monte_carlo_simulation` | Regime-switching Monte Carlo with a fan chart summary |
| 05 | `risk_analytics` | VaR, CVaR, drawdown and Greeks from simulated paths |
| 21 | `hmm_diagnostics` | HMM regime detector accuracy and behaviour |
| 26 | `external_data_experiment` | VIX / MOVE / SKEW / credit-spread enhancements to GJR-GARCH forecasts |
| 27 | `parameter_tuning` | Grid search for the external-data enhancements in example 26 |
| 28 | `var_backtesting` | **VaR backtesting report** — 8 models through the walk-forward harness, Kupiec / Christoffersen / Basel, crisis windows |
| 29 | `option_pricing` | **MC option pricing** — European / Asian / barrier options, implied vol smile, Greeks |
| 30 | `stress_testing` | **Stress testing** — correlated GJR skew-t scenarios (Gaussian copula, decaying vol shocks), historical crisis placement |
| 31 | `model_comparison` | **Model bake-off** — every forecaster scored walk-forward (CRPS, coverage, VaR tests) on 5 ETFs |

Examples 06–25 used the archived regime and ensemble forecasters; they live in
[archive/examples](archive/examples) and run at git tag `pre-augury-archive`.

```bash
python examples/31_model_comparison.py
python examples/28_var_backtesting.py
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

### November 2026 audit

**Forecast correctness** — bugs that made forecasts wrong, each now covered by a regression test:

| Fix | Effect before the fix |
|-----|-----------------------|
| Heston ignored `rho` | No leverage skew: skew was −0.10 for any `rho`; now −1.69 / +1.69 for `rho` = ∓0.9 |
| Heston variance constant `c1` | Returns too volatile when vol-of-vol is low |
| Drift convexity applied twice | Regime forecasts too bearish by ~½σ² a year (4.9% in a 32%-vol regime) |
| "CRPS-optimal" engine weights unused | Hardcoded weights were used; the optimizer now scores the true mixture CRPS and its weights are applied |
| GARCH vol conditioning discarded or capped at 2× | Forecasts ignored current volatility, or understated it in spikes |
| `GARCHModel.simulate` started from long-run vol | Paths ignored current conditions; now conditional, seeded and vectorized |
| Rough Bergomi units and normalization | Daily variance used as annual, missing √(2H): spreads up to 5× too wide |
| Option pricing example | Real-world drift and unconditional vol (ATM delta 0.78); now risk-neutral and conditional |
| Stress testing example | Assets simulated independently from random long-run states; now correlated and conditional |
| `ForecastScorer.crps` | O(n²) memory (~800 MB at 10k samples); now O(n log n) |

**New default forecaster** — GJR-GARCH with skew-t innovations via `forecast_prices()`, chosen by out-of-sample evidence (see Showcase). The regime-switching and ensemble forecasters were archived.

**New modules** — `synthquant.forecasters` (one `fit`/`predict` interface for GJR, FHS, HAR, Laplace and baselines; correlated multi-asset simulation via `GJRGarchForecaster.simulate`) and `synthquant.evaluation` (walk-forward harness, CRPS/PIT/coverage, Kupiec/Christoffersen/Basel, Diebold-Mariano).

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
