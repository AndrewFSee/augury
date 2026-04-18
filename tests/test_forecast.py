"""Tests for the ProbabilisticForecaster module."""

from __future__ import annotations

import numpy as np
import pytest

from synthquant.forecast import ForecastResult, ProbabilisticForecaster

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def two_regime_returns():
    """Synthetic 2-regime return series: 500 calm + 500 volatile days."""
    rng = np.random.default_rng(42)
    calm = rng.normal(0.0005, 0.005, 500)
    volatile = rng.normal(-0.001, 0.025, 500)
    return np.concatenate([calm, volatile])


@pytest.fixture()
def calm_returns():
    """Low-volatility returns only, for testing forecasts in a calm regime."""
    rng = np.random.default_rng(99)
    return rng.normal(0.0003, 0.006, 750)


# ---------------------------------------------------------------------------
# Basic lifecycle
# ---------------------------------------------------------------------------


class TestForecasterLifecycle:
    """Fit → forecast → score roundtrip."""

    def test_fit_returns_self(self, two_regime_returns):
        fc = ProbabilisticForecaster(n_paths=500, random_state=42)
        result = fc.fit(two_regime_returns)
        assert result is fc

    def test_forecast_before_fit_raises(self):
        fc = ProbabilisticForecaster()
        with pytest.raises(RuntimeError, match="not been fitted"):
            fc.forecast()

    def test_forecast_returns_result(self, two_regime_returns):
        fc = ProbabilisticForecaster(n_paths=500, random_state=42)
        fc.fit(two_regime_returns)
        result = fc.forecast(horizon=21, current_price=100.0)
        assert isinstance(result, ForecastResult)

    def test_result_shapes(self, two_regime_returns):
        fc = ProbabilisticForecaster(n_paths=500, random_state=42)
        fc.fit(two_regime_returns)
        r = fc.forecast(horizon=63, current_price=100.0)
        assert r.paths.shape == (500, 64)
        assert r.terminal_prices.shape == (500,)
        assert r.terminal_returns.shape == (500,)
        assert r.max_drawdowns.shape == (500,)

    def test_result_all_prices_positive(self, two_regime_returns):
        fc = ProbabilisticForecaster(n_paths=500, random_state=42)
        fc.fit(two_regime_returns)
        r = fc.forecast(horizon=21)
        assert np.all(r.paths > 0)


# ---------------------------------------------------------------------------
# Regime detection integration
# ---------------------------------------------------------------------------


class TestRegimeDetection:
    """Forecaster correctly identifies regimes from data."""

    def test_detects_two_regimes(self, two_regime_returns):
        fc = ProbabilisticForecaster(n_paths=200, random_state=42)
        fc.fit(two_regime_returns)
        r = fc.forecast(horizon=5)
        assert r.current_regime in (0, 1)
        assert len(r.regime_probabilities) == 2
        assert abs(sum(r.regime_probabilities) - 1.0) < 1e-6

    def test_regime_params_have_mu_sigma(self, two_regime_returns):
        fc = ProbabilisticForecaster(n_paths=200, random_state=42)
        fc.fit(two_regime_returns)
        r = fc.forecast(horizon=5)
        for i in range(2):
            assert "mu" in r.regime_params[i]
            assert "sigma" in r.regime_params[i]

    def test_regime_sigmas_differ(self, two_regime_returns):
        fc = ProbabilisticForecaster(n_paths=200, random_state=42)
        fc.fit(two_regime_returns)
        r = fc.forecast(horizon=5)
        sigmas = [r.regime_params[i]["sigma"] for i in range(2)]
        assert max(sigmas) / min(sigmas) > 2.0, "Regimes should have distinct vol"

    def test_ms_garch_detector(self, two_regime_returns):
        fc = ProbabilisticForecaster(detector="ms_garch", n_paths=200, random_state=42)
        fc.fit(two_regime_returns)
        r = fc.forecast(horizon=5)
        assert r.current_regime in (0, 1)

    def test_ensemble_detector(self, two_regime_returns):
        fc = ProbabilisticForecaster(detector="ensemble", n_paths=200, random_state=42)
        fc.fit(two_regime_returns)
        r = fc.forecast(horizon=5)
        assert r.current_regime in (0, 1)


# ---------------------------------------------------------------------------
# Risk metrics correctness
# ---------------------------------------------------------------------------


class TestRiskMetrics:
    """Forecasted risk metrics are internally consistent."""

    def test_var_negative(self, two_regime_returns):
        fc = ProbabilisticForecaster(n_paths=1000, random_state=42)
        fc.fit(two_regime_returns)
        r = fc.forecast(horizon=63)
        assert r.var_95 < 0, "95% VaR should be a loss (negative)"
        assert r.var_99 < 0, "99% VaR should be a loss (negative)"

    def test_cvar_worse_than_var(self, two_regime_returns):
        fc = ProbabilisticForecaster(n_paths=1000, random_state=42)
        fc.fit(two_regime_returns)
        r = fc.forecast(horizon=63)
        assert r.cvar_95 <= r.var_95, "CVaR should be worse (more negative) than VaR"
        assert r.cvar_99 <= r.var_99

    def test_var99_worse_than_var95(self, two_regime_returns):
        fc = ProbabilisticForecaster(n_paths=1000, random_state=42)
        fc.fit(two_regime_returns)
        r = fc.forecast(horizon=63)
        assert r.var_99 <= r.var_95, "99% VaR worse than 95% VaR"

    def test_max_drawdowns_nonpositive(self, two_regime_returns):
        fc = ProbabilisticForecaster(n_paths=500, random_state=42)
        fc.fit(two_regime_returns)
        r = fc.forecast(horizon=63)
        assert np.all(r.max_drawdowns <= 0)


# ---------------------------------------------------------------------------
# Distribution properties
# ---------------------------------------------------------------------------


class TestDistributionProperties:
    """Forecast distributions have reasonable statistical properties."""

    def test_confidence_bands_ordered(self, two_regime_returns):
        fc = ProbabilisticForecaster(n_paths=1000, random_state=42)
        fc.fit(two_regime_returns)
        r = fc.forecast(horizon=63)
        # p05 <= p50 <= p95 at every timestep
        assert np.all(r.confidence_bands["p05"] <= r.confidence_bands["p50"])
        assert np.all(r.confidence_bands["p50"] <= r.confidence_bands["p95"])

    def test_fan_widens_over_time(self, two_regime_returns):
        fc = ProbabilisticForecaster(n_paths=2000, random_state=42)
        fc.fit(two_regime_returns)
        r = fc.forecast(horizon=126)
        width_early = r.confidence_bands["p95"][10] - r.confidence_bands["p05"][10]
        width_late = r.confidence_bands["p95"][-1] - r.confidence_bands["p05"][-1]
        assert width_late > width_early, "Uncertainty fan should widen over time"

    def test_prob_positive_in_01(self, two_regime_returns):
        fc = ProbabilisticForecaster(n_paths=500, random_state=42)
        fc.fit(two_regime_returns)
        r = fc.forecast(horizon=21)
        assert 0.0 <= r.prob_positive <= 1.0

    def test_calm_regime_lower_vol(self, calm_returns):
        fc = ProbabilisticForecaster(n_paths=1000, random_state=42)
        fc.fit(calm_returns)
        r = fc.forecast(horizon=63)
        # In a calm regime, return std should be moderate
        assert r.return_std < 0.30, "Calm regime should not produce extreme vol"


# ---------------------------------------------------------------------------
# Summary and scoring
# ---------------------------------------------------------------------------


class TestSummaryAndScoring:
    """ForecastResult.summary() and scoring work correctly."""

    def test_summary_keys(self, two_regime_returns):
        fc = ProbabilisticForecaster(n_paths=500, random_state=42)
        fc.fit(two_regime_returns)
        r = fc.forecast(horizon=21)
        s = r.summary()
        required = [
            "horizon_steps",
            "current_price",
            "current_regime",
            "n_paths",
            "expected_return",
            "var_95",
            "cvar_95",
            "prob_positive",
            "price_p50",
        ]
        for key in required:
            assert key in s, f"Missing summary key: {key}"

    def test_score_returns_metrics(self, two_regime_returns):
        rng = np.random.default_rng(123)
        fc = ProbabilisticForecaster(n_paths=500, random_state=42)
        fc.fit(two_regime_returns)
        r = fc.forecast(horizon=21)
        # Simulated "actual" returns for scoring
        actual = rng.normal(0.0, 0.015, 21)
        scores = fc.score(r, actual)
        assert "crps" in scores
        assert "ks_test" in scores
        assert "actual_terminal_return" in scores
        assert isinstance(scores["crps"], float)

    def test_score_coverage_flag(self, two_regime_returns):
        rng = np.random.default_rng(0)
        fc = ProbabilisticForecaster(n_paths=2000, random_state=42)
        fc.fit(two_regime_returns)
        r = fc.forecast(horizon=21)
        # A moderate realised return should fall within the 90% interval
        actual = rng.normal(0.0001, 0.010, 21)
        scores = fc.score(r, actual)
        # We check the flag exists; actual coverage is stochastic
        assert "actual_in_90pct_interval" in scores
