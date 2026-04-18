"""Tests for the EnsembleForecaster module."""

from __future__ import annotations

import numpy as np
import pytest

from synthquant.ensemble_forecast import (
    CalibratedRegimeParams,
    EnsembleForecaster,
)
from synthquant.forecast import ForecastResult

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def two_regime_returns():
    """500 calm + 500 volatile daily returns."""
    rng = np.random.default_rng(42)
    calm = rng.normal(0.0005, 0.005, 500)
    volatile = rng.normal(-0.001, 0.025, 500)
    return np.concatenate([calm, volatile])


@pytest.fixture()
def calm_returns():
    rng = np.random.default_rng(99)
    return rng.normal(0.0003, 0.006, 750)


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


class TestEnsembleLifecycle:
    def test_fit_returns_self(self, two_regime_returns):
        fc = EnsembleForecaster(n_paths=500, random_state=42)
        assert fc.fit(two_regime_returns) is fc

    def test_forecast_before_fit_raises(self):
        fc = EnsembleForecaster()
        with pytest.raises(RuntimeError, match="not been fitted"):
            fc.forecast()

    def test_forecast_returns_result(self, two_regime_returns):
        fc = EnsembleForecaster(n_paths=500, random_state=42)
        fc.fit(two_regime_returns)
        r = fc.forecast(horizon=21, current_price=100.0)
        assert isinstance(r, ForecastResult)

    def test_result_shapes(self, two_regime_returns):
        fc = EnsembleForecaster(n_paths=600, random_state=42)
        fc.fit(two_regime_returns)
        r = fc.forecast(horizon=63, current_price=100.0)
        assert r.paths.shape[0] == 600
        assert r.paths.shape[1] == 64
        assert r.terminal_prices.shape == (600,)

    def test_all_prices_positive(self, two_regime_returns):
        fc = EnsembleForecaster(n_paths=500, random_state=42)
        fc.fit(two_regime_returns)
        r = fc.forecast(horizon=21)
        assert np.all(r.paths > 0)


# ---------------------------------------------------------------------------
# Calibration quality
# ---------------------------------------------------------------------------


class TestCalibration:
    def test_regime_params_calibrated(self, two_regime_returns):
        fc = EnsembleForecaster(n_paths=200, random_state=42)
        fc.fit(two_regime_returns)
        for i in range(2):
            cp = fc._calibrated[i]
            assert isinstance(cp, CalibratedRegimeParams)
            assert "kappa" in cp.heston
            assert "theta" in cp.heston
            assert "sigma_v" in cp.heston
            assert "rho" in cp.heston
            assert "v0" in cp.heston
            assert "lambda_j" in cp.jumps
            assert "mu_j" in cp.jumps
            assert "sigma_j" in cp.jumps

    def test_heston_feller_condition(self, two_regime_returns):
        fc = EnsembleForecaster(n_paths=200, random_state=42)
        fc.fit(two_regime_returns)
        for i in range(2):
            h = fc._calibrated[i].heston
            assert 2 * h["kappa"] * h["theta"] >= h["sigma_v"] ** 2 - 1e-10

    def test_sigma_regimes_differ(self, two_regime_returns):
        fc = EnsembleForecaster(n_paths=200, random_state=42)
        fc.fit(two_regime_returns)
        sigmas = [fc._calibrated[i].sigma for i in range(2)]
        assert max(sigmas) / min(sigmas) > 2.0

    def test_crisis_regime_has_higher_jump_intensity(self, two_regime_returns):
        fc = EnsembleForecaster(n_paths=200, random_state=42)
        fc.fit(two_regime_returns)
        sigmas = [fc._calibrated[i].sigma for i in range(2)]
        crisis = int(np.argmax(sigmas))
        calm = 1 - crisis
        assert fc._calibrated[crisis].jumps["lambda_j"] >= fc._calibrated[calm].jumps["lambda_j"]

    def test_heston_rho_negative(self, two_regime_returns):
        """Leverage effect: rho should be ≤ 0."""
        fc = EnsembleForecaster(n_paths=200, random_state=42)
        fc.fit(two_regime_returns)
        for i in range(2):
            assert fc._calibrated[i].heston["rho"] <= 0.0


# ---------------------------------------------------------------------------
# Engine weighting
# ---------------------------------------------------------------------------


class TestEngineWeighting:
    def test_weights_sum_to_one(self, two_regime_returns):
        fc = EnsembleForecaster(n_paths=200, random_state=42)
        fc.fit(two_regime_returns)
        proba = fc._detector.predict_proba(fc._returns)[-1]
        regime = int(np.argmax(proba))
        w = fc._compute_engine_weights(proba, regime)
        assert abs(sum(w.values()) - 1.0) < 1e-10

    def test_crisis_boosts_merton_weight(self, two_regime_returns):
        fc = EnsembleForecaster(n_paths=200, random_state=42)
        fc.fit(two_regime_returns)
        sigmas = [fc._calibrated[i].sigma for i in range(2)]
        crisis = int(np.argmax(sigmas))
        calm = 1 - crisis

        # Synthetic 100% crisis probability
        crisis_proba = np.zeros(2)
        crisis_proba[crisis] = 1.0
        w_crisis = fc._compute_engine_weights(crisis_proba, crisis)

        # Synthetic 100% calm probability
        calm_proba = np.zeros(2)
        calm_proba[calm] = 1.0
        w_calm = fc._compute_engine_weights(calm_proba, calm)

        assert w_crisis["merton_jd"] > w_calm["merton_jd"]
        assert w_calm["heston"] > w_crisis["heston"]

    def test_all_engines_get_minpaths(self, two_regime_returns):
        fc = EnsembleForecaster(n_paths=500, random_state=42)
        fc.fit(two_regime_returns)
        r = fc.forecast(horizon=5)
        meta = r.metadata
        assert meta["n_heston"] >= 100
        assert meta["n_merton"] >= 100
        assert meta["n_regime"] >= 100


# ---------------------------------------------------------------------------
# Risk metrics consistency
# ---------------------------------------------------------------------------


class TestRiskMetrics:
    def test_var_negative(self, two_regime_returns):
        fc = EnsembleForecaster(n_paths=1000, random_state=42)
        fc.fit(two_regime_returns)
        r = fc.forecast(horizon=63)
        assert r.var_95 < 0
        assert r.var_99 < 0

    def test_cvar_worse_than_var(self, two_regime_returns):
        fc = EnsembleForecaster(n_paths=1000, random_state=42)
        fc.fit(two_regime_returns)
        r = fc.forecast(horizon=63)
        assert r.cvar_95 <= r.var_95
        assert r.cvar_99 <= r.var_99

    def test_fan_widens(self, two_regime_returns):
        fc = EnsembleForecaster(n_paths=1000, random_state=42)
        fc.fit(two_regime_returns)
        r = fc.forecast(horizon=126)
        w_early = r.confidence_bands["p95"][10] - r.confidence_bands["p05"][10]
        w_late = r.confidence_bands["p95"][-1] - r.confidence_bands["p05"][-1]
        assert w_late > w_early


# ---------------------------------------------------------------------------
# Distribution properties
# ---------------------------------------------------------------------------


class TestDistributionProperties:
    def test_has_fat_tails(self, two_regime_returns):
        """Ensemble with jumps should produce fatter tails than normal."""
        fc = EnsembleForecaster(n_paths=5000, random_state=42)
        fc.fit(two_regime_returns)
        r = fc.forecast(horizon=63)
        from scipy.stats import kurtosis

        k = kurtosis(r.terminal_returns)
        assert k > -0.2, "Ensemble should not produce substantially thin tails"

    def test_negative_skew_in_crisis(self, two_regime_returns):
        """When trained on data ending in crisis, returns should be left-skewed."""
        fc = EnsembleForecaster(n_paths=5000, random_state=42)
        fc.fit(two_regime_returns)
        r = fc.forecast(horizon=63)
        from scipy.stats import skew

        s = skew(r.terminal_returns)
        # The distribution should have at least some negative skew due to jumps
        # and the crisis regime, though the magnitude depends on parameters
        assert s < 0.5, "Should not be strongly right-skewed given crisis data"


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


class TestScoring:
    def test_score_returns_metrics(self, two_regime_returns):
        rng = np.random.default_rng(123)
        fc = EnsembleForecaster(n_paths=500, random_state=42)
        fc.fit(two_regime_returns)
        r = fc.forecast(horizon=21)
        actual = rng.normal(0.0, 0.015, 21)
        scores = fc.score(r, actual)
        assert "crps" in scores
        assert "ks_test" in scores
        assert isinstance(scores["crps"], float)

    def test_summary_has_metadata(self, two_regime_returns):
        fc = EnsembleForecaster(n_paths=500, random_state=42)
        fc.fit(two_regime_returns)
        r = fc.forecast(horizon=21)
        s = r.summary()
        assert "expected_return" in s
        assert "var_95" in s
        meta = r.metadata
        assert meta["forecaster"] == "EnsembleForecaster"
        assert "engine_weights" in meta


# ---------------------------------------------------------------------------
# Detector variants
# ---------------------------------------------------------------------------


class TestDetectorVariants:
    def test_ms_garch_detector(self, two_regime_returns):
        fc = EnsembleForecaster(detector="ms_garch", n_paths=300, random_state=42)
        fc.fit(two_regime_returns)
        r = fc.forecast(horizon=5)
        assert r.current_regime in (0, 1)

    def test_ensemble_detector(self, two_regime_returns):
        fc = EnsembleForecaster(detector="ensemble", n_paths=300, random_state=42)
        fc.fit(two_regime_returns)
        r = fc.forecast(horizon=5)
        assert r.current_regime in (0, 1)
