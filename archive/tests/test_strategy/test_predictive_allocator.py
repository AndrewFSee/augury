"""Tests for predictive allocator module."""

from __future__ import annotations

import numpy as np
import pytest

from synthquant.forecast import ForecastResult
from synthquant.strategy.predictive_allocator import (
    PredictiveAllocationResult,
    PredictiveAllocator,
)
from synthquant.strategy.predictive_signals import SignalOutput

# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #


def _make_forecast(regime_probs, current_regime=0):
    """Create a minimal ForecastResult for testing allocation logic."""
    n_paths = 10
    horizon = 5
    paths = np.ones((n_paths, horizon + 1)) * 100.0
    terminal = np.zeros(n_paths)
    from synthquant.analytics.distributions import EmpiricalDistribution

    dist = EmpiricalDistribution(terminal)
    return ForecastResult(
        paths=paths,
        terminal_prices=np.full(n_paths, 100.0),
        terminal_returns=terminal,
        distribution=dist,
        var_95=0.0,
        cvar_95=0.0,
        var_99=0.0,
        cvar_99=0.0,
        max_drawdowns=np.zeros(n_paths),
        confidence_bands={},
        current_regime=current_regime,
        regime_probabilities=np.array(regime_probs),
        regime_params={},
        horizon=horizon,
        current_price=100.0,
    )


def _calm_ohlcv(n=250, seed=42):
    """Generate calm OHLCV data."""
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0004, 0.008, n)
    close = 100.0 * np.exp(np.cumsum(returns))
    high = close * (1 + rng.uniform(0.001, 0.008, n))
    low = close * (1 - rng.uniform(0.001, 0.008, n))
    volume = rng.uniform(0.8, 1.2, n) * 1e6
    return returns, close, high, low, volume


def _crisis_ohlcv(n=250, seed=99):
    """Generate crisis OHLCV data (calm then crash)."""
    rng = np.random.default_rng(seed)
    calm = rng.normal(0.0003, 0.008, n - 50)
    crisis = rng.normal(-0.003, 0.025, 50)
    returns = np.concatenate([calm, crisis])
    close = 100.0 * np.exp(np.cumsum(returns))
    high = close * (1 + rng.uniform(0.001, 0.02, n))
    low = close * (1 - rng.uniform(0.001, 0.02, n))
    volume = np.concatenate(
        [
            rng.uniform(0.8, 1.2, n - 50) * 1e6,
            rng.uniform(1.5, 3.0, 50) * 1e6,
        ]
    )
    return returns, close, high, low, volume


# ------------------------------------------------------------------ #
# Tests
# ------------------------------------------------------------------ #


class TestPredictiveAllocator:
    def test_output_type(self):
        allocator = PredictiveAllocator()
        fc = _make_forecast([0.9, 0.1])
        returns, close, high, low, volume = _calm_ohlcv()
        result = allocator.allocate(fc, close, returns, high, low, volume)
        assert isinstance(result, PredictiveAllocationResult)

    def test_exposure_bounded(self):
        allocator = PredictiveAllocator()
        fc = _make_forecast([0.5, 0.5])
        returns, close, high, low, volume = _calm_ohlcv()
        result = allocator.allocate(fc, close, returns, high, low, volume)
        assert 0.0 <= result.exposure <= 1.0

    def test_calm_regime_high_exposure(self):
        allocator = PredictiveAllocator(calm_weight=1.0, crisis_weight=0.25)
        fc = _make_forecast([0.95, 0.05])
        returns, close, high, low, volume = _calm_ohlcv()
        result = allocator.allocate(fc, close, returns, high, low, volume)
        # With strong calm posterior, base exposure is high
        assert result.base_exposure > 0.9

    def test_crisis_regime_low_base_exposure(self):
        allocator = PredictiveAllocator(calm_weight=1.0, crisis_weight=0.25)
        fc = _make_forecast([0.05, 0.95], current_regime=1)
        returns, close, high, low, volume = _crisis_ohlcv()
        result = allocator.allocate(fc, close, returns, high, low, volume)
        # Base exposure should be near crisis_weight
        assert result.base_exposure < 0.35

    def test_predictive_adjustment_present(self):
        allocator = PredictiveAllocator(predictive_strength=0.5)
        fc = _make_forecast([0.5, 0.5])
        returns, close, high, low, volume = _calm_ohlcv()
        result = allocator.allocate(fc, close, returns, high, low, volume)
        # Adjustment may be non-zero
        assert isinstance(result.predictive_adjustment, float)

    def test_predictive_strength_zero_matches_base(self):
        """With predictive_strength=0, adjustment should be 0."""
        allocator = PredictiveAllocator(predictive_strength=0.0)
        fc = _make_forecast([0.7, 0.3])
        returns, close, high, low, volume = _calm_ohlcv()
        result = allocator.allocate(fc, close, returns, high, low, volume)
        assert result.predictive_adjustment == 0.0
        assert abs(result.exposure - result.base_exposure) < 0.01

    def test_higher_strength_larger_adjustment(self):
        fc = _make_forecast([0.5, 0.5])
        returns, close, high, low, volume = _calm_ohlcv()
        r_low = PredictiveAllocator(predictive_strength=0.1).allocate(
            fc, close, returns, high, low, volume
        )
        r_high = PredictiveAllocator(predictive_strength=1.0).allocate(
            fc, close, returns, high, low, volume
        )
        assert abs(r_high.predictive_adjustment) >= abs(r_low.predictive_adjustment) - 0.001

    def test_signal_output_included(self):
        allocator = PredictiveAllocator()
        fc = _make_forecast([0.5, 0.5])
        returns, close, high, low, volume = _calm_ohlcv()
        result = allocator.allocate(fc, close, returns, high, low, volume)
        assert isinstance(result.signal, SignalOutput)
        assert "vol_term_structure" in result.signal.components

    def test_vol_targeting(self):
        fc = _make_forecast([0.5, 0.5])
        returns, close, high, low, volume = _calm_ohlcv()
        # Very low vol target should scale down
        allocator = PredictiveAllocator(vol_target=0.01)
        result = allocator.allocate(fc, close, returns, high, low, volume)
        assert result.vol_scale < 1.0

    def test_vol_targeting_disabled(self):
        fc = _make_forecast([0.5, 0.5])
        returns, close, high, low, volume = _calm_ohlcv()
        allocator = PredictiveAllocator(vol_target=None)
        result = allocator.allocate(fc, close, returns, high, low, volume)
        assert result.vol_scale == 1.0

    def test_min_exposure_floor(self):
        allocator = PredictiveAllocator(
            crisis_weight=0.0, min_exposure=0.1, predictive_strength=1.0
        )
        fc = _make_forecast([0.0, 1.0], current_regime=1)
        returns, close, high, low, volume = _crisis_ohlcv()
        result = allocator.allocate(fc, close, returns, high, low, volume)
        assert result.exposure >= 0.1

    def test_summary(self):
        allocator = PredictiveAllocator()
        fc = _make_forecast([0.5, 0.5])
        returns, close, high, low, volume = _calm_ohlcv()
        result = allocator.allocate(fc, close, returns, high, low, volume)
        s = result.summary()
        for key in ["exposure", "base_exposure", "predictive_adj", "composite_signal", "regime"]:
            assert key in s

    def test_invalid_weights_raises(self):
        with pytest.raises(ValueError):
            PredictiveAllocator(calm_weight=0.5, crisis_weight=0.8)

    def test_works_without_optional_data(self):
        """Should work with just close + returns."""
        allocator = PredictiveAllocator()
        fc = _make_forecast([0.5, 0.5])
        returns, close, _, _, _ = _calm_ohlcv()
        result = allocator.allocate(fc, close, returns)
        assert 0.0 <= result.exposure <= 1.0

    def test_three_regime_support(self):
        allocator = PredictiveAllocator(calm_weight=1.0, crisis_weight=0.2)
        fc = _make_forecast([0.6, 0.3, 0.1], current_regime=0)
        returns, close, high, low, volume = _calm_ohlcv()
        result = allocator.allocate(fc, close, returns, high, low, volume)
        assert 0.0 <= result.exposure <= 1.0
