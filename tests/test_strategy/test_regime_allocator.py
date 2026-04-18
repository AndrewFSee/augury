"""Tests for regime-conditional allocator."""

from __future__ import annotations

import numpy as np
import pytest

from synthquant.strategy.regime_allocator import AllocationResult, RegimeAllocator

# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


class _FakeForecast:
    """Minimal stand-in for ForecastResult."""

    def __init__(self, regime_probabilities: np.ndarray, current_regime: int) -> None:
        self.regime_probabilities = regime_probabilities
        self.current_regime = current_regime


@pytest.fixture()
def allocator() -> RegimeAllocator:
    return RegimeAllocator(calm_weight=1.0, crisis_weight=0.25)


# ------------------------------------------------------------------
# Construction
# ------------------------------------------------------------------


class TestConstruction:
    def test_default_params(self) -> None:
        a = RegimeAllocator()
        assert a.calm_weight == 1.0
        assert a.crisis_weight == 0.25
        assert a.vol_target is None

    def test_custom_params(self) -> None:
        a = RegimeAllocator(calm_weight=0.8, crisis_weight=0.1, vol_target=0.15)
        assert a.calm_weight == 0.8
        assert a.crisis_weight == 0.1
        assert a.vol_target == 0.15

    def test_invalid_weights_raises(self) -> None:
        with pytest.raises(ValueError, match="crisis_weight"):
            RegimeAllocator(calm_weight=0.5, crisis_weight=0.8)

    def test_negative_crisis_raises(self) -> None:
        with pytest.raises(ValueError):
            RegimeAllocator(crisis_weight=-0.1)


# ------------------------------------------------------------------
# Allocation logic
# ------------------------------------------------------------------


class TestAllocate:
    def test_fully_calm(self, allocator: RegimeAllocator) -> None:
        fc = _FakeForecast(np.array([1.0, 0.0]), current_regime=0)
        result = allocator.allocate(fc)
        assert isinstance(result, AllocationResult)
        assert result.exposure == pytest.approx(1.0)
        assert result.regime == 0

    def test_fully_crisis(self, allocator: RegimeAllocator) -> None:
        fc = _FakeForecast(np.array([0.0, 1.0]), current_regime=1)
        result = allocator.allocate(fc)
        assert result.exposure == pytest.approx(0.25)
        assert result.regime == 1

    def test_mixed_probabilities(self, allocator: RegimeAllocator) -> None:
        fc = _FakeForecast(np.array([0.6, 0.4]), current_regime=0)
        result = allocator.allocate(fc)
        expected = 1.0 * 0.6 + 0.25 * 0.4  # 0.70
        assert result.exposure == pytest.approx(expected)

    def test_equal_probabilities(self, allocator: RegimeAllocator) -> None:
        fc = _FakeForecast(np.array([0.5, 0.5]), current_regime=0)
        result = allocator.allocate(fc)
        expected = 1.0 * 0.5 + 0.25 * 0.5  # 0.625
        assert result.exposure == pytest.approx(expected)

    def test_three_regimes(self) -> None:
        a = RegimeAllocator(calm_weight=1.0, crisis_weight=0.0)
        fc = _FakeForecast(np.array([0.5, 0.3, 0.2]), current_regime=0)
        result = a.allocate(fc)
        # weights = linspace(1.0, 0.0, 3) = [1.0, 0.5, 0.0]
        expected = 1.0 * 0.5 + 0.5 * 0.3 + 0.0 * 0.2
        assert result.exposure == pytest.approx(expected)


# ------------------------------------------------------------------
# Vol targeting
# ------------------------------------------------------------------


class TestVolTargeting:
    def test_vol_scale_reduces_exposure(self) -> None:
        a = RegimeAllocator(calm_weight=1.0, crisis_weight=0.25, vol_target=0.15)
        fc = _FakeForecast(np.array([1.0, 0.0]), current_regime=0)
        # 30% annualised vol => scale = 0.15/0.30 = 0.5
        rng = np.random.default_rng(42)
        high_vol_returns = rng.normal(0, 0.30 / np.sqrt(252), 21)
        result = a.allocate(fc, recent_returns=high_vol_returns)
        assert result.exposure < 1.0
        assert result.vol_scale < 1.0

    def test_no_vol_target_ignores_returns(self, allocator: RegimeAllocator) -> None:
        fc = _FakeForecast(np.array([1.0, 0.0]), current_regime=0)
        rng = np.random.default_rng(42)
        rets = rng.normal(0, 0.30 / np.sqrt(252), 21)
        result = allocator.allocate(fc, recent_returns=rets)
        assert result.exposure == pytest.approx(1.0)
        assert result.vol_scale == pytest.approx(1.0)

    def test_low_vol_no_boost(self) -> None:
        """Vol scale never exceeds 1.0 (no leverage)."""
        a = RegimeAllocator(calm_weight=1.0, crisis_weight=0.25, vol_target=0.30)
        fc = _FakeForecast(np.array([1.0, 0.0]), current_regime=0)
        rng = np.random.default_rng(42)
        low_vol_returns = rng.normal(0, 0.05 / np.sqrt(252), 21)
        result = a.allocate(fc, recent_returns=low_vol_returns)
        assert result.exposure <= 1.0
        assert result.vol_scale == pytest.approx(1.0)


# ------------------------------------------------------------------
# Min exposure floor
# ------------------------------------------------------------------


class TestMinExposure:
    def test_floor_enforced(self) -> None:
        a = RegimeAllocator(calm_weight=1.0, crisis_weight=0.0, min_exposure=0.1)
        fc = _FakeForecast(np.array([0.0, 1.0]), current_regime=1)
        result = a.allocate(fc)
        assert result.exposure == pytest.approx(0.1)


# ------------------------------------------------------------------
# Result structure
# ------------------------------------------------------------------


class TestAllocationResult:
    def test_summary_keys(self, allocator: RegimeAllocator) -> None:
        fc = _FakeForecast(np.array([0.7, 0.3]), current_regime=0)
        result = allocator.allocate(fc)
        s = result.summary()
        assert "exposure" in s
        assert "regime" in s
        assert "regime_confidence" in s
        assert "vol_scale" in s
        assert "momentum_boost" in s


# ------------------------------------------------------------------
# Momentum re-entry
# ------------------------------------------------------------------


class TestMomentumReentry:
    """Verify asymmetric re-entry behaviour."""

    def test_positive_momentum_boosts_exposure(self) -> None:
        """Crisis regime + positive trailing returns → higher exposure."""
        a = RegimeAllocator(
            calm_weight=1.0,
            crisis_weight=0.25,
            reentry_lookback=20,
            reentry_scale=0.04,
        )
        fc = _FakeForecast(np.array([0.0, 1.0]), current_regime=1)
        # Trailing 20-day return of +4% → recovery = 1.0, closes full gap
        rets = np.full(20, 0.04 / 20)
        result = a.allocate(fc, recent_returns=rets)
        # Full gap closed: exposure should be calm_weight
        assert result.exposure == pytest.approx(1.0)
        assert result.momentum_boost == pytest.approx(0.75)

    def test_partial_recovery(self) -> None:
        """Half the scale → closes half the gap."""
        a = RegimeAllocator(
            calm_weight=1.0,
            crisis_weight=0.25,
            reentry_lookback=20,
            reentry_scale=0.04,
        )
        fc = _FakeForecast(np.array([0.0, 1.0]), current_regime=1)
        # 2% trailing return → recovery = 0.5 → gap = 0.75 → boost = 0.375
        rets = np.full(20, 0.02 / 20)
        result = a.allocate(fc, recent_returns=rets)
        expected = 0.25 + 0.5 * 0.75  # 0.625
        assert result.exposure == pytest.approx(expected)
        assert result.momentum_boost == pytest.approx(0.375)

    def test_negative_momentum_no_boost(self) -> None:
        """Crisis + falling market → no boost, stays at crisis_weight."""
        a = RegimeAllocator(
            calm_weight=1.0,
            crisis_weight=0.25,
            reentry_lookback=20,
            reentry_scale=0.04,
        )
        fc = _FakeForecast(np.array([0.0, 1.0]), current_regime=1)
        rets = np.full(20, -0.05 / 20)
        result = a.allocate(fc, recent_returns=rets)
        assert result.exposure == pytest.approx(0.25)
        assert result.momentum_boost == pytest.approx(0.0)

    def test_calm_regime_no_boost(self) -> None:
        """Fully calm → exposure == calm_weight → no boost needed."""
        a = RegimeAllocator(
            calm_weight=1.0,
            crisis_weight=0.25,
            reentry_lookback=20,
            reentry_scale=0.04,
        )
        fc = _FakeForecast(np.array([1.0, 0.0]), current_regime=0)
        rets = np.full(20, 0.05 / 20)
        result = a.allocate(fc, recent_returns=rets)
        assert result.exposure == pytest.approx(1.0)
        assert result.momentum_boost == pytest.approx(0.0)

    def test_disabled_when_lookback_zero(self) -> None:
        """reentry_lookback=0 disables momentum re-entry."""
        a = RegimeAllocator(
            calm_weight=1.0,
            crisis_weight=0.25,
            reentry_lookback=0,
            reentry_scale=0.04,
        )
        fc = _FakeForecast(np.array([0.0, 1.0]), current_regime=1)
        rets = np.full(20, 0.10 / 20)  # strong positive momentum
        result = a.allocate(fc, recent_returns=rets)
        assert result.exposure == pytest.approx(0.25)
        assert result.momentum_boost == pytest.approx(0.0)

    def test_no_recent_returns_no_boost(self) -> None:
        """No returns provided → no boost regardless of settings."""
        a = RegimeAllocator(
            calm_weight=1.0,
            crisis_weight=0.25,
            reentry_lookback=20,
            reentry_scale=0.04,
        )
        fc = _FakeForecast(np.array([0.0, 1.0]), current_regime=1)
        result = a.allocate(fc)
        assert result.exposure == pytest.approx(0.25)
        assert result.momentum_boost == pytest.approx(0.0)

    def test_boost_capped_at_calm_weight(self) -> None:
        """Very strong momentum doesn't push above calm_weight."""
        a = RegimeAllocator(
            calm_weight=1.0,
            crisis_weight=0.25,
            reentry_lookback=20,
            reentry_scale=0.04,
        )
        fc = _FakeForecast(np.array([0.0, 1.0]), current_regime=1)
        # 20% return, way above scale → recovery = 1.0, capped
        rets = np.full(20, 0.20 / 20)
        result = a.allocate(fc, recent_returns=rets)
        assert result.exposure == pytest.approx(1.0)

    def test_mixed_regime_with_boost(self) -> None:
        """Mixed regime (50/50) + positive momentum → boosted."""
        a = RegimeAllocator(
            calm_weight=1.0,
            crisis_weight=0.25,
            reentry_lookback=20,
            reentry_scale=0.04,
        )
        fc = _FakeForecast(np.array([0.5, 0.5]), current_regime=0)
        # Base exposure = 0.625, gap = 0.375
        # 4% trailing → full recovery → exposure = 1.0
        rets = np.full(20, 0.04 / 20)
        result = a.allocate(fc, recent_returns=rets)
        assert result.exposure == pytest.approx(1.0)
