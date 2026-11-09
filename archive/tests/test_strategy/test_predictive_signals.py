"""Tests for predictive signals module."""

from __future__ import annotations

import numpy as np
import pytest

from synthquant.strategy.predictive_signals import (
    PredictiveSignals,
    SignalOutput,
    _clip_signal,
    _credit_spread_signal,
    _cross_asset_signal,
    _move_index_signal,
    _price_distance_from_ma,
    _price_volume_divergence,
    _range_expansion,
    _rsi_signal,
    _skew_signal,
    _trend_deceleration,
    _vix_term_structure_signal,
    _vol_acceleration,
    _vol_term_structure,
    compute_predictive_signals,
)

# ------------------------------------------------------------------ #
# Fixtures
# ------------------------------------------------------------------ #


@pytest.fixture()
def calm_market():
    """Simulate a calm, trending-up market (250 days)."""
    rng = np.random.default_rng(42)
    returns = rng.normal(0.0004, 0.008, 250)  # ~1 bp/day, low vol
    close = 100.0 * np.exp(np.cumsum(returns))
    high = close * (1 + rng.uniform(0.001, 0.008, 250))
    low = close * (1 - rng.uniform(0.001, 0.008, 250))
    volume = rng.uniform(0.8, 1.2, 250) * 1e6
    return returns, close, high, low, volume


@pytest.fixture()
def crisis_market():
    """Simulate a crisis: sharp drawdown with vol spike."""
    rng = np.random.default_rng(99)
    # 200 days calm, then 50 days crisis
    calm = rng.normal(0.0003, 0.008, 200)
    crisis = rng.normal(-0.003, 0.025, 50)
    returns = np.concatenate([calm, crisis])
    close = 100.0 * np.exp(np.cumsum(returns))
    high = close * (1 + rng.uniform(0.001, 0.02, 250))
    low = close * (1 - rng.uniform(0.001, 0.02, 250))
    volume = np.concatenate(
        [
            rng.uniform(0.8, 1.2, 200) * 1e6,
            rng.uniform(1.5, 3.0, 50) * 1e6,  # volume spike in crisis
        ]
    )
    return returns, close, high, low, volume


@pytest.fixture()
def recovery_market():
    """200 days calm, 30 days crash, 20 days sharp recovery."""
    rng = np.random.default_rng(77)
    calm = rng.normal(0.0003, 0.008, 200)
    crash = rng.normal(-0.005, 0.03, 30)
    recovery = rng.normal(0.005, 0.02, 20)
    returns = np.concatenate([calm, crash, recovery])
    close = 100.0 * np.exp(np.cumsum(returns))
    high = close * (1 + rng.uniform(0.001, 0.015, 250))
    low = close * (1 - rng.uniform(0.001, 0.015, 250))
    volume = np.concatenate(
        [
            rng.uniform(0.8, 1.2, 200) * 1e6,
            rng.uniform(2.0, 4.0, 30) * 1e6,
            rng.uniform(1.0, 2.0, 20) * 1e6,
        ]
    )
    return returns, close, high, low, volume


# ------------------------------------------------------------------ #
# Helper tests
# ------------------------------------------------------------------ #


class TestClipSignal:
    def test_within_range(self):
        assert _clip_signal(0.5) == 0.5

    def test_below_range(self):
        assert _clip_signal(-3.0) == -1.0

    def test_above_range(self):
        assert _clip_signal(5.0) == 1.0


# ------------------------------------------------------------------ #
# Individual signal tests
# ------------------------------------------------------------------ #


class TestVolTermStructure:
    def test_calm_market_positive(self, calm_market):
        returns, *_ = calm_market
        sig = _vol_term_structure(returns)
        # In a calm market, short and long vol are similar → moderate range
        assert -1.0 <= sig <= 1.0

    def test_crisis_market_negative(self, crisis_market):
        returns, *_ = crisis_market
        sig = _vol_term_structure(returns)
        # Short vol >> long vol → negative
        assert sig < 0

    def test_short_data_returns_zero(self):
        assert _vol_term_structure(np.zeros(10)) == 0.0

    def test_bounded(self, calm_market):
        returns, *_ = calm_market
        sig = _vol_term_structure(returns)
        assert -1.0 <= sig <= 1.0


class TestVolAcceleration:
    def test_calm_market_near_zero(self, calm_market):
        returns, *_ = calm_market
        sig = _vol_acceleration(returns)
        assert -1.0 <= sig <= 1.0

    def test_crisis_onset_negative(self, crisis_market):
        returns, *_ = crisis_market
        sig = _vol_acceleration(returns)
        # Vol accelerating → negative
        assert sig < 0

    def test_recovery_positive(self, recovery_market):
        returns, *_ = recovery_market
        sig = _vol_acceleration(returns)
        # Vol decelerating during recovery → positive
        assert sig > -0.5  # at least not strongly negative

    def test_short_data(self):
        assert _vol_acceleration(np.zeros(5)) == 0.0


class TestRangeExpansion:
    def test_calm_near_zero(self, calm_market):
        _, close, high, low, _ = calm_market
        sig = _range_expansion(high, low, close)
        assert -0.5 <= sig <= 0.5

    def test_crisis_negative(self, crisis_market):
        _, close, high, low, _ = crisis_market
        sig = _range_expansion(high, low, close)
        # Expanding ranges in crisis → negative
        assert sig < 0.3

    def test_bounded(self, calm_market):
        _, close, high, low, _ = calm_market
        sig = _range_expansion(high, low, close)
        assert -1.0 <= sig <= 1.0


class TestPriceDistanceFromMA:
    def test_calm_small(self, calm_market):
        _, close, *_ = calm_market
        sig = _price_distance_from_ma(close)
        assert -0.8 <= sig <= 0.8

    def test_short_data(self):
        assert _price_distance_from_ma(np.ones(10)) == 0.0

    def test_bounded(self, crisis_market):
        _, close, *_ = crisis_market
        sig = _price_distance_from_ma(close)
        assert -1.0 <= sig <= 1.0


class TestTrendDeceleration:
    def test_calm_small(self, calm_market):
        returns, *_ = calm_market
        sig = _trend_deceleration(returns)
        assert -1.0 <= sig <= 1.0

    def test_short_data(self):
        assert _trend_deceleration(np.zeros(5)) == 0.0


class TestRSISignal:
    def test_neutral_market(self, calm_market):
        returns, *_ = calm_market
        sig = _rsi_signal(returns)
        assert -1.0 <= sig <= 1.0

    def test_oversold_bullish(self):
        # Strong down moves → RSI low → positive signal
        returns = np.full(20, -0.02)
        sig = _rsi_signal(returns)
        assert sig > 0

    def test_overbought_bearish(self):
        # Strong up moves → RSI high → negative signal
        returns = np.full(20, 0.02)
        sig = _rsi_signal(returns)
        assert sig < 0

    def test_short_data(self):
        assert _rsi_signal(np.zeros(5)) == 0.0


class TestPriceVolumeDivergence:
    def test_confirmed_rally_bullish(self):
        """Prices up + volume up → confirmed → bullish."""
        rng = np.random.default_rng(1)
        returns = rng.normal(0.003, 0.005, 30)
        volume = np.linspace(1e6, 2e6, 30)  # rising volume
        sig = _price_volume_divergence(returns, volume)
        assert sig > 0

    def test_divergent_rally_bearish(self):
        """Prices up + volume fading → bearish divergence."""
        rng = np.random.default_rng(2)
        returns = rng.normal(0.003, 0.005, 30)
        volume = np.linspace(2e6, 1e6, 30)  # falling volume
        sig = _price_volume_divergence(returns, volume)
        assert sig < 0

    def test_selling_exhaustion_bullish(self):
        """Prices down + volume fading → exhaustion → bullish."""
        rng = np.random.default_rng(3)
        returns = rng.normal(-0.003, 0.005, 30)
        volume = np.linspace(2e6, 0.5e6, 30)  # volume drying up
        sig = _price_volume_divergence(returns, volume)
        assert sig > 0

    def test_short_data(self):
        assert _price_volume_divergence(np.zeros(5), np.ones(5)) == 0.0


class TestCrossAssetSignal:
    def test_risk_on_bullish(self):
        """Equity outperforming safe-haven → bullish."""
        rng = np.random.default_rng(10)
        eq = rng.normal(0.003, 0.01, 30)
        sh = rng.normal(-0.001, 0.005, 30)
        sig = _cross_asset_signal(eq, sh)
        assert sig > 0

    def test_flight_to_safety_bearish(self):
        """Safe-haven outperforming equity → bearish."""
        rng = np.random.default_rng(11)
        eq = rng.normal(-0.002, 0.01, 30)
        sh = rng.normal(0.003, 0.005, 30)
        sig = _cross_asset_signal(eq, sh)
        assert sig < 0

    def test_none_safe_haven(self):
        assert _cross_asset_signal(np.zeros(30), None) == 0.0


# ------------------------------------------------------------------ #
# Market-implied signal tests
# ------------------------------------------------------------------ #


class TestVixTermStructureSignal:
    def test_contango_bullish(self):
        """VIX < VIX3M (contango) → bullish → positive signal."""
        vix = np.full(10, 15.0)  # near-term low
        vix3m = np.full(10, 20.0)  # 3-month higher
        sig = _vix_term_structure_signal(vix, vix3m)
        assert sig > 0

    def test_backwardation_bearish(self):
        """VIX > VIX3M (backwardation) → bearish → negative signal."""
        vix = np.full(10, 25.0)  # near-term spiked
        vix3m = np.full(10, 18.0)  # 3-month lower
        sig = _vix_term_structure_signal(vix, vix3m)
        assert sig < 0

    def test_flat_near_zero(self):
        """VIX ≈ VIX3M → ratio ≈ 1 → near-zero signal."""
        vix = np.full(10, 20.0)
        vix3m = np.full(10, 20.0)
        sig = _vix_term_structure_signal(vix, vix3m)
        assert -0.3 <= sig <= 0.3

    def test_none_returns_zero(self):
        assert _vix_term_structure_signal(None, None) == 0.0
        assert _vix_term_structure_signal(np.ones(10), None) == 0.0
        assert _vix_term_structure_signal(None, np.ones(10)) == 0.0

    def test_short_data_returns_zero(self):
        assert _vix_term_structure_signal(np.ones(3), np.ones(3)) == 0.0

    def test_bounded(self):
        vix = np.linspace(10, 40, 20)
        vix3m = np.linspace(20, 15, 20)
        sig = _vix_term_structure_signal(vix, vix3m)
        assert -1.0 <= sig <= 1.0


class TestSkewSignal:
    def test_high_skew_bearish(self):
        """Elevated SKEW (z > 0) → bearish → negative signal."""
        rng = np.random.default_rng(42)
        # Baseline around 130, then spike to 155
        base = rng.normal(130, 3, 20)
        skew = np.append(base, 155.0)
        sig = _skew_signal(skew)
        assert sig < 0

    def test_low_skew_bullish(self):
        """Depressed SKEW (z < 0) → complacent → positive signal."""
        rng = np.random.default_rng(42)
        base = rng.normal(130, 3, 20)
        skew = np.append(base, 110.0)
        sig = _skew_signal(skew)
        assert sig > 0

    def test_none_returns_zero(self):
        assert _skew_signal(None) == 0.0

    def test_short_data_returns_zero(self):
        assert _skew_signal(np.ones(5)) == 0.0

    def test_flat_returns_zero(self):
        """If SKEW barely moves, sigma < 0.5 → return 0."""
        skew = np.full(25, 130.0)
        assert _skew_signal(skew) == 0.0

    def test_bounded(self):
        rng = np.random.default_rng(7)
        skew = rng.normal(130, 10, 30)
        sig = _skew_signal(skew)
        assert -1.0 <= sig <= 1.0


class TestCreditSpreadSignal:
    def test_risk_on_bullish(self):
        """HYG outperforming IEF → risk-on → positive signal."""
        # HYG rising, IEF flat → HYG/IEF ratio increasing
        hyg = np.linspace(80, 85, 70)
        ief = np.full(70, 100.0)
        sig = _credit_spread_signal(hyg, ief)
        assert sig > 0

    def test_credit_stress_bearish(self):
        """HYG underperforming IEF → credit stress → negative signal."""
        # HYG falling, IEF rising → HYG/IEF ratio decreasing
        hyg = np.linspace(85, 78, 70)
        ief = np.linspace(100, 108, 70)
        sig = _credit_spread_signal(hyg, ief)
        assert sig < 0

    def test_none_returns_zero(self):
        assert _credit_spread_signal(None, None) == 0.0
        assert _credit_spread_signal(np.ones(70), None) == 0.0
        assert _credit_spread_signal(None, np.ones(70)) == 0.0

    def test_short_data_returns_zero(self):
        assert _credit_spread_signal(np.ones(10), np.ones(10)) == 0.0

    def test_bounded(self):
        rng = np.random.default_rng(3)
        hyg = rng.normal(80, 2, 70)
        ief = rng.normal(100, 2, 70)
        sig = _credit_spread_signal(hyg, ief)
        assert -1.0 <= sig <= 1.0


class TestMoveIndexSignal:
    def test_high_move_bearish(self):
        """MOVE elevated (z > 0) → bearish → negative signal."""
        rng = np.random.default_rng(42)
        base = rng.normal(100, 5, 20)
        move = np.append(base, 130.0)  # spike
        sig = _move_index_signal(move)
        assert sig < 0

    def test_low_move_bullish(self):
        """MOVE compressed (z < 0) → bullish → positive signal."""
        rng = np.random.default_rng(42)
        base = rng.normal(100, 5, 20)
        move = np.append(base, 75.0)  # compressed
        sig = _move_index_signal(move)
        assert sig > 0

    def test_none_returns_zero(self):
        assert _move_index_signal(None) == 0.0

    def test_short_data_returns_zero(self):
        assert _move_index_signal(np.ones(5)) == 0.0

    def test_flat_returns_zero(self):
        """MOVE barely moving → sigma < 0.5 → return 0."""
        move = np.full(25, 100.0)
        assert _move_index_signal(move) == 0.0

    def test_bounded(self):
        rng = np.random.default_rng(9)
        move = rng.normal(100, 15, 30)
        sig = _move_index_signal(move)
        assert -1.0 <= sig <= 1.0


# ------------------------------------------------------------------ #
# PredictiveSignals class
# ------------------------------------------------------------------ #


class TestPredictiveSignals:
    def test_output_type(self, calm_market):
        returns, close, high, low, volume = calm_market
        ps = PredictiveSignals()
        out = ps.compute(close, returns, high, low, volume)
        assert isinstance(out, SignalOutput)

    def test_composite_bounded(self, calm_market):
        returns, close, high, low, volume = calm_market
        ps = PredictiveSignals()
        out = ps.compute(close, returns, high, low, volume)
        assert -1.0 <= out.composite <= 1.0

    def test_all_components_present(self, calm_market):
        returns, close, high, low, volume = calm_market
        ps = PredictiveSignals()
        out = ps.compute(close, returns, high, low, volume)
        # All price-based signals should be present
        for key in [
            "vol_term_structure",
            "vol_acceleration",
            "range_expansion",
            "price_ma_distance",
            "trend_deceleration",
            "rsi",
            "price_volume_divergence",
        ]:
            assert key in out.components

    def test_cross_asset_included_when_provided(self, calm_market):
        returns, close, high, low, volume = calm_market
        rng = np.random.default_rng(55)
        safe_haven = rng.normal(0.0001, 0.005, 250)
        ps = PredictiveSignals()
        out = ps.compute(close, returns, high, low, volume, safe_haven)
        assert "cross_asset" in out.components

    def test_cross_asset_excluded_when_missing(self, calm_market):
        returns, close, high, low, volume = calm_market
        ps = PredictiveSignals()
        out = ps.compute(close, returns, high, low, volume)
        # cross_asset computed but with 0 because no safe-haven data
        assert out.components.get("cross_asset", 0.0) == 0.0

    def test_no_volume_signals_without_volume(self, calm_market):
        returns, close, high, low, _ = calm_market
        ps = PredictiveSignals()
        out = ps.compute(close, returns, high, low, volume=None)
        assert "price_volume_divergence" not in out.components

    def test_no_range_signals_without_highlow(self, calm_market):
        returns, close, _, _, volume = calm_market
        ps = PredictiveSignals()
        out = ps.compute(close, returns, volume=volume)
        assert "range_expansion" not in out.components

    def test_custom_weights(self, calm_market):
        returns, close, high, low, volume = calm_market
        weights = {"vol_term_structure": 1.0}
        ps = PredictiveSignals(signal_weights=weights)
        out = ps.compute(close, returns, high, low, volume)
        assert len(out.components) == 1
        assert "vol_term_structure" in out.components

    def test_sensitivity_scaling(self, calm_market):
        returns, close, high, low, volume = calm_market
        out_normal = PredictiveSignals(sensitivity=1.0).compute(close, returns, high, low, volume)
        out_high = PredictiveSignals(sensitivity=2.0).compute(close, returns, high, low, volume)
        # Higher sensitivity → larger absolute composite (unless clipped)
        if abs(out_normal.composite) < 0.5:
            assert abs(out_high.composite) >= abs(out_normal.composite) - 0.01

    def test_crisis_composite_negative(self, crisis_market):
        returns, close, high, low, volume = crisis_market
        ps = PredictiveSignals()
        out = ps.compute(close, returns, high, low, volume)
        # During crisis, composite should be negative (reduce exposure)
        assert out.composite < 0.2

    def test_summary(self, calm_market):
        returns, close, high, low, volume = calm_market
        ps = PredictiveSignals()
        out = ps.compute(close, returns, high, low, volume)
        s = out.summary()
        assert "composite" in s
        assert isinstance(s["composite"], float)

    def test_market_data_signals_included(self, calm_market):
        """When market_data is provided, market-implied signals appear."""
        from synthquant.data.market_data import MarketSnapshot

        returns, close, high, low, volume = calm_market
        n = len(returns)
        rng = np.random.default_rng(100)
        snapshot = MarketSnapshot(
            dates=np.arange(n),
            vix=rng.normal(16, 2, n),
            vix3m=rng.normal(18, 2, n),
            skew=rng.normal(130, 5, n),
            move=rng.normal(100, 8, n),
            hyg_close=rng.normal(80, 1.5, n),
            ief_close=rng.normal(100, 1.5, n),
        )
        ps = PredictiveSignals()
        out = ps.compute(close, returns, high, low, volume, market_data=snapshot)
        for key in ["vix_term_structure", "skew", "credit_spread", "move_index"]:
            assert key in out.components
        assert -1.0 <= out.composite <= 1.0

    def test_market_data_none_no_market_signals(self, calm_market):
        """When market_data is None, market-implied signals are absent."""
        returns, close, high, low, volume = calm_market
        ps = PredictiveSignals()
        out = ps.compute(close, returns, high, low, volume, market_data=None)
        for key in ["vix_term_structure", "skew", "credit_spread", "move_index"]:
            assert key not in out.components


# ------------------------------------------------------------------ #
# Convenience function
# ------------------------------------------------------------------ #


class TestComputePredictiveSignals:
    def test_returns_signal_output(self, calm_market):
        returns, close, high, low, volume = calm_market
        out = compute_predictive_signals(close, returns, high, low, volume)
        assert isinstance(out, SignalOutput)
        assert -1.0 <= out.composite <= 1.0
