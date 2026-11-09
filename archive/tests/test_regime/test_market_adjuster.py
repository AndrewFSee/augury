"""Tests for MarketRegimeAdjuster — Bayesian regime probability adjustment."""

from __future__ import annotations

import numpy as np
import pytest

from synthquant.data.market_data import MarketSnapshot
from synthquant.regime.market_adjuster import MarketRegimeAdjuster

# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #


def _make_snapshot(
    n: int = 30,
    vix_level: float = 16.0,
    vix3m_level: float = 18.0,
    skew_level: float = 130.0,
    move_level: float = 100.0,
    hyg_trend: float = 0.0,
    ief_trend: float = 0.0,
    seed: int = 42,
) -> MarketSnapshot:
    """Create a synthetic MarketSnapshot for testing."""
    rng = np.random.default_rng(seed)
    return MarketSnapshot(
        dates=np.arange(n),
        vix=rng.normal(vix_level, 1.0, n),
        vix3m=rng.normal(vix3m_level, 1.0, n),
        skew=rng.normal(skew_level, 3.0, n),
        move=rng.normal(move_level, 5.0, n),
        hyg_close=np.linspace(80, 80 + hyg_trend, n) + rng.normal(0, 0.3, n),
        ief_close=np.linspace(100, 100 + ief_trend, n) + rng.normal(0, 0.3, n),
    )


def _calm_snapshot(n: int = 70) -> MarketSnapshot:
    """VIX contango, low SKEW, low MOVE, HYG outperforming IEF."""
    return _make_snapshot(
        n=n,
        vix_level=13.0,
        vix3m_level=18.0,
        skew_level=120.0,
        move_level=80.0,
        hyg_trend=3.0,
        ief_trend=-1.0,
    )


def _crisis_snapshot(n: int = 70) -> MarketSnapshot:
    """VIX backwardation, high SKEW, high MOVE, HYG underperforming IEF."""
    return _make_snapshot(
        n=n,
        vix_level=30.0,
        vix3m_level=22.0,
        skew_level=145.0,
        move_level=140.0,
        hyg_trend=-5.0,
        ief_trend=3.0,
    )


# ------------------------------------------------------------------ #
# MarketRegimeAdjuster construction
# ------------------------------------------------------------------ #


class TestInit:
    def test_default_construction(self):
        adj = MarketRegimeAdjuster()
        assert adj.strength == 0.5
        assert "vix_term_structure" in adj.signal_weights

    def test_custom_strength(self):
        adj = MarketRegimeAdjuster(strength=0.8)
        assert adj.strength == 0.8

    def test_invalid_strength(self):
        with pytest.raises(ValueError, match="strength"):
            MarketRegimeAdjuster(strength=1.5)
        with pytest.raises(ValueError, match="strength"):
            MarketRegimeAdjuster(strength=-0.1)

    def test_custom_weights(self):
        w = {"vix_term_structure": 1.0}
        adj = MarketRegimeAdjuster(signal_weights=w)
        assert adj.signal_weights == {"vix_term_structure": 1.0}


# ------------------------------------------------------------------ #
# Crisis score computation
# ------------------------------------------------------------------ #


class TestCrisisScore:
    def test_calm_market_negative(self):
        """Calm market data → negative crisis score (calm evidence)."""
        adj = MarketRegimeAdjuster()
        score = adj.compute_crisis_score(_calm_snapshot())
        assert score < 0

    def test_crisis_market_positive(self):
        """Crisis market data → positive crisis score."""
        adj = MarketRegimeAdjuster()
        score = adj.compute_crisis_score(_crisis_snapshot())
        assert score > 0

    def test_bounded(self):
        adj = MarketRegimeAdjuster()
        for snap in [_calm_snapshot(), _crisis_snapshot()]:
            score = adj.compute_crisis_score(snap)
            assert -1.0 <= score <= 1.0

    def test_neutral_near_zero(self):
        """Default-level data → near-zero score."""
        adj = MarketRegimeAdjuster()
        snap = _make_snapshot(n=70, vix_level=18.0, vix3m_level=18.0)
        score = adj.compute_crisis_score(snap)
        # Should be relatively small
        assert -0.5 <= score <= 0.5


# ------------------------------------------------------------------ #
# Probability adjustment — 2 regime case
# ------------------------------------------------------------------ #


class TestAdjust2Regime:
    def test_no_market_data_passthrough(self):
        adj = MarketRegimeAdjuster()
        proba = np.array([0.8, 0.2])
        result = adj.adjust(proba, market_data=None)
        np.testing.assert_array_equal(result, proba)

    def test_zero_strength_passthrough(self):
        adj = MarketRegimeAdjuster(strength=0.0)
        proba = np.array([0.8, 0.2])
        snap = _crisis_snapshot()
        result = adj.adjust(proba, snap)
        np.testing.assert_array_equal(result, proba)

    def test_crisis_evidence_increases_crisis_prob(self):
        """When market says crisis, P(crisis) should increase."""
        adj = MarketRegimeAdjuster(strength=0.5)
        proba = np.array([0.85, 0.15])  # HMM says 85% calm
        snap = _crisis_snapshot()
        result = adj.adjust(proba, snap)
        assert result[1] > proba[1]  # crisis prob increased
        assert result[0] < proba[0]  # calm prob decreased

    def test_calm_evidence_decreases_crisis_prob(self):
        """When market says calm, P(crisis) should decrease."""
        adj = MarketRegimeAdjuster(strength=0.5)
        proba = np.array([0.5, 0.5])  # HMM uncertain
        snap = _calm_snapshot()
        result = adj.adjust(proba, snap)
        assert result[1] < proba[1]  # crisis prob decreased
        assert result[0] > proba[0]  # calm prob increased

    def test_sums_to_one(self):
        adj = MarketRegimeAdjuster(strength=0.8)
        for snap in [_calm_snapshot(), _crisis_snapshot()]:
            for proba in [
                np.array([0.9, 0.1]),
                np.array([0.5, 0.5]),
                np.array([0.1, 0.9]),
            ]:
                result = adj.adjust(proba, snap)
                assert abs(result.sum() - 1.0) < 1e-10

    def test_probabilities_valid(self):
        adj = MarketRegimeAdjuster(strength=1.0)
        proba = np.array([0.99, 0.01])  # extreme
        snap = _crisis_snapshot()
        result = adj.adjust(proba, snap)
        assert np.all(result >= 0)
        assert np.all(result <= 1)

    def test_higher_strength_larger_shift(self):
        """Stronger adjustment → larger probability shift."""
        proba = np.array([0.7, 0.3])
        snap = _crisis_snapshot()

        low = MarketRegimeAdjuster(strength=0.2).adjust(proba, snap)
        high = MarketRegimeAdjuster(strength=0.8).adjust(proba, snap)

        # Higher strength should shift crisis prob more
        assert high[1] > low[1]


# ------------------------------------------------------------------ #
# Multi-regime (3+ states)
# ------------------------------------------------------------------ #


class TestAdjustMultiRegime:
    def test_3_regime_sums_to_one(self):
        adj = MarketRegimeAdjuster(strength=0.5)
        proba = np.array([0.6, 0.25, 0.15])
        snap = _crisis_snapshot()
        result = adj.adjust(proba, snap)
        assert abs(result.sum() - 1.0) < 1e-10
        assert np.all(result >= 0)

    def test_3_regime_crisis_increases(self):
        adj = MarketRegimeAdjuster(strength=0.5)
        proba = np.array([0.6, 0.25, 0.15])  # last = crisis
        snap = _crisis_snapshot()
        result = adj.adjust(proba, snap)
        assert result[2] > proba[2]  # crisis increased
        assert result[0] < proba[0]  # calm decreased

    def test_middle_state_preserved(self):
        """Middle states should keep their relative proportion."""
        adj = MarketRegimeAdjuster(strength=0.5)
        proba = np.array([0.5, 0.3, 0.2])
        snap = _crisis_snapshot()
        result = adj.adjust(proba, snap)
        # Middle state (index 1) should stay at 0.3 (not shifted)
        assert abs(result[1] - 0.3) < 0.01

    def test_single_regime_passthrough(self):
        adj = MarketRegimeAdjuster(strength=0.5)
        proba = np.array([1.0])
        snap = _crisis_snapshot()
        result = adj.adjust(proba, snap)
        np.testing.assert_array_almost_equal(result, proba)
