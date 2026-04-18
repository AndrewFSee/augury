"""Tests for delta and gamma hedging utilities."""

from __future__ import annotations

import numpy as np
import pytest

from synthquant.simulation.engines.gbm import GBMEngine
from synthquant.strategy.hedging import DeltaHedger, GammaHedger


@pytest.fixture()
def paths() -> np.ndarray:
    """10 000 GBM paths for hedging tests."""
    engine = GBMEngine()
    return engine.simulate(
        n_paths=10_000,
        n_steps=252,
        dt=1 / 252,
        S0=100.0,
        mu=0.05,
        sigma=0.20,
        random_state=42,
    )


@pytest.fixture()
def delta_hedger() -> DeltaHedger:
    return DeltaHedger()


@pytest.fixture()
def gamma_hedger() -> GammaHedger:
    return GammaHedger()


# ── DeltaHedger ───────────────────────────────────────────────────────────────


class TestDeltaHedger:
    def test_call_delta_between_zero_and_one(self, delta_hedger, paths) -> None:
        delta = delta_hedger.compute_hedge_ratio(
            paths,
            K=100.0,
            T=1.0,
            r=0.05,
            option_type="call",
        )
        assert -0.1 <= delta <= 1.1

    def test_put_delta_between_neg_one_and_zero(self, delta_hedger, paths) -> None:
        delta = delta_hedger.compute_hedge_ratio(
            paths,
            K=100.0,
            T=1.0,
            r=0.05,
            option_type="put",
        )
        assert -1.1 <= delta <= 0.1

    def test_deep_itm_call_delta_near_one(self, delta_hedger, paths) -> None:
        """Deep ITM call (low K) should have delta ≈ 1."""
        delta = delta_hedger.compute_hedge_ratio(
            paths,
            K=30.0,
            T=1.0,
            r=0.05,
            option_type="call",
        )
        assert delta > 0.8

    def test_deep_otm_call_delta_near_zero(self, delta_hedger, paths) -> None:
        """Deep OTM call (high K) should have delta ≈ 0."""
        delta = delta_hedger.compute_hedge_ratio(
            paths,
            K=300.0,
            T=1.0,
            r=0.05,
            option_type="call",
        )
        assert delta < 0.2

    def test_delta_changes_with_strike(self, delta_hedger, paths) -> None:
        """Higher strike → lower call delta."""
        d_low_k = delta_hedger.compute_hedge_ratio(
            paths,
            K=80.0,
            T=1.0,
            r=0.05,
            option_type="call",
        )
        d_high_k = delta_hedger.compute_hedge_ratio(
            paths,
            K=120.0,
            T=1.0,
            r=0.05,
            option_type="call",
        )
        assert d_low_k > d_high_k

    def test_custom_bump_size(self, delta_hedger, paths) -> None:
        delta = delta_hedger.compute_hedge_ratio(
            paths,
            K=100.0,
            T=1.0,
            r=0.05,
            bump_size=0.005,
            option_type="call",
        )
        assert isinstance(delta, float)


# ── GammaHedger ───────────────────────────────────────────────────────────────


class TestGammaHedger:
    def test_gamma_hedger_inherits_delta(self, gamma_hedger, paths) -> None:
        """GammaHedger should still compute delta via inheritance."""
        delta = gamma_hedger.compute_hedge_ratio(
            paths,
            K=100.0,
            T=1.0,
            r=0.05,
            option_type="call",
        )
        assert isinstance(delta, float)

    def test_call_gamma_positive(self, gamma_hedger, paths) -> None:
        gamma = gamma_hedger.compute_gamma(
            paths,
            K=100.0,
            T=1.0,
            r=0.05,
            option_type="call",
        )
        assert gamma > 0

    def test_put_gamma_positive(self, gamma_hedger, paths) -> None:
        """Gamma is positive for both calls and puts."""
        gamma = gamma_hedger.compute_gamma(
            paths,
            K=100.0,
            T=1.0,
            r=0.05,
            option_type="put",
        )
        assert gamma > 0

    def test_atm_gamma_greater_than_otm(self, gamma_hedger, paths) -> None:
        """ATM options have higher gamma than deep OTM."""
        gamma_atm = gamma_hedger.compute_gamma(
            paths,
            K=100.0,
            T=1.0,
            r=0.05,
            option_type="call",
        )
        gamma_otm = gamma_hedger.compute_gamma(
            paths,
            K=200.0,
            T=1.0,
            r=0.05,
            option_type="call",
        )
        assert gamma_atm > gamma_otm

    def test_gamma_returns_float(self, gamma_hedger, paths) -> None:
        gamma = gamma_hedger.compute_gamma(
            paths,
            K=100.0,
            T=1.0,
            r=0.05,
            option_type="call",
        )
        assert isinstance(gamma, float)

    def test_custom_bump_size(self, gamma_hedger, paths) -> None:
        gamma = gamma_hedger.compute_gamma(
            paths,
            K=100.0,
            T=1.0,
            r=0.05,
            bump_size=0.005,
            option_type="call",
        )
        assert isinstance(gamma, float)
        assert gamma > 0
