"""Extended tests for options.py: Greeks, ImpliedVolSurface, _bs_call, and barrier edge cases."""

from __future__ import annotations

import numpy as np
import pytest

from synthquant.analytics.options import ImpliedVolSurface, MCOptionPricer, _bs_call
from synthquant.simulation.engines.gbm import GBMEngine


@pytest.fixture()
def gbm_paths() -> np.ndarray:
    engine = GBMEngine()
    return engine.simulate(
        n_paths=5000,
        n_steps=252,
        dt=1 / 252,
        S0=100.0,
        mu=0.05,
        sigma=0.20,
        random_state=42,
    )


@pytest.fixture()
def pricer() -> MCOptionPricer:
    return MCOptionPricer()


# ── _bs_call helper ───────────────────────────────────────────────────────────


class TestBSCall:
    def test_atm_call_positive(self) -> None:
        price = _bs_call(S=100, K=100, T=1.0, r=0.05, sigma=0.2)
        assert price > 0

    def test_deep_itm_near_intrinsic(self) -> None:
        price = _bs_call(S=200, K=100, T=1.0, r=0.05, sigma=0.2)
        intrinsic = 200 - 100 * np.exp(-0.05)
        assert price >= intrinsic

    def test_zero_vol_returns_intrinsic(self) -> None:
        price = _bs_call(S=110, K=100, T=1.0, r=0.0, sigma=0.0)
        assert np.isclose(price, 10.0, atol=0.01)

    def test_zero_time_returns_intrinsic(self) -> None:
        price = _bs_call(S=110, K=100, T=0.0, r=0.05, sigma=0.2)
        assert np.isclose(price, 10.0, atol=0.01)

    def test_zero_vol_otm_returns_zero(self) -> None:
        price = _bs_call(S=90, K=100, T=1.0, r=0.0, sigma=0.0)
        assert price == 0.0

    def test_increasing_vol_increases_price(self) -> None:
        p1 = _bs_call(S=100, K=100, T=1.0, r=0.05, sigma=0.1)
        p2 = _bs_call(S=100, K=100, T=1.0, r=0.05, sigma=0.4)
        assert p2 > p1


# ── compute_greeks ────────────────────────────────────────────────────────────


class TestComputeGreeks:
    def test_returns_all_keys(self, pricer, gbm_paths) -> None:
        greeks = pricer.compute_greeks(gbm_paths, K=100.0, T=1.0, r=0.05)
        assert "delta" in greeks
        assert "gamma" in greeks
        assert "vega" in greeks

    def test_call_delta_reasonable(self, pricer, gbm_paths) -> None:
        greeks = pricer.compute_greeks(gbm_paths, K=100.0, T=1.0, r=0.05, option_type="call")
        assert -0.1 <= greeks["delta"] <= 1.1

    def test_put_delta_reasonable(self, pricer, gbm_paths) -> None:
        greeks = pricer.compute_greeks(gbm_paths, K=100.0, T=1.0, r=0.05, option_type="put")
        assert -1.1 <= greeks["delta"] <= 0.1

    def test_gamma_positive_for_call(self, pricer, gbm_paths) -> None:
        greeks = pricer.compute_greeks(gbm_paths, K=100.0, T=1.0, r=0.05, option_type="call")
        assert greeks["gamma"] > 0

    def test_vega_positive_for_call(self, pricer, gbm_paths) -> None:
        greeks = pricer.compute_greeks(gbm_paths, K=100.0, T=1.0, r=0.05, option_type="call")
        assert greeks["vega"] >= 0

    def test_custom_bump_size(self, pricer, gbm_paths) -> None:
        greeks = pricer.compute_greeks(
            gbm_paths,
            K=100.0,
            T=1.0,
            r=0.05,
            bump_size=0.005,
            option_type="call",
        )
        assert isinstance(greeks["delta"], float)


# ── Barrier edge cases ────────────────────────────────────────────────────────


class TestBarrierEdge:
    def test_invalid_barrier_type_raises(self, pricer, gbm_paths) -> None:
        with pytest.raises(ValueError, match="barrier_type"):
            pricer.price_barrier(
                gbm_paths,
                K=100.0,
                B=120.0,
                T=1.0,
                r=0.0,
                barrier_type="knock_sideways",
            )

    def test_knockin_plus_knockout_equals_vanilla(self, pricer, gbm_paths) -> None:
        """knock_in + knock_out ≈ vanilla European (up barrier)."""
        K, B, T, r = 100.0, 130.0, 1.0, 0.0
        ko = pricer.price_barrier(gbm_paths, K=K, B=B, T=T, r=r, barrier_type="knock_out")
        ki = pricer.price_barrier(gbm_paths, K=K, B=B, T=T, r=r, barrier_type="knock_in")
        vanilla = pricer.price_european(gbm_paths, K=K, T=T, r=r)
        np.testing.assert_allclose(ko + ki, vanilla, atol=0.5)

    def test_down_barrier_put(self, pricer) -> None:
        """Down barrier (B < S0) for put."""
        engine = GBMEngine()
        paths = engine.simulate(
            n_paths=5000,
            n_steps=252,
            dt=1 / 252,
            S0=100.0,
            mu=0.0,
            sigma=0.2,
            random_state=7,
        )
        price = pricer.price_barrier(
            paths,
            K=100.0,
            B=80.0,
            T=1.0,
            r=0.0,
            barrier_type="knock_out",
            option_type="put",
        )
        assert price >= 0.0


# ── ImpliedVolSurface ─────────────────────────────────────────────────────────


class TestImpliedVolSurface:
    def test_output_shape(self) -> None:
        iv = ImpliedVolSurface()
        strikes = np.array([90.0, 100.0, 110.0])
        maturities = np.array([0.25, 0.5])
        # Generate plausible BS prices
        prices = np.array([[_bs_call(100, K, T, 0.05, 0.2) for K in strikes] for T in maturities])
        surface = iv.compute(
            S=100.0, strikes=strikes, maturities=maturities, option_prices=prices, r=0.05
        )
        assert surface.shape == (2, 3)

    def test_recovers_input_vol(self) -> None:
        """If prices come from σ=0.2, implied vol should ≈ 0.2."""
        iv = ImpliedVolSurface()
        strikes = np.array([95.0, 100.0, 105.0])
        maturities = np.array([0.5, 1.0])
        true_sigma = 0.20
        prices = np.array(
            [[_bs_call(100, K, T, 0.05, true_sigma) for K in strikes] for T in maturities]
        )
        surface = iv.compute(
            S=100.0, strikes=strikes, maturities=maturities, option_prices=prices, r=0.05
        )
        # All non-NaN entries should be close to 0.2
        valid = surface[~np.isnan(surface)]
        np.testing.assert_allclose(valid, true_sigma, atol=1e-4)

    def test_below_intrinsic_returns_nan(self) -> None:
        """If market price <= intrinsic, IV should be NaN."""
        iv = ImpliedVolSurface()
        strikes = np.array([100.0])
        maturities = np.array([1.0])
        # Price below intrinsic: S - K*exp(-rT) = 100 - 100*exp(-0.05) ≈ 4.88
        prices = np.array([[0.01]])
        surface = iv.compute(
            S=100.0, strikes=strikes, maturities=maturities, option_prices=prices, r=0.05
        )
        assert np.isnan(surface[0, 0])
