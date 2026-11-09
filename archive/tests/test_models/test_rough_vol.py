"""Tests for Rough Bergomi stochastic volatility model."""

from __future__ import annotations

import numpy as np
import pytest

from synthquant.models.rough_vol import RoughBergomi


@pytest.fixture()
def model() -> RoughBergomi:
    return RoughBergomi()


def test_simulate_returns_tuple(model) -> None:
    """simulate() returns (price_paths, var_paths)."""
    result = model.simulate(S0=100.0, T=1.0, n_paths=50, n_steps=20, random_state=42)
    assert isinstance(result, tuple)
    assert len(result) == 2


def test_simulate_shapes(model) -> None:
    """Both arrays have shape (n_paths, n_steps+1)."""
    S, V = model.simulate(S0=100.0, T=1.0, n_paths=100, n_steps=50, random_state=42)
    assert S.shape == (100, 51)
    assert V.shape == (100, 51)


def test_initial_price(model) -> None:
    """All price paths start at S0."""
    S0 = 150.0
    S, _ = model.simulate(S0=S0, T=1.0, n_paths=200, n_steps=20, random_state=42)
    np.testing.assert_allclose(S[:, 0], S0)


def test_initial_variance_equals_xi(model) -> None:
    """At t=0, V(0) = xi * exp(0) = xi."""
    _, V = model.simulate(S0=100.0, T=1.0, n_paths=200, n_steps=20, random_state=42)
    np.testing.assert_allclose(V[:, 0], model.xi)


def test_all_prices_positive(model) -> None:
    """Price paths are strictly positive."""
    S, _ = model.simulate(S0=100.0, T=1.0, n_paths=500, n_steps=50, random_state=42)
    assert np.all(S > 0)


def test_variance_positive(model) -> None:
    """Variance is always positive (exponential form guarantees this)."""
    _, V = model.simulate(S0=100.0, T=1.0, n_paths=500, n_steps=50, random_state=42)
    assert np.all(V > 0)


def test_reproducibility() -> None:
    """Same seed produces identical paths."""
    m = RoughBergomi()
    S1, V1 = m.simulate(S0=100.0, T=1.0, n_paths=50, n_steps=30, random_state=7)
    S2, V2 = m.simulate(S0=100.0, T=1.0, n_paths=50, n_steps=30, random_state=7)
    np.testing.assert_array_equal(S1, S2)
    np.testing.assert_array_equal(V1, V2)


def test_different_seeds_differ() -> None:
    """Different seeds produce different paths."""
    m = RoughBergomi()
    S1, _ = m.simulate(S0=100.0, T=1.0, n_paths=20, n_steps=20, random_state=1)
    S2, _ = m.simulate(S0=100.0, T=1.0, n_paths=20, n_steps=20, random_state=2)
    assert not np.array_equal(S1, S2)


def test_invalid_hurst_too_high() -> None:
    """H >= 0.5 should raise ValueError."""
    with pytest.raises(ValueError, match="Hurst"):
        RoughBergomi(H=0.5)


def test_invalid_hurst_too_low() -> None:
    """H <= 0 should raise ValueError."""
    with pytest.raises(ValueError, match="Hurst"):
        RoughBergomi(H=0.0)


def test_hurst_near_boundary() -> None:
    """H very close to 0 or 0.5 should still work."""
    m_low = RoughBergomi(H=0.01)
    S, V = m_low.simulate(S0=100.0, T=0.5, n_paths=50, n_steps=20, random_state=42)
    assert S.shape == (50, 21)
    assert np.all(S > 0)

    m_high = RoughBergomi(H=0.49)
    S, V = m_high.simulate(S0=100.0, T=0.5, n_paths=50, n_steps=20, random_state=42)
    assert S.shape == (50, 21)
    assert np.all(S > 0)


def test_lower_hurst_produces_rougher_paths() -> None:
    """Lower H produces more irregular (rougher) variance paths.

    We measure roughness by the variance of increments — rougher paths
    have higher variance in their increments relative to their level.
    """
    S_smooth, V_smooth = RoughBergomi(H=0.45, eta=1.0).simulate(
        S0=100.0, T=1.0, n_paths=1000, n_steps=252, random_state=42
    )
    S_rough, V_rough = RoughBergomi(H=0.05, eta=1.0).simulate(
        S0=100.0, T=1.0, n_paths=1000, n_steps=252, random_state=42
    )
    # Ratio of increment std to level std is higher for rougher paths
    increments_smooth = np.diff(np.log(V_smooth), axis=1)
    increments_rough = np.diff(np.log(V_rough), axis=1)
    assert np.std(increments_rough) > np.std(increments_smooth)


def test_higher_eta_increases_vol_of_vol() -> None:
    """Higher eta -> wider spread in variance paths."""
    _, V_low = RoughBergomi(eta=0.5).simulate(
        S0=100.0, T=1.0, n_paths=1000, n_steps=100, random_state=42
    )
    _, V_high = RoughBergomi(eta=3.0).simulate(
        S0=100.0, T=1.0, n_paths=1000, n_steps=100, random_state=42
    )
    assert np.std(V_high[:, -1]) > np.std(V_low[:, -1])


def test_custom_parameters_stored() -> None:
    """Constructor stores all parameters correctly."""
    m = RoughBergomi(H=0.2, xi=0.09, eta=2.5, rho=-0.8, mu=0.03)
    assert m.H == 0.2
    assert m.xi == 0.09
    assert m.eta == 2.5
    assert m.rho == -0.8
    assert m.mu == 0.03
