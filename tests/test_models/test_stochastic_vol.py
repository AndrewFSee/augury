"""Tests for Heston stochastic volatility model."""

from __future__ import annotations

import numpy as np
import pytest

from augury.models.stochastic_vol import HestonModel


@pytest.fixture()
def model() -> HestonModel:
    """Default Heston model."""
    return HestonModel()


def test_simulate_returns_tuple_of_two_arrays(model) -> None:
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
    S, _ = model.simulate(S0=S0, T=1.0, n_paths=200, n_steps=10, random_state=42)
    np.testing.assert_allclose(S[:, 0], S0)


def test_initial_variance(model) -> None:
    """All variance paths start at v0."""
    _, V = model.simulate(S0=100.0, T=1.0, n_paths=200, n_steps=10, random_state=42)
    np.testing.assert_allclose(V[:, 0], model.v0)


def test_all_prices_positive(model) -> None:
    """Price paths are strictly positive (log-Euler scheme)."""
    S, _ = model.simulate(S0=100.0, T=1.0, n_paths=500, n_steps=50, random_state=42)
    assert np.all(S > 0)


def test_variance_non_negative(model) -> None:
    """Variance paths are clipped to be non-negative."""
    m = HestonModel(sigma_v=1.5)  # high vol-of-vol stress test
    _, V = m.simulate(S0=100.0, T=1.0, n_paths=500, n_steps=50, random_state=42)
    assert np.all(V >= 0)


def test_reproducibility() -> None:
    """Same random_state produces identical outputs."""
    m = HestonModel()
    S1, V1 = m.simulate(S0=100.0, T=1.0, n_paths=50, n_steps=30, random_state=7)
    S2, V2 = m.simulate(S0=100.0, T=1.0, n_paths=50, n_steps=30, random_state=7)
    np.testing.assert_array_equal(S1, S2)
    np.testing.assert_array_equal(V1, V2)


def test_different_seeds_differ() -> None:
    """Different seeds produce different paths."""
    m = HestonModel()
    S1, _ = m.simulate(S0=100.0, T=1.0, n_paths=20, n_steps=20, random_state=1)
    S2, _ = m.simulate(S0=100.0, T=1.0, n_paths=20, n_steps=20, random_state=2)
    assert not np.array_equal(S1, S2)


def test_mean_reversion_of_variance() -> None:
    """With fast mean-reversion, terminal variance mean -> theta."""
    m = HestonModel(v0=0.01, kappa=5.0, theta=0.04, sigma_v=0.2)
    _, V = m.simulate(S0=100.0, T=2.0, n_paths=5000, n_steps=504, random_state=42)
    mean_terminal_var = V[:, -1].mean()
    np.testing.assert_allclose(mean_terminal_var, m.theta, atol=0.01)


def test_higher_vol_of_vol_increases_variance_spread() -> None:
    """Higher sigma_v widens the distribution of terminal variance."""
    _, V_low = HestonModel(sigma_v=0.1).simulate(
        S0=100.0, T=1.0, n_paths=5000, n_steps=252, random_state=42
    )
    _, V_high = HestonModel(sigma_v=1.0).simulate(
        S0=100.0, T=1.0, n_paths=5000, n_steps=252, random_state=42
    )
    assert np.std(V_high[:, -1]) > np.std(V_low[:, -1])


def test_negative_correlation_produces_leverage_effect() -> None:
    """Negative rho: drops in price should coincide with rises in variance."""
    m = HestonModel(rho=-0.9, sigma_v=0.5, kappa=1.0)
    S, V = m.simulate(S0=100.0, T=1.0, n_paths=3000, n_steps=252, random_state=42)
    price_rets = np.diff(np.log(S), axis=1)
    var_changes = np.diff(V, axis=1)
    corr = np.corrcoef(price_rets.flatten(), var_changes.flatten())[0, 1]
    assert corr < 0, f"Expected negative correlation, got {corr:.4f}"


def test_custom_parameters_stored() -> None:
    """Constructor stores all parameters correctly."""
    m = HestonModel(v0=0.09, kappa=3.0, theta=0.06, sigma_v=0.5, rho=-0.5, mu=0.08)
    assert m.v0 == 0.09
    assert m.kappa == 3.0
    assert m.theta == 0.06
    assert m.sigma_v == 0.5
    assert m.rho == -0.5
    assert m.mu == 0.08
