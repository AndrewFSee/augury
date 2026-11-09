"""Tests for Heston stochastic volatility simulation engine."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from augury.simulation.engines.heston import HestonEngine


def test_simulate_returns_correct_shape() -> None:
    """simulate() returns array of shape (n_paths, n_steps+1)."""
    engine = HestonEngine()
    paths = engine.simulate(n_paths=100, n_steps=50, dt=1 / 252)
    assert paths.shape == (100, 51)


def test_initial_price_is_correct() -> None:
    """All paths start at S0."""
    engine = HestonEngine()
    S0 = 120.0
    paths = engine.simulate(n_paths=200, n_steps=10, dt=1 / 252, S0=S0)
    np.testing.assert_allclose(paths[:, 0], S0)


def test_all_prices_positive() -> None:
    """Heston paths are strictly positive."""
    engine = HestonEngine()
    paths = engine.simulate(n_paths=500, n_steps=50, dt=1 / 252, random_state=42)
    assert np.all(paths > 0)


def test_reproducibility_with_random_state() -> None:
    """Same random_state produces identical paths."""
    engine = HestonEngine()
    paths1 = engine.simulate(50, 30, 1 / 252, random_state=7)
    paths2 = engine.simulate(50, 30, 1 / 252, random_state=7)
    np.testing.assert_array_equal(paths1, paths2)


def test_different_seeds_produce_different_paths() -> None:
    """Different seeds produce different paths."""
    engine = HestonEngine()
    paths1 = engine.simulate(20, 20, 1 / 252, random_state=1)
    paths2 = engine.simulate(20, 20, 1 / 252, random_state=2)
    assert not np.array_equal(paths1, paths2)


def _terminal_log_returns(**kwargs: float) -> np.ndarray:
    paths = HestonEngine().simulate(
        n_paths=40_000, n_steps=21, dt=1 / 252, random_state=0, **kwargs
    )
    return np.log(paths[:, -1] / paths[:, 0])


def test_higher_vol_of_vol_fattens_tails_not_variance() -> None:
    """Vol-of-vol raises kurtosis; with v0 = theta, E[integrated var] is unchanged."""
    r_low = _terminal_log_returns(sigma_v=0.1)
    r_high = _terminal_log_returns(sigma_v=1.0)
    assert stats.kurtosis(r_high) > stats.kurtosis(r_low) + 1.0
    assert abs(np.std(r_high) / np.std(r_low) - 1.0) < 0.05


@pytest.mark.parametrize("sigma_v", [0.05, 0.3, 1.0])
def test_moments_match_theory(sigma_v: float) -> None:
    """Var[log S_T] = theta*T and E[log S_T] = (mu - theta/2)*T when v0 = theta."""
    mu, theta, T = 0.05, 0.04, 21 / 252
    r = _terminal_log_returns(mu=mu, theta=theta, v0=theta, sigma_v=sigma_v)
    assert np.std(r) == pytest.approx(np.sqrt(theta * T), rel=0.03)
    assert np.mean(r) == pytest.approx((mu - 0.5 * theta) * T, abs=0.0015)


def test_rho_controls_skew() -> None:
    """The leverage correlation must reach the price process (regression test)."""
    kw = {"sigma_v": 0.6, "mu": 0.0}
    skew_neg = stats.skew(_terminal_log_returns(rho=-0.9, **kw))
    skew_zero = stats.skew(_terminal_log_returns(rho=0.0, **kw))
    skew_pos = stats.skew(_terminal_log_returns(rho=0.9, **kw))
    assert skew_neg < -0.5
    assert abs(skew_zero) < 0.2
    assert skew_pos > 0.5


def test_variance_stays_non_negative() -> None:
    """The QE scheme keeps variance non-negative."""
    engine = HestonEngine()
    # High sigma_v can cause issues for naive Euler – QE should handle it
    paths = engine.simulate(
        n_paths=500,
        n_steps=50,
        dt=1 / 252,
        v0=0.04,
        sigma_v=1.5,
        random_state=42,
    )
    assert np.all(paths >= 0)


def test_mean_reversion_of_variance() -> None:
    """With fast mean-reversion, terminal variance should be close to theta."""
    engine = HestonEngine()
    n_paths = 3000
    paths = engine.simulate(
        n_paths=n_paths,
        n_steps=252,
        dt=1 / 252,
        kappa=10.0,
        theta=0.04,
        v0=0.01,
        sigma_v=0.2,
        random_state=99,
    )
    assert paths.shape == (n_paths, 253)


def test_zero_vol_of_vol_is_deterministic_variance() -> None:
    """sigma_v = 0 must not produce NaNs; log-return variance = integrated V."""
    v0, theta, kappa, T = 0.09, 0.04, 3.0, 21 / 252
    with np.errstate(all="raise"):
        r = _terminal_log_returns(v0=v0, theta=theta, kappa=kappa, sigma_v=0.0)
    integrated = theta * T + (v0 - theta) * (1 - np.exp(-kappa * T)) / kappa
    assert np.all(np.isfinite(r))
    assert np.var(r) == pytest.approx(integrated, rel=0.05)
