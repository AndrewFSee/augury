"""Tests for Rough Bergomi simulation engine."""

from __future__ import annotations

import numpy as np
import pytest

from synthquant.simulation.engines.rough_bergomi import RoughBergomiEngine


@pytest.fixture()
def engine() -> RoughBergomiEngine:
    return RoughBergomiEngine()


def test_simulate_shape(engine) -> None:
    """Output has shape (n_paths, n_steps+1)."""
    paths = engine.simulate(n_paths=100, n_steps=50, dt=1 / 252, random_state=42)
    assert paths.shape == (100, 51)


def test_initial_price(engine) -> None:
    """All paths start at S0."""
    S0 = 120.0
    paths = engine.simulate(n_paths=200, n_steps=20, dt=1 / 252, S0=S0, random_state=42)
    np.testing.assert_allclose(paths[:, 0], S0)


def test_all_prices_positive(engine) -> None:
    """Price paths are strictly positive."""
    paths = engine.simulate(n_paths=500, n_steps=50, dt=1 / 252, random_state=42)
    assert np.all(paths > 0)


def test_reproducibility(engine) -> None:
    """Same seed gives identical paths."""
    p1 = engine.simulate(50, 30, 1 / 252, random_state=7)
    p2 = engine.simulate(50, 30, 1 / 252, random_state=7)
    np.testing.assert_array_equal(p1, p2)


def test_different_seeds_differ(engine) -> None:
    """Different seeds produce different paths."""
    p1 = engine.simulate(20, 20, 1 / 252, random_state=1)
    p2 = engine.simulate(20, 20, 1 / 252, random_state=2)
    assert not np.array_equal(p1, p2)


def test_invalid_hurst_raises(engine) -> None:
    """H outside (0, 0.5) raises ValueError."""
    with pytest.raises(ValueError, match="H must be in"):
        engine.simulate(50, 20, 1 / 252, H=0.5)
    with pytest.raises(ValueError, match="H must be in"):
        engine.simulate(50, 20, 1 / 252, H=0.0)
    with pytest.raises(ValueError, match="H must be in"):
        engine.simulate(50, 20, 1 / 252, H=-0.1)


def test_hurst_boundary_values(engine) -> None:
    """H near boundaries should still work."""
    paths = engine.simulate(50, 20, 1 / 252, H=0.01, random_state=42)
    assert np.all(paths > 0)
    paths = engine.simulate(50, 20, 1 / 252, H=0.49, random_state=42)
    assert np.all(paths > 0)


def test_default_parameters(engine) -> None:
    """Engine uses sensible defaults from the simulate signature."""
    paths = engine.simulate(n_paths=100, n_steps=50, dt=1 / 252, random_state=42)
    # S0 default is 100
    np.testing.assert_allclose(paths[:, 0], 100.0)


def test_higher_eta_fattens_tails(engine) -> None:
    """Higher eta (vol-of-vol) -> fatter tails; E[V] = xi keeps variance ~fixed."""
    from scipy.stats import kurtosis

    r_low = np.log(engine.simulate(20_000, 63, 1 / 252, eta=0.5, random_state=0)[:, -1] / 100)
    r_high = np.log(engine.simulate(20_000, 63, 1 / 252, eta=3.0, random_state=0)[:, -1] / 100)
    assert kurtosis(r_high) > kurtosis(r_low) + 5.0


@pytest.mark.parametrize(("H", "eta"), [(0.05, 1.9), (0.1, 1.9), (0.3, 1.0)])
def test_mean_variance_matches_xi(engine, H: float, eta: float) -> None:
    """E[V(t)] = xi, so E[log S_T] = (mu - xi/2) T and Var[log S_T] ~= xi T.

    Regression test: the Volterra process previously lacked its sqrt(2H)
    normalisation, inflating variance ~5x at H=0.1, eta=1.9.
    """
    xi, T = 0.04, 63 / 252
    p = engine.simulate(40_000, 63, 1 / 252, xi=xi, H=H, eta=eta, mu=0.0, random_state=0)
    r = np.log(p[:, -1] / p[:, 0])
    assert np.mean(r) == pytest.approx(-0.5 * xi * T, abs=0.001)
    assert np.std(r) == pytest.approx(np.sqrt(xi * T), rel=0.06)


def test_lower_hurst_rougher_volatility(engine) -> None:
    """Lower H -> rougher volatility: larger week-to-week changes in log RV."""

    def weekly_log_rv_roughness(H: float) -> float:
        p = engine.simulate(2000, 250, 1 / 252, H=H, eta=1.0, rho=0.0, random_state=1)
        r = np.diff(np.log(p), axis=1)
        rv = (r**2).reshape(2000, -1, 5).sum(axis=2)
        return float(np.std(np.diff(np.log(rv), axis=1)))

    assert weekly_log_rv_roughness(0.05) > weekly_log_rv_roughness(0.45) + 0.05


def test_is_simulation_engine_subclass() -> None:
    """RoughBergomiEngine extends SimulationEngine."""
    from synthquant.simulation.engines.base import SimulationEngine

    assert issubclass(RoughBergomiEngine, SimulationEngine)
