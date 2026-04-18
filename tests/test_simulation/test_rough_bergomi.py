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


def test_higher_eta_increases_spread(engine) -> None:
    """Higher eta (vol-of-vol) -> wider terminal price distribution."""
    p_low = engine.simulate(2000, 252, 1 / 252, eta=0.5, random_state=42)
    p_high = engine.simulate(2000, 252, 1 / 252, eta=3.0, random_state=42)
    assert np.std(p_high[:, -1]) > np.std(p_low[:, -1])


def test_lower_hurst_rougher_paths(engine) -> None:
    """Lower H produces rougher (more jagged) price paths."""
    p_smooth = engine.simulate(1000, 252, 1 / 252, H=0.45, eta=1.0, random_state=42)
    p_rough = engine.simulate(1000, 252, 1 / 252, H=0.05, eta=1.0, random_state=42)
    # Rougher paths have higher variance in log-return increments
    inc_smooth = np.diff(np.log(p_smooth), axis=1)
    inc_rough = np.diff(np.log(p_rough), axis=1)
    assert np.std(inc_rough) > np.std(inc_smooth)


def test_is_simulation_engine_subclass() -> None:
    """RoughBergomiEngine extends SimulationEngine."""
    from synthquant.simulation.engines.base import SimulationEngine

    assert issubclass(RoughBergomiEngine, SimulationEngine)
