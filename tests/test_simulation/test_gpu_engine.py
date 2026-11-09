"""Tests for GPU-accelerated simulation engine (NumPy fallback path)."""

from __future__ import annotations

import numpy as np
import pytest

from augury.simulation.engines.gbm import GBMEngine
from augury.simulation.gpu_engine import GPUEngine


@pytest.fixture()
def gpu_engine() -> GPUEngine:
    """GPUEngine wrapping a standard GBMEngine."""
    return GPUEngine(engine=GBMEngine())


class TestGPUEngine:
    def test_simulate_shape(self, gpu_engine) -> None:
        paths = gpu_engine.simulate(n_paths=100, n_steps=50, dt=1 / 252, S0=100.0)
        assert paths.shape == (100, 51)

    def test_simulate_returns_numpy(self, gpu_engine) -> None:
        paths = gpu_engine.simulate(n_paths=100, n_steps=50, dt=1 / 252)
        assert isinstance(paths, np.ndarray)

    def test_initial_price(self, gpu_engine) -> None:
        paths = gpu_engine.simulate(n_paths=100, n_steps=50, dt=1 / 252, S0=42.0)
        np.testing.assert_allclose(paths[:, 0], 42.0)

    def test_all_positive(self, gpu_engine) -> None:
        paths = gpu_engine.simulate(n_paths=500, n_steps=100, dt=1 / 252, S0=100.0)
        assert np.all(paths > 0)

    def test_reproducibility_via_engine(self, gpu_engine) -> None:
        """Same random_state → same paths (NumPy fallback)."""
        p1 = gpu_engine.simulate(
            n_paths=50,
            n_steps=20,
            dt=1 / 252,
            S0=100.0,
            random_state=99,
        )
        p2 = gpu_engine.simulate(
            n_paths=50,
            n_steps=20,
            dt=1 / 252,
            S0=100.0,
            random_state=99,
        )
        np.testing.assert_array_equal(p1, p2)

    def test_jax_available_is_bool(self, gpu_engine) -> None:
        assert isinstance(gpu_engine.jax_available, bool)

    def test_forwards_extra_params(self) -> None:
        """Extra kwargs like mu and sigma are forwarded to the underlying engine."""
        eng = GPUEngine(engine=GBMEngine())
        paths = eng.simulate(
            n_paths=100,
            n_steps=50,
            dt=1 / 252,
            S0=100.0,
            mu=0.10,
            sigma=0.30,
            random_state=7,
        )
        assert paths.shape == (100, 51)
        assert np.all(paths > 0)
