"""Tests for variance reduction techniques."""

from __future__ import annotations

import numpy as np
import pytest

from synthquant.simulation.engines.gbm import GBMEngine
from synthquant.simulation.variance_reduction import (
    antithetic_variates,
    control_variates,
    importance_sampling,
    stratified_sampling,
)


@pytest.fixture()
def gbm_paths() -> np.ndarray:
    """1000 GBM paths for testing."""
    engine = GBMEngine()
    return engine.simulate(
        n_paths=1000,
        n_steps=252,
        dt=1 / 252,
        S0=100.0,
        mu=0.07,
        sigma=0.20,
        random_state=42,
    )


# ─── antithetic_variates ──────────────────────────────────────────────


class TestAntitheticVariates:
    def test_doubles_path_count(self, gbm_paths) -> None:
        """Output has 2x the original number of paths."""
        combined = antithetic_variates(gbm_paths)
        assert combined.shape[0] == 2 * gbm_paths.shape[0]

    def test_preserves_n_steps(self, gbm_paths) -> None:
        """Number of time steps is unchanged."""
        combined = antithetic_variates(gbm_paths)
        assert combined.shape[1] == gbm_paths.shape[1]

    def test_original_paths_preserved(self, gbm_paths) -> None:
        """First half of output equals the original paths."""
        combined = antithetic_variates(gbm_paths)
        np.testing.assert_array_equal(combined[: gbm_paths.shape[0]], gbm_paths)

    def test_antithetic_paths_start_at_s0(self, gbm_paths) -> None:
        """Antithetic paths start at the same S0."""
        combined = antithetic_variates(gbm_paths)
        anti = combined[gbm_paths.shape[0] :]
        np.testing.assert_allclose(anti[:, 0], gbm_paths[:, 0])

    def test_antithetic_paths_positive(self, gbm_paths) -> None:
        """Antithetic paths are still strictly positive."""
        combined = antithetic_variates(gbm_paths)
        assert np.all(combined > 0)

    def test_reduces_mean_estimator_variance(self) -> None:
        """Antithetic estimator has lower variance than naive for symmetric payoffs."""
        engine = GBMEngine()
        rng = np.random.default_rng(42)
        naive_means = []
        anti_means = []
        for _ in range(50):
            seed = rng.integers(0, 2**31)
            paths = engine.simulate(
                n_paths=500,
                n_steps=252,
                dt=1 / 252,
                S0=100.0,
                mu=0.07,
                sigma=0.20,
                random_state=seed,
            )
            naive_means.append(paths[:, -1].mean())
            combined = antithetic_variates(paths)
            anti_means.append(combined[:, -1].mean())
        # Antithetic estimator should have lower variance
        assert np.var(anti_means) < np.var(naive_means)


# ─── control_variates ─────────────────────────────────────────────────


class TestControlVariates:
    def test_returns_correct_shape(self, gbm_paths) -> None:
        """Returns 1D array of length n_paths."""
        control = gbm_paths.copy()
        corrected = control_variates(gbm_paths, control, control_mean=100.0)
        assert corrected.shape == (gbm_paths.shape[0],)

    def test_perfect_control_reduces_to_constant(self) -> None:
        """When control == target, corrected values approach control_mean."""
        engine = GBMEngine()
        paths = engine.simulate(
            n_paths=5000,
            n_steps=252,
            dt=1 / 252,
            S0=100.0,
            mu=0.07,
            sigma=0.20,
            random_state=42,
        )
        # Using paths as their own control with known mean
        known_mean = paths[:, -1].mean()
        corrected = control_variates(paths, paths, control_mean=known_mean)
        # Corrected values should have much smaller variance
        assert np.std(corrected) < np.std(paths[:, -1])

    def test_reduces_variance(self) -> None:
        """Control variate correction reduces estimator variance."""
        engine = GBMEngine()
        # Generate two correlated sets of paths (same seed = perfect correlation)
        paths = engine.simulate(
            n_paths=2000,
            n_steps=252,
            dt=1 / 252,
            S0=100.0,
            mu=0.07,
            sigma=0.20,
            random_state=42,
        )
        # GBM theoretical mean: S0 * exp(mu * T)
        control_mean = 100.0 * np.exp(0.07 * 1.0)
        corrected = control_variates(paths, paths, control_mean=control_mean)
        assert np.std(corrected) < np.std(paths[:, -1])


# ─── stratified_sampling ──────────────────────────────────────────────


class TestStratifiedSampling:
    def test_returns_correct_length(self) -> None:
        """Output length equals n_paths."""
        samples = stratified_sampling(n_paths=1000, n_strata=10)
        assert samples.shape == (1000,)

    def test_all_in_unit_interval(self) -> None:
        """All samples are in [0, 1]."""
        samples = stratified_sampling(n_paths=1000, n_strata=10, rng=np.random.default_rng(42))
        assert np.all(samples >= 0)
        assert np.all(samples <= 1)

    def test_each_stratum_covered(self) -> None:
        """Each stratum has exactly n_paths/n_strata samples."""
        n_strata = 10
        n_paths = 1000
        samples = stratified_sampling(
            n_paths=n_paths, n_strata=n_strata, rng=np.random.default_rng(42)
        )
        # Check that each stratum interval has the expected count
        for i in range(n_strata):
            lo = i / n_strata
            hi = (i + 1) / n_strata
            count = np.sum((samples >= lo) & (samples < hi))
            assert count == n_paths // n_strata

    def test_better_coverage_than_random(self) -> None:
        """Stratified samples should be more uniformly distributed than random."""
        rng = np.random.default_rng(42)
        n = 1000
        stratified = stratified_sampling(n_paths=n, n_strata=100, rng=rng)
        random_samples = rng.random(n)
        # KS statistic against uniform should be smaller for stratified
        from scipy.stats import kstest

        ks_strat = kstest(stratified, "uniform").statistic
        ks_rand = kstest(random_samples, "uniform").statistic
        assert ks_strat < ks_rand

    def test_reproducibility(self) -> None:
        """Same rng state gives same samples."""
        s1 = stratified_sampling(n_paths=100, n_strata=10, rng=np.random.default_rng(42))
        s2 = stratified_sampling(n_paths=100, n_strata=10, rng=np.random.default_rng(42))
        np.testing.assert_array_equal(s1, s2)


# ─── importance_sampling ──────────────────────────────────────────────


class TestImportanceSampling:
    def test_returns_correct_shape(self, gbm_paths) -> None:
        """Weights array has length n_paths."""
        weights = importance_sampling(gbm_paths, target_quantile=0.05)
        assert weights.shape == (gbm_paths.shape[0],)

    def test_weights_normalized(self, gbm_paths) -> None:
        """Weights sum to n_paths (self-normalizing)."""
        weights = importance_sampling(gbm_paths, target_quantile=0.05)
        np.testing.assert_allclose(weights.sum(), gbm_paths.shape[0], rtol=1e-10)

    def test_weights_non_negative(self, gbm_paths) -> None:
        """All weights are non-negative."""
        weights = importance_sampling(gbm_paths, target_quantile=0.05)
        assert np.all(weights >= 0)

    def test_tail_paths_have_higher_weight(self, gbm_paths) -> None:
        """Paths in the tail get higher weight than body paths."""
        weights = importance_sampling(gbm_paths, target_quantile=0.05)
        terminal = gbm_paths[:, -1]
        threshold = np.quantile(terminal, 0.05)
        tail_weights = weights[terminal <= threshold]
        body_weights = weights[terminal > threshold]
        assert tail_weights.mean() > body_weights.mean()

    def test_different_quantiles_change_weights(self, gbm_paths) -> None:
        """Different target quantiles produce different weight distributions."""
        w1 = importance_sampling(gbm_paths, target_quantile=0.01)
        w2 = importance_sampling(gbm_paths, target_quantile=0.10)
        assert not np.allclose(w1, w2)
