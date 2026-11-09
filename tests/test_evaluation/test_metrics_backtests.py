"""Tests for scoring rules and VaR / forecast-comparison backtests."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from synthquant.analytics.scoring import ForecastScorer
from synthquant.evaluation import (
    basel_zone,
    christoffersen_test,
    conditional_coverage_test,
    crps_ensemble,
    diebold_mariano,
    interval_coverage,
    kupiec_test,
    pit_values,
)


def _crps_bruteforce(x: np.ndarray, y: float) -> float:
    return float(np.mean(np.abs(x - y)) - 0.5 * np.mean(np.abs(x[:, None] - x[None, :])))


class TestCRPS:
    def test_matches_pairwise_formula(self):
        rng = np.random.default_rng(0)
        for _ in range(5):
            x = rng.standard_t(4, size=301)
            y = float(rng.normal())
            assert crps_ensemble(x, y) == pytest.approx(_crps_bruteforce(x, y), rel=1e-12)

    def test_batch_matches_single(self):
        rng = np.random.default_rng(1)
        X = rng.normal(size=(7, 200))
        y = rng.normal(size=7)
        batch = crps_ensemble(X, y)
        assert batch.shape == (7,)
        np.testing.assert_allclose(batch, [crps_ensemble(X[i], y[i]) for i in range(7)])

    def test_point_mass_is_absolute_error(self):
        assert crps_ensemble(np.full(50, 2.0), 3.5) == pytest.approx(1.5)

    def test_normal_closed_form(self):
        """Quantile-grid samples of N(0, s^2) reproduce the Gneiting closed form."""
        s, y = 0.02, 0.013
        grid = (np.arange(4000) + 0.5) / 4000
        x = s * stats.norm.ppf(grid)
        z = y / s
        exact = s * (z * (2 * stats.norm.cdf(z) - 1) + 2 * stats.norm.pdf(z) - 1 / np.sqrt(np.pi))
        assert crps_ensemble(x, y) == pytest.approx(exact, rel=1e-3)

    def test_forecast_scorer_uses_fast_crps(self):
        """ForecastScorer.crps keeps its old (pairwise) values with O(n) memory."""
        rng = np.random.default_rng(2)
        F = rng.normal(size=(4, 150))
        y = rng.normal(size=4)
        expected = np.mean([_crps_bruteforce(F[i], y[i]) for i in range(4)])
        assert ForecastScorer.crps(F, y) == pytest.approx(expected, rel=1e-12)
        # 10k members would need ~800 MB with the pairwise implementation
        big = rng.normal(size=(2, 10_000))
        assert np.isfinite(ForecastScorer.crps(big, np.zeros(2)))


class TestPITAndCoverage:
    def test_calibrated_pit_is_uniform(self):
        rng = np.random.default_rng(3)
        samples = rng.normal(size=(2000, 500))
        y = rng.normal(size=2000)
        pit = pit_values(samples, y)
        assert stats.kstest(pit, "uniform").pvalue > 0.01

    def test_interval_coverage(self):
        rng = np.random.default_rng(4)
        samples = rng.normal(size=(3000, 1000))
        y = rng.normal(size=3000)
        covered, width = interval_coverage(samples, y, 0.9)
        assert covered.mean() == pytest.approx(0.9, abs=0.02)
        assert width.mean() == pytest.approx(2 * 1.645, rel=0.03)


class TestKupiec:
    def test_exact_rate_not_rejected(self):
        hits = np.zeros(1000, dtype=bool)
        hits[::100] = True  # exactly 1%
        lr, p = kupiec_test(hits, 0.01)
        assert lr == pytest.approx(0.0, abs=1e-9)
        assert p == pytest.approx(1.0)

    def test_zero_violations_is_rejected(self):
        """No violations in 1000 days at 1% is evidence of an over-wide VaR."""
        lr, p = kupiec_test(np.zeros(1000, dtype=bool), 0.01)
        assert lr == pytest.approx(-2 * 1000 * np.log(0.99))
        assert p < 0.001

    def test_too_many_violations_rejected(self):
        hits = np.zeros(1000, dtype=bool)
        hits[:30] = True
        assert kupiec_test(hits, 0.01)[1] < 1e-5


class TestChristoffersen:
    def test_clustered_violations_rejected(self):
        hits = np.zeros(1000, dtype=bool)
        for start in range(50, 1000, 200):
            hits[start : start + 4] = True  # 5 clusters of 4
        assert christoffersen_test(hits)[1] < 1e-4
        # Rate is 2% so coverage alone is borderline; the joint test rejects.
        assert conditional_coverage_test(hits, 0.02)[1] < 1e-3

    def test_independent_violations_not_rejected(self):
        rng = np.random.default_rng(5)
        hits = rng.random(2000) < 0.05
        assert christoffersen_test(hits)[1] > 0.01

    def test_no_violations(self):
        assert christoffersen_test(np.zeros(100, dtype=bool)) == (0.0, 1.0)


class TestBaselZone:
    @pytest.mark.parametrize(
        ("k", "zone"),
        [(0, "Green"), (4, "Green"), (5, "Yellow"), (9, "Yellow"), (10, "Red")],
    )
    def test_regulatory_table(self, k, zone):
        assert basel_zone(k, 250, 0.01) == zone


class TestDieboldMariano:
    def test_better_model_detected(self):
        rng = np.random.default_rng(6)
        base = rng.gamma(2.0, 1.0, size=500)
        better = base * 0.9 + rng.normal(0, 0.05, size=500)
        stat, p = diebold_mariano(better, base)
        assert stat < 0 and p < 1e-3
        assert diebold_mariano(better, base, alternative="less")[1] < p

    def test_equal_models_not_rejected(self):
        rng = np.random.default_rng(7)
        a, b = rng.gamma(2.0, 1.0, size=(2, 500))
        assert diebold_mariano(a, b, lags=4)[1] > 0.01

    def test_identical_losses_undefined(self):
        x = np.ones(50)
        stat, p = diebold_mariano(x, x)
        assert np.isnan(stat) and np.isnan(p)
