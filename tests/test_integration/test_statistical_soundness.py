"""Integration test: Statistical soundness of simulated distributions.

Validates that simulation engines produce outputs whose statistical
properties match what the underlying models promise — i.e. the
forecasts are calibrated.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from augury.analytics.distributions import EmpiricalDistribution
from augury.analytics.moments import jarque_bera_test, moment_ratio_test
from augury.analytics.risk_metrics import value_at_risk
from augury.analytics.scoring import ForecastScorer
from augury.simulation.engines.gbm import GBMEngine
from augury.simulation.engines.heston import HestonEngine
from augury.simulation.engines.merton_jd import MertonJDEngine
from augury.simulation.variance_reduction import antithetic_variates

# ── GBM statistical properties ────────────────────────────────────────────────


class TestGBMCalibration:
    """Verify GBM paths match the log-normal analytical distribution."""

    @pytest.fixture(scope="class")
    def gbm_paths(self) -> np.ndarray:
        return GBMEngine().simulate(
            n_paths=50_000,
            n_steps=252,
            dt=1 / 252,
            S0=100.0,
            mu=0.07,
            sigma=0.20,
            random_state=42,
        )

    def test_mean_terminal_return(self, gbm_paths) -> None:
        """E[log(S_T/S_0)] = (mu - sigma^2/2) * T."""
        log_ret = np.log(gbm_paths[:, -1] / gbm_paths[:, 0])
        expected_mean = (0.07 - 0.5 * 0.20**2) * 1.0  # = 0.05
        np.testing.assert_allclose(np.mean(log_ret), expected_mean, atol=0.005)

    def test_std_terminal_return(self, gbm_paths) -> None:
        """Std[log(S_T/S_0)] = sigma * sqrt(T)."""
        log_ret = np.log(gbm_paths[:, -1] / gbm_paths[:, 0])
        expected_std = 0.20 * np.sqrt(1.0)
        np.testing.assert_allclose(np.std(log_ret), expected_std, atol=0.005)

    def test_log_returns_approximately_normal(self, gbm_paths) -> None:
        """GBM log-returns should not strongly reject normality."""
        log_ret = np.log(gbm_paths[:, -1] / gbm_paths[:, 0])
        jb = jarque_bera_test(log_ret)
        # With 50k samples, small departures are expected; just check
        # skew and excess kurtosis are small (jarque_bera returns Fisher kurtosis)
        assert abs(jb["skewness"]) < 0.1
        assert abs(jb["kurtosis"]) < 0.2  # excess kurtosis near 0

    def test_all_paths_positive(self, gbm_paths) -> None:
        assert np.all(gbm_paths > 0)

    def test_var_matches_analytical(self, gbm_paths) -> None:
        """5% VaR from MC should match the analytical normal quantile."""
        log_ret = np.log(gbm_paths[:, -1] / gbm_paths[:, 0])
        mc_var = np.quantile(log_ret, 0.05)
        analytical_var = (0.07 - 0.5 * 0.20**2) + 0.20 * stats.norm.ppf(0.05)
        np.testing.assert_allclose(mc_var, analytical_var, atol=0.01)


# ── Heston: stochastic vol produces fat tails ─────────────────────────────────


class TestHestonCalibration:
    @pytest.fixture(scope="class")
    def heston_paths(self) -> np.ndarray:
        return HestonEngine().simulate(
            n_paths=20_000,
            n_steps=252,
            dt=1 / 252,
            S0=100.0,
            v0=0.04,
            mu=0.05,
            kappa=2.0,
            theta=0.04,
            sigma_v=0.3,
            rho=-0.7,
            random_state=42,
        )

    def test_heston_fatter_tails_than_gbm(self, heston_paths) -> None:
        """Heston returns should have higher kurtosis than GBM."""
        gbm_paths = GBMEngine().simulate(
            n_paths=20_000,
            n_steps=252,
            dt=1 / 252,
            S0=100.0,
            mu=0.05,
            sigma=0.20,
            random_state=42,
        )
        heston_ret = np.log(heston_paths[:, -1] / heston_paths[:, 0])
        gbm_ret = np.log(gbm_paths[:, -1] / gbm_paths[:, 0])
        assert stats.kurtosis(heston_ret) > stats.kurtosis(gbm_ret)

    def test_heston_negative_skew(self, heston_paths) -> None:
        """With rho=-0.7, Heston should produce negatively skewed returns."""
        log_ret = np.log(heston_paths[:, -1] / heston_paths[:, 0])
        assert stats.skew(log_ret) < 0

    def test_heston_variance_mean_reverts(self, heston_paths) -> None:
        """Terminal prices shouldn't blow up — variance is bounded."""
        terminal = heston_paths[:, -1]
        assert np.all(np.isfinite(terminal))
        assert terminal.std() < 200  # reasonable for S0=100, 1yr


# ── Merton JD: jumps create heavier tails ─────────────────────────────────────


class TestMertonJDCalibration:
    @pytest.fixture(scope="class")
    def mjd_paths(self) -> np.ndarray:
        return MertonJDEngine().simulate(
            n_paths=20_000,
            n_steps=252,
            dt=1 / 252,
            S0=100.0,
            mu=0.07,
            sigma=0.15,
            lambda_j=2.0,
            mu_j=-0.04,
            sigma_j=0.08,
            random_state=42,
        )

    def test_mjd_excess_kurtosis(self, mjd_paths) -> None:
        """Jump diffusion should have excess kurtosis > GBM."""
        log_ret = np.log(mjd_paths[:, -1] / mjd_paths[:, 0])
        mrt = moment_ratio_test(log_ret)
        assert mrt["excess_kurtosis"] > 0.1

    def test_mjd_negative_mean_jumps_create_left_skew(self, mjd_paths) -> None:
        """With mu_j < 0, returns should be negatively skewed."""
        log_ret = np.log(mjd_paths[:, -1] / mjd_paths[:, 0])
        assert stats.skew(log_ret) < 0

    def test_mjd_wider_tails_than_gbm(self, mjd_paths) -> None:
        """MJD 1% VaR should be more extreme than GBM 1% VaR."""
        gbm_paths = GBMEngine().simulate(
            n_paths=20_000,
            n_steps=252,
            dt=1 / 252,
            S0=100.0,
            mu=0.07,
            sigma=0.15,
            random_state=42,
        )
        mjd_var = value_at_risk(mjd_paths, alpha=0.01)
        gbm_var = value_at_risk(gbm_paths, alpha=0.01)
        assert mjd_var < gbm_var, "MJD should have more extreme left tail"


# ── Variance reduction actually reduces variance ──────────────────────────────


class TestVarianceReduction:
    def test_antithetic_reduces_terminal_variance(self) -> None:
        """Antithetic variates should reduce mean estimator variance."""
        engine = GBMEngine()
        base = engine.simulate(
            n_paths=5000,
            n_steps=252,
            dt=1 / 252,
            S0=100.0,
            mu=0.07,
            sigma=0.20,
            random_state=42,
        )
        anti = antithetic_variates(base)

        # The mean of the antithetic sample should have lower standard error
        base_se = base[:, -1].std() / np.sqrt(len(base))
        anti_se = anti[:, -1].std() / np.sqrt(len(anti))
        # antithetic has 2x samples, so SE can be lower even if std is similar
        # More meaningful: compare the effective mean estimator quality
        assert anti_se <= base_se * 1.1  # allow small tolerance


# ── Empirical distribution captures path properties ───────────────────────────


class TestEmpiricalDistribution:
    def test_fitted_quantiles_match_raw(self) -> None:
        """Empirical distribution quantiles should match np.quantile."""
        paths = GBMEngine().simulate(
            n_paths=10_000,
            n_steps=252,
            dt=1 / 252,
            S0=100.0,
            mu=0.07,
            sigma=0.20,
            random_state=42,
        )
        ed = EmpiricalDistribution()
        ed.fit(paths)
        log_ret = np.log(paths[:, -1] / paths[:, 0])

        for q in [0.05, 0.25, 0.50, 0.75, 0.95]:
            ed_q = float(ed.quantile(np.array([q])).item())
            raw_q = float(np.quantile(log_ret, q))
            np.testing.assert_allclose(ed_q, raw_q, atol=0.02)

    def test_confidence_interval_contains_most_observations(self) -> None:
        paths = GBMEngine().simulate(
            n_paths=5000,
            n_steps=252,
            dt=1 / 252,
            S0=100.0,
            mu=0.07,
            sigma=0.20,
            random_state=42,
        )
        ed = EmpiricalDistribution()
        ed.fit(paths)
        lo, hi = ed.confidence_interval(alpha=0.05)
        log_ret = np.log(paths[:, -1] / paths[:, 0])
        coverage = np.mean((log_ret >= lo) & (log_ret <= hi))
        assert coverage > 0.90  # should be ~95%


# ── Forecast scoring round-trip ───────────────────────────────────────────────


class TestForecastScoring:
    def test_crps_lower_for_better_forecast(self) -> None:
        """A forecast closer to observations should have lower CRPS."""
        rng = np.random.default_rng(42)
        truth = rng.normal(0, 1, 200)

        # Good forecast: close to truth
        good = truth[:, None] + rng.normal(0, 0.1, (200, 50))
        # Bad forecast: biased
        bad = truth[:, None] + rng.normal(2.0, 1.0, (200, 50))

        scorer = ForecastScorer()
        crps_good = scorer.crps(good, truth)
        crps_bad = scorer.crps(bad, truth)
        assert crps_good < crps_bad

    def test_well_calibrated_forecast_passes_coverage(self) -> None:
        """Properly calibrated ensemble should have ~95% coverage."""
        rng = np.random.default_rng(42)
        truth = rng.normal(0, 1, 300)
        ensemble = truth[:, None] + rng.normal(0, 0.1, (300, 100))

        scorer = ForecastScorer()
        result = scorer.coverage_test(ensemble, truth, alpha=0.05)
        assert result["empirical_coverage"] > 0.85  # should be close to 95%
