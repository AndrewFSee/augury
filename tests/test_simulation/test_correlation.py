"""Tests for copula-based multi-asset correlated simulation."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from augury.simulation.correlation import CopulaSimulator


@pytest.fixture()
def corr_2x2() -> np.ndarray:
    """Simple 2-asset correlation matrix."""
    return np.array([[1.0, 0.6], [0.6, 1.0]])


@pytest.fixture()
def corr_3x3() -> np.ndarray:
    """3-asset correlation matrix."""
    return np.array(
        [
            [1.0, 0.5, 0.3],
            [0.5, 1.0, 0.4],
            [0.3, 0.4, 1.0],
        ]
    )


# ─── Gaussian copula ──────────────────────────────────────────────────


class TestGaussianCopula:
    def test_output_shape(self, corr_2x2) -> None:
        sim = CopulaSimulator(random_state=42)
        U = sim.simulate(n_paths=500, n_assets=2, correlation_matrix=corr_2x2)
        assert U.shape == (500, 2)

    def test_marginals_in_unit_interval(self, corr_2x2) -> None:
        sim = CopulaSimulator(random_state=42)
        U = sim.simulate(n_paths=1000, n_assets=2, correlation_matrix=corr_2x2)
        assert np.all(U >= 0) and np.all(U <= 1)

    def test_marginals_approximately_uniform(self, corr_2x2) -> None:
        """Each marginal should be approximately U(0,1)."""
        sim = CopulaSimulator(random_state=42)
        U = sim.simulate(n_paths=5000, n_assets=2, correlation_matrix=corr_2x2)
        for i in range(2):
            ks_stat, p_val = stats.kstest(U[:, i], "uniform")
            assert p_val > 0.01, f"Marginal {i} failed uniformity: p={p_val:.4f}"

    def test_correlation_preserved(self, corr_2x2) -> None:
        """Rank correlation should roughly match input correlation."""
        sim = CopulaSimulator(random_state=42)
        U = sim.simulate(n_paths=10000, n_assets=2, correlation_matrix=corr_2x2)
        # Spearman rank correlation for Gaussian copula ≈ (6/pi)*arcsin(rho/2)
        spearman_rho = stats.spearmanr(U[:, 0], U[:, 1]).statistic
        assert spearman_rho > 0.3, f"Expected positive correlation, got {spearman_rho:.4f}"

    def test_three_assets(self, corr_3x3) -> None:
        sim = CopulaSimulator(random_state=42)
        U = sim.simulate(n_paths=1000, n_assets=3, correlation_matrix=corr_3x3)
        assert U.shape == (1000, 3)
        assert np.all(U >= 0) and np.all(U <= 1)

    def test_reproducibility(self, corr_2x2) -> None:
        U1 = CopulaSimulator(random_state=42).simulate(500, 2, corr_2x2)
        U2 = CopulaSimulator(random_state=42).simulate(500, 2, corr_2x2)
        np.testing.assert_array_equal(U1, U2)

    def test_identity_correlation_independent(self) -> None:
        """Identity correlation matrix produces approximately independent marginals."""
        corr = np.eye(3)
        sim = CopulaSimulator(random_state=42)
        U = sim.simulate(n_paths=5000, n_assets=3, correlation_matrix=corr)
        for i in range(3):
            for j in range(i + 1, 3):
                r = np.corrcoef(U[:, i], U[:, j])[0, 1]
                assert abs(r) < 0.05, f"Assets {i},{j} not independent: r={r:.4f}"


# ─── Student-t copula ─────────────────────────────────────────────────


class TestStudentTCopula:
    def test_output_shape(self, corr_2x2) -> None:
        sim = CopulaSimulator(random_state=42)
        U = sim.simulate(500, 2, corr_2x2, copula_type="student_t", df=5.0)
        assert U.shape == (500, 2)

    def test_marginals_in_unit_interval(self, corr_2x2) -> None:
        sim = CopulaSimulator(random_state=42)
        U = sim.simulate(1000, 2, corr_2x2, copula_type="student_t", df=5.0)
        assert np.all(U >= 0) and np.all(U <= 1)

    def test_marginals_approximately_uniform(self, corr_2x2) -> None:
        sim = CopulaSimulator(random_state=42)
        U = sim.simulate(5000, 2, corr_2x2, copula_type="student_t", df=5.0)
        for i in range(2):
            ks_stat, p_val = stats.kstest(U[:, i], "uniform")
            assert p_val > 0.01, f"Marginal {i} not uniform: p={p_val:.4f}"

    def test_tail_dependence_stronger_than_gaussian(self, corr_2x2) -> None:
        """Student-t copula should show stronger lower tail dependence."""
        n = 10000
        sim = CopulaSimulator(random_state=42)
        U_gauss = sim.simulate(n, 2, corr_2x2, copula_type="gaussian")
        sim2 = CopulaSimulator(random_state=42)
        U_t = sim2.simulate(n, 2, corr_2x2, copula_type="student_t", df=3.0)
        # Count joint tail events (both below 5th percentile)
        q = 0.05
        gauss_tail = np.mean((U_gauss[:, 0] < q) & (U_gauss[:, 1] < q))
        t_tail = np.mean((U_t[:, 0] < q) & (U_t[:, 1] < q))
        assert t_tail > gauss_tail

    def test_lower_df_more_tail_dependence(self, corr_2x2) -> None:
        """Lower degrees of freedom -> more extreme co-movements."""
        n = 10000
        U_high = CopulaSimulator(random_state=42).simulate(
            n, 2, corr_2x2, copula_type="student_t", df=30.0
        )
        U_low = CopulaSimulator(random_state=42).simulate(
            n, 2, corr_2x2, copula_type="student_t", df=3.0
        )
        q = 0.05
        high_tail = np.mean((U_high[:, 0] < q) & (U_high[:, 1] < q))
        low_tail = np.mean((U_low[:, 0] < q) & (U_low[:, 1] < q))
        assert low_tail >= high_tail


# ─── Clayton copula ───────────────────────────────────────────────────


class TestClaytonCopula:
    def test_output_shape(self) -> None:
        corr = np.eye(2)  # not used by Clayton, but required by API
        sim = CopulaSimulator(random_state=42)
        U = sim.simulate(500, 2, corr, copula_type="clayton", theta=2.0)
        assert U.shape == (500, 2)

    def test_marginals_in_unit_interval(self) -> None:
        corr = np.eye(2)
        sim = CopulaSimulator(random_state=42)
        U = sim.simulate(1000, 2, corr, copula_type="clayton", theta=2.0)
        assert np.all(U > 0) and np.all(U < 1)

    def test_higher_theta_stronger_dependence(self) -> None:
        """Higher theta -> stronger lower tail dependence."""
        corr = np.eye(2)
        n = 10000
        U_low = CopulaSimulator(random_state=42).simulate(
            n, 2, corr, copula_type="clayton", theta=0.5
        )
        U_high = CopulaSimulator(random_state=42).simulate(
            n, 2, corr, copula_type="clayton", theta=5.0
        )
        r_low = stats.spearmanr(U_low[:, 0], U_low[:, 1]).statistic
        r_high = stats.spearmanr(U_high[:, 0], U_high[:, 1]).statistic
        assert r_high > r_low

    def test_positive_dependence(self) -> None:
        """Clayton copula produces positive dependence."""
        corr = np.eye(2)
        sim = CopulaSimulator(random_state=42)
        U = sim.simulate(5000, 2, corr, copula_type="clayton", theta=2.0)
        r = stats.spearmanr(U[:, 0], U[:, 1]).statistic
        assert r > 0


# ─── Invalid copula type ──────────────────────────────────────────────


def test_invalid_copula_type_raises(corr_2x2) -> None:
    sim = CopulaSimulator(random_state=42)
    with pytest.raises(ValueError, match="Unknown copula_type"):
        sim.simulate(100, 2, corr_2x2, copula_type="frank")


# ─── generate_correlated_returns ──────────────────────────────────────


class TestGenerateCorrelatedReturns:
    def test_output_shape(self, corr_2x2) -> None:
        sim = CopulaSimulator(random_state=42)
        marginals = [stats.norm.ppf, stats.norm.ppf]
        returns = sim.generate_correlated_returns(corr_2x2, marginals, n_paths=500)
        assert returns.shape == (500, 2)

    def test_custom_marginals(self, corr_2x2) -> None:
        """Returns reflect the marginal distributions provided."""
        sim = CopulaSimulator(random_state=42)
        # Asset 0: normal(0, 1), Asset 1: t(df=3)
        marginals = [stats.norm.ppf, lambda u: stats.t.ppf(u, df=3)]
        returns = sim.generate_correlated_returns(corr_2x2, marginals, n_paths=5000)
        # t-distribution has heavier tails -> higher kurtosis
        from scipy.stats import kurtosis

        k_norm = kurtosis(returns[:, 0])
        k_t = kurtosis(returns[:, 1])
        assert k_t > k_norm

    def test_correlation_flows_through(self, corr_2x2) -> None:
        """Correlated returns should preserve rank correlation."""
        sim = CopulaSimulator(random_state=42)
        marginals = [stats.norm.ppf, stats.norm.ppf]
        returns = sim.generate_correlated_returns(corr_2x2, marginals, n_paths=10000)
        r = np.corrcoef(returns[:, 0], returns[:, 1])[0, 1]
        assert r > 0.3, f"Expected positive correlation, got {r:.4f}"
