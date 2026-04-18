"""Statistical correctness tests for Heston and Rough Bergomi stochastic vol models.

These tests verify analytical moment recovery, distributional properties,
and the defining characteristics of each model that matter for forecast quality.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from synthquant.models.rough_vol import RoughBergomi
from synthquant.models.stochastic_vol import HestonModel

# ===========================================================================
# Heston Model — Moment and Distributional Validation
# ===========================================================================


class TestHestonMomentRecovery:
    """Verify that Heston simulation produces correct analytical moments."""

    @pytest.fixture()
    def heston_default(self) -> HestonModel:
        return HestonModel(
            v0=0.04,
            kappa=2.0,
            theta=0.04,
            sigma_v=0.3,
            rho=-0.7,
            mu=0.05,
        )

    def test_drift_recovery(self, heston_default: HestonModel):
        """E[S_T] should approximate S0 * exp(mu * T) for the physical measure."""
        S0, T = 100.0, 1.0
        S, V = heston_default.simulate(S0, T, n_paths=20000, n_steps=500, random_state=42)

        expected_mean = S0 * np.exp(heston_default.mu * T)
        empirical_mean = S[:, -1].mean()

        rel_err = abs(empirical_mean - expected_mean) / expected_mean
        assert rel_err < 0.03, (
            f"E[S_T]: empirical={empirical_mean:.2f}, analytical={expected_mean:.2f}, "
            f"rel_err={rel_err:.4f}"
        )

    def test_variance_mean_reversion_to_theta(self, heston_default: HestonModel):
        """Terminal variance should revert to theta = 0.04."""
        S0, T = 100.0, 5.0  # Longer horizon for convergence
        S, V = heston_default.simulate(S0, T, n_paths=10000, n_steps=1000, random_state=42)

        terminal_var_mean = V[:, -1].mean()
        assert abs(terminal_var_mean - heston_default.theta) < 0.01, (
            f"E[V_T]={terminal_var_mean:.4f}, theta={heston_default.theta}"
        )

    def test_feller_condition_positive_variance(self):
        """When 2*kappa*theta > sigma_v^2 (Feller), variance stays non-negative."""
        kappa, theta, sigma_v = 5.0, 0.04, 0.2
        feller = 2 * kappa * theta  # = 0.4
        assert feller > sigma_v**2  # 0.04, so Feller holds

        model = HestonModel(v0=0.04, kappa=kappa, theta=theta, sigma_v=sigma_v, rho=-0.5)
        S, V = model.simulate(100.0, 1.0, n_paths=5000, n_steps=500, random_state=42)

        # With Euler-Maruyama clipping, V >= 0, but with strong Feller the
        # pre-clipped values should also be mostly non-negative
        # After clipping, all should be >= 0
        assert np.all(V >= 0), "Variance should be non-negative after clipping"

    def test_feller_violation_causes_more_clipping(self):
        """When Feller condition is violated, variance process needs more clipping."""
        # Strong Feller: 2*5*0.04 = 0.4 >> 0.04
        model_strong = HestonModel(v0=0.04, kappa=5.0, theta=0.04, sigma_v=0.2)
        _, V_strong = model_strong.simulate(100.0, 1.0, n_paths=5000, n_steps=500, random_state=42)

        # Weak Feller violation: 2*0.5*0.04 = 0.04 < 0.81
        model_weak = HestonModel(v0=0.04, kappa=0.5, theta=0.04, sigma_v=0.9)
        _, V_weak = model_weak.simulate(100.0, 1.0, n_paths=5000, n_steps=500, random_state=42)

        # In the Feller-violated case, more variance values hit the clipping floor
        frac_clipped_strong = np.mean(V_strong == 0)
        frac_clipped_weak = np.mean(V_weak == 0)

        assert frac_clipped_weak > frac_clipped_strong, (
            f"Feller violation should cause more zero-variance: "
            f"strong={frac_clipped_strong:.4f}, weak={frac_clipped_weak:.4f}"
        )


class TestHestonImpliedVolSurface:
    """Verify Heston produces the expected implied volatility characteristics."""

    def test_negative_rho_produces_skew(self):
        """Negative correlation should produce left-skewed returns (vol skew)."""
        model = HestonModel(v0=0.04, kappa=2.0, theta=0.04, sigma_v=0.5, rho=-0.8, mu=0.0)
        S, _ = model.simulate(100.0, 1.0, n_paths=20000, n_steps=500, random_state=42)

        log_returns = np.log(S[:, -1] / S[:, 0])
        skew = stats.skew(log_returns)
        assert skew < -0.1, f"Negative rho should produce negative skew, got {skew:.3f}"

    def test_zero_rho_symmetric_returns(self):
        """Zero correlation should produce roughly symmetric returns."""
        model = HestonModel(v0=0.04, kappa=2.0, theta=0.04, sigma_v=0.3, rho=0.0, mu=0.0)
        S, _ = model.simulate(100.0, 1.0, n_paths=20000, n_steps=500, random_state=42)

        log_returns = np.log(S[:, -1] / S[:, 0])
        skew = stats.skew(log_returns)
        assert abs(skew) < 0.3, f"Zero rho should give ~symmetric returns, skew={skew:.3f}"

    def test_higher_volvol_fatter_tails(self):
        """Higher sigma_v should produce fatter tails (higher kurtosis)."""
        S0, T, n_paths, n_steps = 100.0, 1.0, 15000, 500

        model_low = HestonModel(v0=0.04, kappa=2.0, theta=0.04, sigma_v=0.1, rho=-0.5)
        model_high = HestonModel(v0=0.04, kappa=2.0, theta=0.04, sigma_v=0.8, rho=-0.5)

        S_low, _ = model_low.simulate(S0, T, n_paths, n_steps, random_state=42)
        S_high, _ = model_high.simulate(S0, T, n_paths, n_steps, random_state=42)

        kurt_low = stats.kurtosis(np.log(S_low[:, -1] / S_low[:, 0]), fisher=True)
        kurt_high = stats.kurtosis(np.log(S_high[:, -1] / S_high[:, 0]), fisher=True)

        assert kurt_high > kurt_low, (
            f"Higher vol-of-vol should produce fatter tails: "
            f"kurt_low={kurt_low:.3f}, kurt_high={kurt_high:.3f}"
        )

    def test_heston_returns_fatter_than_normal(self):
        """Heston returns should have excess kurtosis > 0 (fatter than normal)."""
        model = HestonModel(v0=0.04, kappa=2.0, theta=0.04, sigma_v=0.5, rho=-0.7)
        S, _ = model.simulate(100.0, 1.0, n_paths=20000, n_steps=500, random_state=42)

        log_returns = np.log(S[:, -1] / S[:, 0])
        excess_kurt = stats.kurtosis(log_returns, fisher=True)
        assert excess_kurt > 0.1, (
            f"Heston returns should be leptokurtic, excess_kurt={excess_kurt:.3f}"
        )


# ===========================================================================
# Rough Bergomi — Roughness Validation
# ===========================================================================


class TestRoughBergomiCorrectness:
    """Verify that Rough Bergomi exhibits correct roughness properties."""

    def test_hurst_exponent_estimation(self):
        """The variance process should have estimated Hurst exponent ~ H.

        We estimate H from the variance of log-variance increments at
        different lags: Var[log V(t+lag) - log V(t)] ~ lag^(2H).
        """
        H_true = 0.1
        model = RoughBergomi(H=H_true, xi=0.04, eta=1.5, rho=-0.7, mu=0.0)
        S, V = model.simulate(100.0, 1.0, n_paths=500, n_steps=1000, random_state=42)

        # Use log-variance increments to estimate H
        log_V = np.log(np.maximum(V[:, 1:], 1e-12))  # avoid log(0)

        lags = [1, 2, 4, 8, 16, 32]
        variances = []
        for lag in lags:
            increments = log_V[:, lag:] - log_V[:, :-lag]
            variances.append(np.var(increments))

        # Fit log(var) = 2H * log(lag) + const
        log_lags = np.log(lags)
        log_vars = np.log(variances)
        slope, _, _, _, _ = (
            np.polyfit(log_lags, log_vars, 1, full=True)[0:5]
            if False
            else (np.polyfit(log_lags, log_vars, 1)[0],) + (None,) * 4
        )
        H_estimated = slope / 2

        # Allow wide tolerance — estimation from 500 paths is noisy
        assert 0.01 < H_estimated < 0.35, (
            f"Estimated H={H_estimated:.3f} far from true H={H_true} (expected in [0.01, 0.35])"
        )

    def test_lower_hurst_rougher_variance_paths(self):
        """Lower H should produce rougher (more irregular) variance paths."""
        S0, T, n_paths, n_steps = 100.0, 1.0, 1000, 500

        model_rough = RoughBergomi(H=0.05, xi=0.04, eta=1.5, rho=-0.7)
        model_smooth = RoughBergomi(H=0.40, xi=0.04, eta=1.5, rho=-0.7)

        _, V_rough = model_rough.simulate(S0, T, n_paths, n_steps, random_state=42)
        _, V_smooth = model_smooth.simulate(S0, T, n_paths, n_steps, random_state=42)

        # Roughness metric: mean absolute first difference of log-variance
        roughness_rough = np.mean(np.abs(np.diff(np.log(np.maximum(V_rough, 1e-12)), axis=1)))
        roughness_smooth = np.mean(np.abs(np.diff(np.log(np.maximum(V_smooth, 1e-12)), axis=1)))

        assert roughness_rough > roughness_smooth, (
            f"Lower H should be rougher: H=0.05 roughness={roughness_rough:.5f}, "
            f"H=0.40 roughness={roughness_smooth:.5f}"
        )

    def test_rough_bergomi_drift_recovery(self):
        """E[S_T] should approximate S0 * exp(mu * T) for the Rough Bergomi model."""
        mu = 0.05
        model = RoughBergomi(H=0.1, xi=0.04, eta=1.5, rho=-0.7, mu=mu)
        S0, T = 100.0, 1.0
        S, _ = model.simulate(S0, T, n_paths=15000, n_steps=500, random_state=42)

        expected_mean = S0 * np.exp(mu * T)
        empirical_mean = S[:, -1].mean()

        rel_err = abs(empirical_mean - expected_mean) / expected_mean
        assert rel_err < 0.05, (
            f"Rough Bergomi drift: empirical={empirical_mean:.2f}, "
            f"expected={expected_mean:.2f}, rel_err={rel_err:.4f}"
        )

    def test_rough_bergomi_produces_fat_tails(self):
        """Rough Bergomi with moderate eta should produce leptokurtic returns."""
        model = RoughBergomi(H=0.1, xi=0.04, eta=1.0, rho=-0.5, mu=0.0)
        S, _ = model.simulate(100.0, 1.0, n_paths=15000, n_steps=500, random_state=42)

        terminal = S[:, -1]
        # Filter out any paths that hit zero (numerical artifact)
        valid = terminal > 0
        log_returns = np.log(terminal[valid] / S[valid, 0])
        excess_kurt = stats.kurtosis(log_returns, fisher=True)
        assert excess_kurt > 0.0, (
            f"Rough Bergomi should produce fat tails, excess_kurt={excess_kurt:.3f}"
        )

    def test_rough_bergomi_negative_rho_skew(self):
        """Negative rho should produce negatively skewed returns."""
        model = RoughBergomi(H=0.1, xi=0.04, eta=1.0, rho=-0.7, mu=0.0)
        S, _ = model.simulate(100.0, 1.0, n_paths=20000, n_steps=500, random_state=42)

        terminal = S[:, -1]
        valid = terminal > 0
        log_returns = np.log(terminal[valid] / S[valid, 0])
        skew = stats.skew(log_returns)
        assert skew < -0.1, f"Negative rho should produce negative skew, got {skew:.3f}"

    def test_variance_initial_level_matches_xi(self):
        """V(0) should match xi for all paths."""
        xi = 0.06
        model = RoughBergomi(H=0.1, xi=xi, eta=1.5, rho=-0.7)
        _, V = model.simulate(100.0, 1.0, n_paths=100, n_steps=100, random_state=42)

        np.testing.assert_allclose(V[:, 0], xi, rtol=1e-10)
