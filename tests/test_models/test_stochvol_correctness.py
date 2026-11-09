"""Statistical correctness tests for the Heston stochastic vol model.

These tests verify analytical moment recovery, distributional properties,
and the defining characteristics of each model that matter for forecast quality.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from augury.models.stochastic_vol import HestonModel

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
