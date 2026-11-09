"""Statistical correctness tests for GARCH family models.

These tests verify that GARCH models can recover known parameters
from data simulated with those parameters, and that forecasts
converge to known unconditional values.
"""

from __future__ import annotations

import numpy as np
import pytest

from augury.models.garch import GARCHModel

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _simulate_garch11_data(
    omega: float = 0.00001,
    alpha: float = 0.10,
    beta: float = 0.85,
    n_obs: int = 2000,
    seed: int = 42,
) -> np.ndarray:
    """Simulate data from a known GARCH(1,1) process.

    sigma_t^2 = omega + alpha * r_{t-1}^2 + beta * sigma_{t-1}^2
    r_t = sigma_t * z_t,   z_t ~ N(0,1)
    """
    rng = np.random.default_rng(seed)
    r = np.zeros(n_obs)
    sigma2 = np.zeros(n_obs)

    # Unconditional variance as starting value
    sigma2[0] = omega / (1 - alpha - beta)

    for t in range(1, n_obs):
        sigma2[t] = omega + alpha * r[t - 1] ** 2 + beta * sigma2[t - 1]
        r[t] = np.sqrt(sigma2[t]) * rng.standard_normal()

    return r


# ===========================================================================
# GARCH(1,1) Parameter Recovery
# ===========================================================================


class TestGARCHParameterRecovery:
    """Verify that GARCH model recovers known parameters from simulated data."""

    def test_garch11_recovers_alpha_beta(self):
        """Fit GARCH(1,1) to data generated with known alpha/beta and verify recovery."""
        true_alpha = 0.10
        true_beta = 0.85
        true_omega = 0.00001

        returns = _simulate_garch11_data(
            omega=true_omega,
            alpha=true_alpha,
            beta=true_beta,
            n_obs=3000,
            seed=42,
        )

        model = GARCHModel(vol_model="GARCH", p=1, q=1, dist="normal")
        model.fit(returns, scale=100.0)

        # Extract fitted parameters (arch library convention):
        # omega, alpha[1], beta[1] in the result params
        params = model._result.params
        param_names = [n.lower() for n in model._result.model.volatility.parameter_names()]

        fitted = dict(zip(param_names, params[-len(param_names) :], strict=False))

        # omega is scaled by scale^2 = 10000
        scale2 = 100.0**2
        _fitted_omega = fitted.get("omega", 0) / scale2
        fitted_alpha = fitted.get("alpha[1]", fitted.get("alpha(1)", 0))
        fitted_beta = fitted.get("beta[1]", fitted.get("beta(1)", 0))

        assert abs(fitted_alpha - true_alpha) < 0.05, (
            f"alpha: fitted={fitted_alpha:.4f}, true={true_alpha}"
        )
        assert abs(fitted_beta - true_beta) < 0.08, (
            f"beta: fitted={fitted_beta:.4f}, true={true_beta}"
        )
        # Persistence (alpha + beta) is the key quantity
        true_persistence = true_alpha + true_beta
        fitted_persistence = fitted_alpha + fitted_beta
        assert abs(fitted_persistence - true_persistence) < 0.05, (
            f"persistence: fitted={fitted_persistence:.4f}, true={true_persistence}"
        )

    def test_unconditional_variance_recovery(self):
        """Unconditional variance ω/(1-α-β) should match empirical."""
        true_omega = 0.00001
        true_alpha = 0.10
        true_beta = 0.85
        true_uncond_var = true_omega / (1 - true_alpha - true_beta)

        returns = _simulate_garch11_data(
            omega=true_omega,
            alpha=true_alpha,
            beta=true_beta,
            n_obs=5000,
            seed=77,
        )

        # Empirical variance should be close to unconditional
        empirical_var = np.var(returns)
        assert abs(empirical_var - true_uncond_var) / true_uncond_var < 0.3, (
            f"Empirical variance {empirical_var:.6f} far from unconditional {true_uncond_var:.6f}"
        )

    def test_garch_forecast_converges_to_unconditional(self):
        """Long-horizon GARCH forecasts should converge to unconditional volatility."""
        true_omega = 0.00001
        true_alpha = 0.10
        true_beta = 0.85
        true_uncond_var = true_omega / (1 - true_alpha - true_beta)
        true_uncond_vol_annual = np.sqrt(true_uncond_var * 252)

        returns = _simulate_garch11_data(
            omega=true_omega,
            alpha=true_alpha,
            beta=true_beta,
            n_obs=3000,
            seed=42,
        )

        model = GARCHModel(vol_model="GARCH", p=1, q=1, dist="normal")
        model.fit(returns, scale=100.0)

        # 252-day forecast
        vol_forecast = model.forecast(horizon=252)
        # Last forecast should be close to unconditional annualized vol
        terminal_vol = vol_forecast[-1]
        assert abs(terminal_vol - true_uncond_vol_annual) / true_uncond_vol_annual < 0.3, (
            f"Terminal vol forecast {terminal_vol:.4f} far from "
            f"unconditional {true_uncond_vol_annual:.4f}"
        )

    def test_garch_simulation_mean_return_near_zero(self):
        """GARCH simulated returns should have mean ~0 (zero-mean model)."""
        returns = _simulate_garch11_data(n_obs=2000, seed=42)
        model = GARCHModel(vol_model="GARCH", p=1, q=1, dist="normal")
        model.fit(returns, scale=100.0)

        paths = model.simulate(n_paths=500, horizon=252, random_state=42)
        mean_return = paths.mean()
        assert abs(mean_return) < 0.005, f"Mean simulated return {mean_return:.5f} far from 0"

    def test_high_persistence_garch_has_slow_mean_reversion(self):
        """High-persistence GARCH should show slowly decaying vol forecasts."""
        returns = _simulate_garch11_data(
            omega=0.000005,
            alpha=0.05,
            beta=0.93,
            n_obs=3000,
            seed=42,
        )
        model = GARCHModel(vol_model="GARCH", p=1, q=1, dist="normal")
        model.fit(returns, scale=100.0)

        vol_forecast = model.forecast(horizon=100)

        # With high persistence, early forecasts should vary more from terminal
        # Check that forecasts are monotonically approaching a limit
        diffs = np.abs(np.diff(vol_forecast))
        # Later diffs should be smaller (convergence)
        early_diffs = diffs[:10].mean()
        late_diffs = diffs[-10:].mean()
        assert late_diffs < early_diffs + 1e-6, (
            "High-persistence GARCH should converge, not diverge"
        )


# ===========================================================================
# GARCH Distributional Properties
# ===========================================================================


class TestGARCHDistributionalProperties:
    """Verify GARCH model produces correct distributional properties."""

    def test_garch_volatility_clustering(self):
        """GARCH data should exhibit volatility clustering:
        autocorrelation of squared returns should be positive at lag 1."""
        returns = _simulate_garch11_data(
            omega=0.00001,
            alpha=0.10,
            beta=0.85,
            n_obs=3000,
            seed=42,
        )
        model = GARCHModel(vol_model="GARCH", p=1, q=1, dist="normal")
        model.fit(returns, scale=100.0)

        paths = model.simulate(n_paths=1, horizon=3000, random_state=42)
        sim_returns = paths[0]

        # Autocorrelation of squared returns at lag 1
        sq = sim_returns**2
        sq_demean = sq - sq.mean()
        acf_1 = np.correlate(sq_demean[:-1], sq_demean[1:]) / np.sum(sq_demean**2)
        assert acf_1[0] > 0, "Squared returns should show positive autocorrelation (vol clustering)"

    def test_garch_excess_kurtosis(self):
        """GARCH-simulated returns should be leptokurtic (excess kurtosis > 0)."""
        returns = _simulate_garch11_data(
            omega=0.00001,
            alpha=0.10,
            beta=0.85,
            n_obs=2000,
            seed=42,
        )
        model = GARCHModel(vol_model="GARCH", p=1, q=1, dist="normal")
        model.fit(returns, scale=100.0)

        # Pool many simulated paths to get kurtosis
        paths = model.simulate(n_paths=200, horizon=252, random_state=42)
        all_returns = paths.flatten()
        from scipy.stats import kurtosis as sp_kurtosis

        k = sp_kurtosis(all_returns, fisher=True)
        assert k > 0, f"GARCH returns should be leptokurtic, got excess kurtosis={k:.3f}"


# ===========================================================================
# Conditional simulation (regression tests)
# ===========================================================================


class TestGARCHConditionalSimulation:
    """simulate() must start from today's conditional variance, not the
    unconditional level, and must honour random_state."""

    @staticmethod
    def _fit_ending_in_stress() -> GARCHModel:
        calm = _simulate_garch11_data(n_obs=1500, seed=7)
        rng = np.random.default_rng(8)
        stress = rng.standard_normal(20) * 0.04  # ~4% daily moves at the end
        model = GARCHModel(vol_model="GARCH", p=1, q=1, dist="normal")
        return model.fit(np.concatenate([calm, stress]), scale=100.0)

    def test_first_step_vol_matches_conditional_forecast(self):
        model = self._fit_ending_in_stress()
        paths = model.simulate(n_paths=20_000, horizon=5, random_state=0)
        sim_vol_step1 = float(np.std(paths[:, 0]))
        cond_vol_step1 = float(model.forecast(horizon=1)[0] / np.sqrt(252))
        res = model._result
        persistence = res.params["alpha[1]"] + res.params["beta[1]"]
        uncond_vol = float(np.sqrt(res.params["omega"] / (1 - persistence)) / 100.0)

        assert sim_vol_step1 == pytest.approx(cond_vol_step1, rel=0.05)
        assert sim_vol_step1 > 1.5 * uncond_vol

    def test_random_state_is_reproducible(self):
        model = self._fit_ending_in_stress()
        a = model.simulate(n_paths=50, horizon=10, random_state=123)
        b = model.simulate(n_paths=50, horizon=10, random_state=123)
        c = model.simulate(n_paths=50, horizon=10, random_state=124)
        np.testing.assert_array_equal(a, b)
        assert not np.array_equal(a, c)
