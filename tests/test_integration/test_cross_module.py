"""Integration test: Cross-module compatibility.

Ensures shapes, types, and contracts are consistent as data flows
between all major engine types and downstream analytics/strategy modules.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats as sp_stats

from augury.analytics.options import MCOptionPricer
from augury.analytics.risk_metrics import (
    conditional_var,
    max_drawdown_distribution,
    tail_ratio,
    value_at_risk,
)
from augury.simulation.correlation import CopulaSimulator
from augury.simulation.engines.gbm import GBMEngine
from augury.simulation.engines.heston import HestonEngine
from augury.simulation.engines.merton_jd import MertonJDEngine
from augury.simulation.engines.regime_switching import RegimeSwitchingEngine
from augury.simulation.variance_reduction import antithetic_variates
from augury.strategy.hedging import DeltaHedger, GammaHedger

# ── helpers ────────────────────────────────────────────────────────────────────


def _gbm_paths(n_paths: int = 5000, seed: int = 42) -> np.ndarray:
    return GBMEngine().simulate(
        n_paths=n_paths,
        n_steps=252,
        dt=1 / 252,
        S0=100.0,
        mu=0.07,
        sigma=0.20,
        random_state=seed,
    )


def _heston_paths(n_paths: int = 5000, seed: int = 42) -> np.ndarray:
    return HestonEngine().simulate(
        n_paths=n_paths,
        n_steps=252,
        dt=1 / 252,
        S0=100.0,
        v0=0.04,
        mu=0.05,
        kappa=2.0,
        theta=0.04,
        sigma_v=0.3,
        rho=-0.7,
        random_state=seed,
    )


def _merton_paths(n_paths: int = 5000, seed: int = 42) -> np.ndarray:
    return MertonJDEngine().simulate(
        n_paths=n_paths,
        n_steps=252,
        dt=1 / 252,
        S0=100.0,
        mu=0.07,
        sigma=0.15,
        lambda_j=2.0,
        mu_j=-0.04,
        sigma_j=0.08,
        random_state=seed,
    )


def _regime_paths(n_paths: int = 5000, seed: int = 42) -> np.ndarray:
    regime_params = [
        {"mu": 0.10, "sigma": 0.12},
        {"mu": -0.05, "sigma": 0.30},
    ]
    tm = np.array([[0.98, 0.02], [0.05, 0.95]])
    return RegimeSwitchingEngine(regime_params, tm, initial_regime=0).simulate(
        n_paths=n_paths,
        n_steps=252,
        dt=1 / 252,
        S0=100.0,
        random_state=seed,
    )


ALL_ENGINES = [
    pytest.param(_gbm_paths, id="GBM"),
    pytest.param(_heston_paths, id="Heston"),
    pytest.param(_merton_paths, id="MertonJD"),
    pytest.param(_regime_paths, id="RegimeSwitching"),
]


# ── Shape and dtype contract ──────────────────────────────────────────────────


class TestEngineOutputContract:
    """All engines must produce (n_paths, n_steps+1) float64 paths."""

    @pytest.mark.parametrize("engine_fn", ALL_ENGINES)
    def test_shape(self, engine_fn) -> None:
        paths = engine_fn(n_paths=100, seed=0)
        assert paths.shape == (100, 253)

    @pytest.mark.parametrize("engine_fn", ALL_ENGINES)
    def test_dtype(self, engine_fn) -> None:
        paths = engine_fn(n_paths=100, seed=0)
        assert paths.dtype == np.float64

    @pytest.mark.parametrize("engine_fn", ALL_ENGINES)
    def test_positive_prices(self, engine_fn) -> None:
        paths = engine_fn(n_paths=500, seed=42)
        assert np.all(np.isfinite(paths))
        assert np.all(paths > 0)


# ── Every engine works with risk metrics ──────────────────────────────────────


class TestEngineToRiskMetrics:
    @pytest.mark.parametrize("engine_fn", ALL_ENGINES)
    def test_var_finite(self, engine_fn) -> None:
        paths = engine_fn()
        var = value_at_risk(paths, alpha=0.05)
        assert np.isfinite(var)

    @pytest.mark.parametrize("engine_fn", ALL_ENGINES)
    def test_cvar_leq_var(self, engine_fn) -> None:
        paths = engine_fn()
        var = value_at_risk(paths, alpha=0.05)
        cvar = conditional_var(paths, alpha=0.05)
        assert cvar <= var + 1e-12

    @pytest.mark.parametrize("engine_fn", ALL_ENGINES)
    def test_drawdowns_nonpositive(self, engine_fn) -> None:
        paths = engine_fn()
        dd = max_drawdown_distribution(paths)
        assert dd.shape == (paths.shape[0],)
        assert np.all(dd <= 0)

    @pytest.mark.parametrize("engine_fn", ALL_ENGINES)
    def test_tail_ratio_positive(self, engine_fn) -> None:
        paths = engine_fn()
        tr = tail_ratio(paths)
        assert np.isfinite(tr)
        assert tr > 0


# ── Every engine works with option pricing ────────────────────────────────────


class TestEngineToOptionPricing:
    @pytest.mark.parametrize("engine_fn", ALL_ENGINES)
    def test_price_european_call(self, engine_fn) -> None:
        paths = engine_fn()
        pricer = MCOptionPricer()
        price = pricer.price_european(paths, K=100.0, T=1.0, r=0.04)
        assert np.isfinite(price)
        assert price > 0

    @pytest.mark.parametrize("engine_fn", ALL_ENGINES)
    def test_put_call_parity_approx(self, engine_fn) -> None:
        """C - P ≈ E[S_T]*exp(-rT) - K*exp(-rT) under physical measure MC."""
        paths = engine_fn()
        pricer = MCOptionPricer()
        K, T, r = 100.0, 1.0, 0.04
        call = pricer.price_european(paths, K=K, T=T, r=r, option_type="call")
        put = pricer.price_european(paths, K=K, T=T, r=r, option_type="put")
        # Under MC (physical measure), PCP: C - P = exp(-rT)*(E[S_T] - K)
        expected_diff = np.exp(-r * T) * (np.mean(paths[:, -1]) - K)
        np.testing.assert_allclose(call - put, expected_diff, atol=2.0)

    @pytest.mark.parametrize("engine_fn", ALL_ENGINES)
    def test_price_asian_nonnegative(self, engine_fn) -> None:
        paths = engine_fn()
        pricer = MCOptionPricer()
        price = pricer.price_asian(paths, K=100.0, T=1.0, r=0.04)
        assert price >= 0


# ── Every engine works with hedging ───────────────────────────────────────────


class TestEngineToHedging:
    @pytest.mark.parametrize("engine_fn", ALL_ENGINES)
    def test_delta_bounded(self, engine_fn) -> None:
        paths = engine_fn()
        dh = DeltaHedger()
        delta = dh.compute_hedge_ratio(paths, K=100.0, T=1.0, r=0.04)
        assert -1.5 < delta < 1.5  # delta should be roughly in [0,1] for calls

    @pytest.mark.parametrize("engine_fn", ALL_ENGINES)
    def test_gamma_finite(self, engine_fn) -> None:
        paths = engine_fn()
        gh = GammaHedger()
        gamma = gh.compute_gamma(paths, K=100.0, T=1.0, r=0.04)
        assert np.isfinite(gamma)
        assert gamma >= 0  # gamma is always non-negative


# ── Antithetic variates preserve engine contract ──────────────────────────────


class TestAntitheticCompatibility:
    @pytest.mark.parametrize("engine_fn", ALL_ENGINES)
    def test_antithetic_shape(self, engine_fn) -> None:
        paths = engine_fn(n_paths=200, seed=42)
        anti = antithetic_variates(paths)
        assert anti.shape[0] == 2 * paths.shape[0]
        assert anti.shape[1] == paths.shape[1]

    @pytest.mark.parametrize("engine_fn", ALL_ENGINES)
    def test_antithetic_still_works_with_var(self, engine_fn) -> None:
        paths = engine_fn(n_paths=200, seed=42)
        anti = antithetic_variates(paths)
        var = value_at_risk(anti, alpha=0.05)
        assert np.isfinite(var)


# ── Copula -> multi-asset backtest shape contract ─────────────────────────────


class TestCopulaIntegration:
    def test_gaussian_copula_shape(self) -> None:
        corr = np.array(
            [
                [1.0, 0.6, 0.3],
                [0.6, 1.0, 0.4],
                [0.3, 0.4, 1.0],
            ]
        )
        sim = CopulaSimulator(random_state=42)
        U = sim.simulate(n_paths=1000, n_assets=3, correlation_matrix=corr)
        assert U.shape == (1000, 3)
        assert np.all((U >= 0) & (U <= 1))

    def test_copula_marginals_roughly_uniform(self) -> None:
        corr = np.eye(3)
        sim = CopulaSimulator(random_state=42)
        U = sim.simulate(n_paths=5000, n_assets=3, correlation_matrix=corr)
        for i in range(3):
            _, p = sp_stats.kstest(U[:, i], "uniform")
            assert p > 0.01, f"Marginal {i} not uniform (p={p:.4f})"

    def test_correlated_returns_shape(self) -> None:
        corr = np.array([[1.0, 0.5], [0.5, 1.0]])
        marginals = [
            lambda u: sp_stats.norm.ppf(u, loc=0.0005, scale=0.01),
            lambda u: sp_stats.norm.ppf(u, loc=0.0003, scale=0.015),
        ]
        sim = CopulaSimulator(random_state=42)
        ret = sim.generate_correlated_returns(corr, marginals, n_paths=1000)
        assert ret.shape == (1000, 2)
        assert np.all(np.isfinite(ret))
