"""Integration test: Full pipeline from synthetic data to backtest results.

Mirrors the flow of examples/06_full_pipeline.py, validating that every
module composes correctly and produces statistically coherent outputs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from augury.analytics.risk_metrics import (
    conditional_var,
    max_drawdown_distribution,
    value_at_risk,
)
from augury.data.features import FeatureEngine
from augury.regime.hmm import HMMRegimeDetector
from augury.simulation.engines.gbm import GBMEngine
from augury.simulation.engines.regime_switching import RegimeSwitchingEngine
from augury.strategy.allocation import RiskParityAllocator
from augury.strategy.backtest import WalkForwardBacktest
from augury.strategy.signals import RegimeSignalGenerator
from augury.strategy.sizing import KellyCriterion, RiskParitySizer

# ── Shared synthetic data ─────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def multi_asset_prices() -> pd.DataFrame:
    """Create correlated 4-asset price history with controlled regime structure."""
    rng = np.random.default_rng(42)
    n = 1500
    dates = pd.date_range("2018-01-01", periods=n, freq="B")
    assets = ["SPY", "QQQ", "GLD", "TLT"]
    mus = [0.08, 0.12, 0.03, 0.02]
    sigmas = [0.18, 0.22, 0.14, 0.10]
    corr = np.array(
        [
            [1.00, 0.85, -0.20, -0.40],
            [0.85, 1.00, -0.15, -0.35],
            [-0.20, -0.15, 1.00, 0.10],
            [-0.40, -0.35, 0.10, 1.00],
        ]
    )
    L = np.linalg.cholesky(corr)
    Z = rng.standard_normal((n, 4)) @ L.T
    ret_matrix = np.column_stack(
        [Z[:, i] * sigmas[i] / np.sqrt(252) + mus[i] / 252 for i in range(4)]
    )
    price_matrix = 100.0 * np.cumprod(1 + ret_matrix, axis=0)
    return pd.DataFrame(price_matrix, index=dates, columns=assets)


@pytest.fixture(scope="module")
def spy_returns(multi_asset_prices) -> np.ndarray:
    fe = FeatureEngine()
    returns_series = fe.rolling_returns(multi_asset_prices["SPY"], window=1).dropna()
    return returns_series.values


# ── 1. Feature engineering → Regime detection ─────────────────────────────────


class TestFeaturesToRegimes:
    def test_feature_engine_produces_valid_returns(self, spy_returns) -> None:
        assert len(spy_returns) > 1000
        assert np.all(np.isfinite(spy_returns))
        # Daily log returns should be small
        assert np.abs(spy_returns).max() < 0.5

    def test_hmm_fits_on_feature_returns(self, spy_returns) -> None:
        detector = HMMRegimeDetector(n_components=2, random_state=42)
        detector.fit(spy_returns[:1000])
        labels = detector.predict(spy_returns)
        assert labels.shape == spy_returns.shape
        assert set(np.unique(labels)).issubset({0, 1})

    def test_regime_params_are_distinct(self, spy_returns) -> None:
        """Detected regimes should have meaningfully different volatilities."""
        detector = HMMRegimeDetector(n_components=2, random_state=42)
        detector.fit(spy_returns[:1000])
        params = detector.get_regime_params()
        sigmas = [params[i]["sigma"] for i in range(2)]
        ratio = max(sigmas) / min(sigmas)
        assert ratio > 1.3, f"Regime sigmas not distinct: {sigmas}"


# ── 2. Regime detection → Monte Carlo simulation ─────────────────────────────


class TestRegimesToSimulation:
    def test_regime_params_feed_into_mc(self, spy_returns) -> None:
        """Calibrated regime params produce valid MC paths."""
        detector = HMMRegimeDetector(n_components=2, random_state=42)
        detector.fit(spy_returns[:1000])
        params = detector.get_regime_params()
        labels = detector.predict(spy_returns)

        rs_params = [
            {
                "mu": params[i]["mu"] * 252,
                "sigma": params[i]["sigma"] * np.sqrt(252),
            }
            for i in range(2)
        ]
        transition = np.array([[0.97, 0.03], [0.05, 0.95]])
        engine = RegimeSwitchingEngine(rs_params, transition, initial_regime=int(labels[-1]))

        paths = engine.simulate(n_paths=5000, n_steps=252, dt=1 / 252, S0=100.0, random_state=42)
        assert paths.shape == (5000, 253)
        assert np.all(paths > 0)
        assert np.all(np.isfinite(paths))

    def test_mc_terminal_distribution_plausible(self, spy_returns) -> None:
        """Terminal prices shouldn't all be the same or extremely wild."""
        detector = HMMRegimeDetector(n_components=2, random_state=42)
        detector.fit(spy_returns[:1000])
        params = detector.get_regime_params()

        rs_params = [
            {"mu": params[i]["mu"] * 252, "sigma": params[i]["sigma"] * np.sqrt(252)}
            for i in range(2)
        ]
        transition = np.array([[0.97, 0.03], [0.05, 0.95]])
        engine = RegimeSwitchingEngine(rs_params, transition, initial_regime=0)
        paths = engine.simulate(n_paths=5000, n_steps=252, dt=1 / 252, S0=100.0, random_state=42)

        terminal = paths[:, -1]
        # Spread should be meaningful
        assert terminal.std() > 1.0
        # Median should be in a plausible range around S0=100
        assert 30 < np.median(terminal) < 300


# ── 3. Simulation paths → Risk analytics ─────────────────────────────────────


class TestSimulationToRisk:
    @pytest.fixture()
    def mc_paths(self) -> np.ndarray:
        engine = GBMEngine()
        return engine.simulate(
            n_paths=10_000,
            n_steps=252,
            dt=1 / 252,
            S0=100.0,
            mu=0.07,
            sigma=0.20,
            random_state=42,
        )

    def test_var_cvar_ordering(self, mc_paths) -> None:
        """CVaR should be at least as extreme as VaR."""
        var = value_at_risk(mc_paths, alpha=0.05)
        cvar = conditional_var(mc_paths, alpha=0.05)
        assert cvar <= var, f"CVaR ({cvar}) should be <= VaR ({var})"

    def test_var_scales_with_confidence(self, mc_paths) -> None:
        """99% VaR should be more extreme than 95% VaR."""
        var_95 = value_at_risk(mc_paths, alpha=0.05)
        var_99 = value_at_risk(mc_paths, alpha=0.01)
        assert var_99 < var_95

    def test_drawdowns_are_negative(self, mc_paths) -> None:
        dds = max_drawdown_distribution(mc_paths)
        assert np.all(dds <= 0)
        assert np.all(np.isfinite(dds))

    def test_risk_metrics_from_regime_switching_paths(self, spy_returns) -> None:
        """Full chain: regime → MC → risk metrics all produce valid numbers."""
        detector = HMMRegimeDetector(n_components=2, random_state=42)
        detector.fit(spy_returns[:1000])
        params = detector.get_regime_params()

        rs_params = [
            {"mu": params[i]["mu"] * 252, "sigma": params[i]["sigma"] * np.sqrt(252)}
            for i in range(2)
        ]
        engine = RegimeSwitchingEngine(
            rs_params,
            np.array([[0.97, 0.03], [0.05, 0.95]]),
            initial_regime=0,
        )
        paths = engine.simulate(n_paths=5000, n_steps=252, dt=1 / 252, S0=100.0, random_state=7)

        var = value_at_risk(paths, alpha=0.05)
        cvar = conditional_var(paths, alpha=0.05)
        assert isinstance(var, float) and np.isfinite(var)
        assert isinstance(cvar, float) and np.isfinite(cvar)
        assert cvar <= var


# ── 4. Regime signals → Sizing → Allocation ──────────────────────────────────


class TestSignalsToAllocation:
    def test_regime_signals_valid(self, spy_returns) -> None:
        detector = HMMRegimeDetector(n_components=2, random_state=42)
        detector.fit(spy_returns[:1000])
        labels = detector.predict(spy_returns)

        gen = RegimeSignalGenerator(regime_signal_map={0: 1, 1: 0})
        signals = gen.generate(labels)

        assert signals["signal"].shape == labels.shape
        assert set(np.unique(signals["signal"])).issubset({-1, 0, 1})
        assert signals["entry"].dtype == bool
        assert signals["exit"].dtype == bool

    def test_kelly_sizing_from_returns(self, spy_returns) -> None:
        kelly = KellyCriterion()
        fk = kelly.full_kelly(spy_returns)
        hk = kelly.fractional_kelly(spy_returns, fraction=0.5)
        assert np.isfinite(fk)
        assert abs(hk) < abs(fk) + 1e-12

    def test_risk_parity_weights_sum_to_one(self, multi_asset_prices) -> None:
        log_rets = np.log(multi_asset_prices / multi_asset_prices.shift(1)).dropna().values
        sizer = RiskParitySizer()
        weights = sizer.compute_weights(log_rets, target_vol=0.10)
        np.testing.assert_allclose(weights.sum(), 1.0, atol=1e-6)
        assert np.all(weights >= 0)

    def test_erc_allocator_from_covariance(self, multi_asset_prices) -> None:
        log_rets = np.log(multi_asset_prices / multi_asset_prices.shift(1)).dropna().values
        cov = np.cov(log_rets.T) * 252
        allocator = RiskParityAllocator()
        weights = allocator.allocate(cov)
        np.testing.assert_allclose(weights.sum(), 1.0, atol=1e-6)
        assert np.all(weights > 0)


# ── 5. Full pipeline → Walk-forward backtest ──────────────────────────────────


class TestFullBacktest:
    def test_walk_forward_completes(self, multi_asset_prices) -> None:
        def equal_weight_strategy(train_prices: pd.DataFrame) -> np.ndarray:
            return np.ones(train_prices.shape[1]) / train_prices.shape[1]

        bt = WalkForwardBacktest(train_window=252, rebalance_freq=21, transaction_cost=0.001)
        results = bt.run(multi_asset_prices, equal_weight_strategy)

        assert len(results.pnl) > 0
        assert np.isfinite(results.sharpe)
        assert np.isfinite(results.total_return)
        assert results.max_drawdown <= 0
        assert results.n_trades > 0

    def test_backtest_with_risk_parity_strategy(self, multi_asset_prices) -> None:
        """Use RiskParityAllocator as the strategy function."""
        allocator = RiskParityAllocator()

        def rp_strategy(train_prices: pd.DataFrame) -> np.ndarray:
            log_rets = np.log(train_prices / train_prices.shift(1)).dropna().values
            cov = np.cov(log_rets.T) * 252
            return allocator.allocate(cov)

        bt = WalkForwardBacktest(train_window=252, rebalance_freq=63, transaction_cost=0.001)
        results = bt.run(multi_asset_prices, rp_strategy)

        assert len(results.pnl) > 0
        assert results.n_trades >= 1
        # Weights should be non-trivial
        for w in results.weights_history:
            assert np.all(w >= 0)
            np.testing.assert_allclose(w.sum(), 1.0, atol=1e-6)
