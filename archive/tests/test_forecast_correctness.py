"""Regression tests for forecasting correctness bugs found in the 2026-11 audit.

Covers: drift convexity double-counting, GARCH vol conditioning being
discarded/over-clamped, CRPS-optimal engine weights never being used,
Rough Bergomi variance units, and intraday path allocation.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from synthquant.ensemble_forecast import EnsembleForecaster
from synthquant.forecast import ProbabilisticForecaster
from synthquant.simulation.drift import (
    gbm_drift,
    heston_drift,
    merton_drift,
    rough_bergomi_drift,
)
from synthquant.simulation.engines.heston import HestonEngine
from synthquant.simulation.engines.merton_jd import MertonJDEngine
from synthquant.simulation.engines.regime_switching import RegimeSwitchingEngine
from synthquant.simulation.engines.rough_bergomi import RoughBergomiEngine

pytestmark = pytest.mark.filterwarnings("ignore")

N = 40_000
T_STEPS = 63
DT = 1 / 252
T = T_STEPS * DT


def _mean_log_return(paths: np.ndarray) -> float:
    return float(np.mean(np.log(paths[:, -1] / paths[:, 0])))


@pytest.fixture(scope="module")
def high_vol_iid() -> np.ndarray:
    """iid returns at ~64% annual vol, where a sigma^2/2 bias is large."""
    return np.random.default_rng(3).standard_normal(1500) * 0.04 + 0.0005


@pytest.fixture(scope="module")
def ends_in_stress() -> np.ndarray:
    """Calm history followed by a short burst of ~4% daily moves."""
    rng = np.random.default_rng(11)
    return np.concatenate([rng.standard_normal(1200) * 0.008, rng.standard_normal(15) * 0.04])


# ---------------------------------------------------------------------------
# Drift conversion: engines fed converted drift reproduce the target log mean
# ---------------------------------------------------------------------------


class TestDriftConversion:
    MU_LOG = 0.10  # annualised mean log return target

    def test_gbm(self) -> None:
        sigma = 0.5
        eng = RegimeSwitchingEngine(
            [{"mu": gbm_drift(self.MU_LOG, sigma), "sigma": sigma}], np.array([[1.0]]), 0
        )
        paths = eng.simulate(n_paths=N, n_steps=T_STEPS, dt=DT, random_state=0)
        assert _mean_log_return(paths) == pytest.approx(self.MU_LOG * T, abs=0.006)

    def test_heston_with_v0_away_from_theta(self) -> None:
        kw = {"v0": 0.25, "theta": 0.06, "kappa": 3.0}
        paths = HestonEngine().simulate(
            n_paths=N,
            n_steps=T_STEPS,
            dt=DT,
            mu=heston_drift(self.MU_LOG, horizon_years=T, **kw),
            sigma_v=0.4,
            rho=-0.6,
            random_state=0,
            **kw,
        )
        assert _mean_log_return(paths) == pytest.approx(self.MU_LOG * T, abs=0.006)

    def test_merton_with_jumps(self) -> None:
        jumps = {"lambda_j": 10.0, "mu_j": -0.05, "sigma_j": 0.05}
        sigma = 0.25
        paths = MertonJDEngine().simulate(
            n_paths=N,
            n_steps=T_STEPS,
            dt=DT,
            mu=merton_drift(self.MU_LOG, sigma, **jumps),
            sigma=sigma,
            random_state=0,
            **jumps,
        )
        assert _mean_log_return(paths) == pytest.approx(self.MU_LOG * T, abs=0.006)

    def test_rough_bergomi(self) -> None:
        xi = 0.09
        paths = RoughBergomiEngine().simulate(
            n_paths=N,
            n_steps=T_STEPS,
            dt=DT,
            xi=xi,
            mu=rough_bergomi_drift(self.MU_LOG, xi),
            random_state=0,
        )
        assert _mean_log_return(paths) == pytest.approx(self.MU_LOG * T, abs=0.006)


class TestForecasterDriftUnbiased:
    def test_probabilistic_forecaster_matches_sample_mean(self, high_vol_iid) -> None:
        """Previously biased by -sigma^2/2 * H (about -0.017 here)."""
        horizon = 21
        fc = ProbabilisticForecaster(n_paths=N, random_state=0).fit(high_vol_iid)
        res = fc.forecast(horizon=horizon)
        assert res.expected_return == pytest.approx(horizon * high_vol_iid.mean(), abs=0.006)

    @pytest.mark.parametrize("engine", ["heston", "merton_jd", "rough_bergomi"])
    def test_ensemble_single_regime_engines_hit_regime_mean(self, high_vol_iid, engine) -> None:
        """Each engine's simulated mean log return equals the regime's mean log return."""
        horizon = 21
        fc = EnsembleForecaster(n_paths=1000, random_state=0, use_rough_bergomi=True)
        fc.fit(high_vol_iid)
        cond = fc._conditioning(fc._current_proba(), horizon, DT)
        paths = fc._simulate_engine(engine, N, horizon, DT, 100.0, cond, seed=1)
        target = fc._calibrated[cond["regime"]].mu * horizon
        se = np.std(np.log(paths[:, -1] / 100)) / np.sqrt(N)
        assert _mean_log_return(paths) == pytest.approx(target, abs=max(5 * se, 0.003))


# ---------------------------------------------------------------------------
# GARCH conditioning
# ---------------------------------------------------------------------------


class TestGarchConditioning:
    """Contract: over the forecast horizon, the simulated variance matches
    GARCH's variance forecast from today's conditional variance (previously
    the scale was discarded in the ensemble and capped at 2x elsewhere)."""

    HORIZON = 5

    @classmethod
    def _garch_horizon_std(cls, fc) -> float:
        daily_var = fc._garch.forecast(horizon=cls.HORIZON) ** 2 / 252
        return float(np.sqrt(daily_var.sum()))

    def test_probabilistic_forecaster_spread_tracks_garch(self, ends_in_stress) -> None:
        fc = ProbabilisticForecaster(n_paths=N, random_state=0).fit(ends_in_stress)
        res = fc.forecast(horizon=self.HORIZON)
        assert res.return_std == pytest.approx(self._garch_horizon_std(fc), rel=0.15)

    def test_probabilistic_forecaster_reacts_to_recent_stress(self, ends_in_stress) -> None:
        calm = ends_in_stress[:-15]
        std_calm = (
            ProbabilisticForecaster(n_paths=N, random_state=0)
            .fit(calm)
            .forecast(horizon=self.HORIZON)
            .return_std
        )
        std_stress = (
            ProbabilisticForecaster(n_paths=N, random_state=0)
            .fit(ends_in_stress)
            .forecast(horizon=self.HORIZON)
            .return_std
        )
        assert std_stress > 1.5 * std_calm

    @pytest.mark.parametrize("engine", ["merton_jd", "regime_switching"])
    def test_ensemble_engines_track_garch(self, ends_in_stress, engine) -> None:
        fc = EnsembleForecaster(n_paths=1000, random_state=0).fit(ends_in_stress)
        cond = fc._conditioning(fc._current_proba(), self.HORIZON, DT)
        paths = fc._simulate_engine(engine, N, self.HORIZON, DT, 100.0, cond, seed=1)
        sim_std = float(np.std(np.log(paths[:, -1] / 100)))
        assert sim_std == pytest.approx(self._garch_horizon_std(fc), rel=0.15)

    def test_ensemble_vol_scale_near_one_without_vol_news(self) -> None:
        r = np.random.default_rng(5).standard_normal(1500) * 0.01
        fc = EnsembleForecaster(n_paths=1000, random_state=0).fit(r)
        meta = fc.forecast(horizon=21).metadata
        assert 0.75 < meta["vol_scale_regime"] < 1.33


# ---------------------------------------------------------------------------
# CRPS-optimal engine weights
# ---------------------------------------------------------------------------


class TestOptimizedEngineWeights:
    @pytest.fixture(scope="class")
    def optimized(self):
        returns = np.random.default_rng(9).standard_normal(1200) * 0.012
        fc = EnsembleForecaster(n_paths=1000, random_state=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = fc.optimize_engine_weights(returns, n_folds=6, horizon=10, n_eval_paths=200)
        return fc, out, returns

    def test_weights_are_a_distribution(self, optimized) -> None:
        _, out, _ = optimized
        w = out["optimal_weights"]
        assert np.all(w >= 0)
        assert w.sum() == pytest.approx(1.0)
        assert out["n_folds_used"] == 6

    def test_mixture_no_worse_than_best_single_engine(self, optimized) -> None:
        _, out, _ = optimized
        best_single = float(out["crps_by_fold"].mean(axis=0).min())
        assert out["mixture_crps"] <= best_single + 1e-9

    def test_forecast_uses_optimized_weights(self, optimized) -> None:
        fc, out, _ = optimized
        res = fc.forecast(horizon=5)
        assert res.metadata["weights_source"] == "optimized"
        used = res.metadata["engine_weights"]
        for name, w in zip(out["engine_names"], out["optimal_weights"], strict=True):
            assert used[name] == pytest.approx(w)

    def test_refit_on_full_sample_afterwards(self, optimized) -> None:
        fc, _, returns = optimized
        assert len(fc._returns) == len(returns)

    def test_raises_without_usable_folds(self) -> None:
        fc = EnsembleForecaster(n_paths=200, random_state=0)
        with pytest.raises(ValueError, match="No usable folds"):
            fc.optimize_engine_weights(np.zeros(100), n_folds=5, horizon=10)


# ---------------------------------------------------------------------------
# Units and allocation
# ---------------------------------------------------------------------------


class TestUnitsAndAllocation:
    def test_rough_bergomi_xi_is_annualised(self, high_vol_iid) -> None:
        fc = EnsembleForecaster(n_paths=500, random_state=0, use_rough_bergomi=True)
        fc.fit(high_vol_iid)
        for cp in fc._calibrated.values():
            assert cp.rough_bergomi["xi"] == pytest.approx(cp.sigma**2 * 252)

    def test_intraday_default_paths_does_not_crash(self) -> None:
        """forecast_intraday's default n_paths=100 used to allocate -100 paths."""
        r = np.random.default_rng(0).standard_normal(600) * 0.01
        fc = EnsembleForecaster(n_paths=500, random_state=0).fit(r)
        res = fc.forecast_intraday()
        assert res.paths.shape == (100, 289)

    @pytest.mark.parametrize("n_paths", [10, 100, 333, 10_000])
    def test_allocation_sums_to_n_and_is_non_negative(self, n_paths: int) -> None:
        fc = EnsembleForecaster(n_paths=n_paths)
        weights = {"heston": 0.6, "merton_jd": 0.35, "regime_switching": 0.05}
        alloc = fc._allocate_paths(weights, n_paths)
        assert sum(alloc.values()) == n_paths
        assert all(v >= 0 for v in alloc.values())
