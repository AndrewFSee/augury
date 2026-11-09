"""Drift conversion: engines fed converted drift reproduce the target mean log return.

Regression tests for the drift convexity double-count found in the 2026-11 audit.
"""

from __future__ import annotations

import numpy as np
import pytest

from synthquant.simulation.drift import gbm_drift, heston_drift, merton_drift
from synthquant.simulation.engines.heston import HestonEngine
from synthquant.simulation.engines.merton_jd import MertonJDEngine
from synthquant.simulation.engines.regime_switching import RegimeSwitchingEngine

N = 40_000
T_STEPS = 63
DT = 1 / 252
T = T_STEPS * DT


def _mean_log_return(paths: np.ndarray) -> float:
    return float(np.mean(np.log(paths[:, -1] / paths[:, 0])))


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
