"""Tests for the walk-forward evaluation harness."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
import pytest

from augury.evaluation import walk_forward
from augury.forecasters import (
    GaussianForecaster,
    GJRGarchForecaster,
    LaplaceForecaster,
    ReturnForecast,
    ReturnForecaster,
)


class Spy(ReturnForecaster):
    """Records exactly what the harness shows it; forecasts a point mass at 0."""

    name = "spy"

    def __init__(self, fail_first_fits: int = 0) -> None:
        self.fit_windows: list[np.ndarray] = []
        self.histories: list[np.ndarray] = []
        self.fail_first_fits = fail_first_fits

    def fit(self, returns: np.ndarray) -> Spy:
        if self.fail_first_fits > 0:
            self.fail_first_fits -= 1
            raise RuntimeError("boom")
        self.fit_windows.append(np.array(returns))
        return self

    def predict(
        self,
        history: np.ndarray,
        horizons: Sequence[int],
        n_samples: int = 2000,
        random_state: int | np.random.Generator | None = None,
    ) -> ReturnForecast:
        self.histories.append(np.array(history))
        return ReturnForecast(tuple(horizons), np.zeros((n_samples, len(horizons))))


@pytest.fixture
def ramp() -> pd.Series:
    """Returns equal to their own position, so leaks are easy to detect."""
    n = 300
    return pd.Series(np.arange(n, dtype=float), index=pd.bdate_range("2020-01-01", periods=n))


class TestNoLookahead:
    def test_history_ends_before_target(self, ramp):
        spy = Spy()
        res = walk_forward(ramp, [spy], horizons=(1, 5), lookback=100, refit_every=10, n_samples=10)
        recs = res.records
        origins = recs["origin_idx"].unique()
        assert len(spy.histories) == len(origins)
        for t, hist in zip(origins, spy.histories, strict=True):
            np.testing.assert_array_equal(hist, np.arange(t - 100, t))
        # Target for horizon h at origin t is the sum of returns t .. t+h-1
        row = recs[(recs["origin_idx"] == 150) & (recs["horizon"] == 5)].iloc[0]
        assert row["y"] == sum(range(150, 155))
        # Origin label is the date of the last observed return
        assert row["origin"] == ramp.index[149]

    def test_fit_windows_are_rolling_and_on_schedule(self, ramp):
        spy = Spy()
        walk_forward(ramp, [spy], horizons=(1,), lookback=100, refit_every=25, n_samples=10)
        assert len(spy.fit_windows) == int(np.ceil(200 / 25))
        for k, w in enumerate(spy.fit_windows):
            t = 100 + 25 * k
            np.testing.assert_array_equal(w, np.arange(t - 100, t))

    def test_horizons_beyond_data_are_dropped(self, ramp):
        res = walk_forward(ramp, [Spy()], horizons=(1, 21), lookback=100, n_samples=10)
        recs = res.records
        assert recs[recs["horizon"] == 1]["origin_idx"].max() == 299
        assert recs[recs["horizon"] == 21]["origin_idx"].max() == 279


class TestHarnessMechanics:
    def test_eval_every_and_start(self, ramp):
        res = walk_forward(
            ramp,
            [Spy()],
            horizons=(1,),
            lookback=100,
            eval_every=10,
            n_samples=10,
            start="2020-07-01",
        )
        idx = res.records["origin_idx"].to_numpy()
        assert np.all(np.diff(idx) == 10)
        assert res.records["origin"].min() >= pd.Timestamp("2020-07-01")

    def test_fit_failures_are_skipped_and_counted(self, ramp):
        spy = Spy(fail_first_fits=2)
        res = walk_forward(ramp, [spy], horizons=(1,), lookback=100, refit_every=50, n_samples=10)
        assert res.fit_failures["spy"] == 2
        assert res.records["origin_idx"].min() == 102  # first two origins had no model

    def test_duplicate_names_rejected_mapping_accepted(self, ramp):
        with pytest.raises(ValueError, match="Duplicate"):
            walk_forward(ramp, [Spy(), Spy()], lookback=100)
        res = walk_forward(ramp, {"a": Spy(), "b": Spy()}, horizons=(1,), lookback=100, n_samples=5)
        assert res.models == ["a", "b"]
        assert res.baseline == "a"

    def test_array_input(self):
        r = np.random.default_rng(0).normal(0, 0.01, 400)
        res = walk_forward(r, [LaplaceForecaster()], horizons=(1,), lookback=200, n_samples=100)
        assert len(res.records) == 200
        assert res.records["origin"].iloc[0] == 199


class TestScoring:
    def test_calibrated_model_passes(self):
        """A correctly specified model shows nominal coverage and passes VaR tests."""
        rng = np.random.default_rng(1)
        r = pd.Series(rng.normal(0, 0.01, 2500))
        res = walk_forward(
            r, [GaussianForecaster(window=None)], horizons=(1, 5), lookback=500, n_samples=2000
        )
        s = res.summary().set_index("horizon")
        assert s.loc[1, "cov_90"] == pytest.approx(0.9, abs=0.02)
        assert s.loc[5, "cov_50"] == pytest.approx(0.5, abs=0.05)
        assert s.loc[1, "pit_ks_p"] > 0.01
        assert np.isnan(s.loc[5, "pit_ks_p"])  # overlapping: test not reported
        vb = res.var_backtest()
        assert (vb["kupiec_p"] > 0.01).all()
        assert (vb["christoffersen_p"] > 0.01).all()

    def test_crps_values_match_metric(self):
        r = pd.Series(np.random.default_rng(2).normal(0, 0.01, 260))
        res = walk_forward(r, [LaplaceForecaster()], horizons=(1,), lookback=250, n_samples=500)
        from augury.evaluation import crps_ensemble

        row = res.records.iloc[3]
        t = int(row["origin_idx"])
        fc = LaplaceForecaster().predict(r.to_numpy()[t - 250 : t], (1,), n_samples=500)
        assert row["crps"] == pytest.approx(float(crps_ensemble(fc.at(1), r.iloc[t])))

    def test_volatility_model_beats_rolling_on_garch_data(self, gjr_series):
        res = walk_forward(
            gjr_series,
            [GaussianForecaster(), GJRGarchForecaster("t")],
            horizons=(1, 5),
            lookback=504,
            refit_every=63,
            n_samples=1000,
        )
        s = res.summary().set_index(["model", "horizon"])
        assert s.loc[("gaussian", 1), "skill"] == 0.0
        assert s.loc[("gjr-t", 1), "skill"] > 0.005
        assert s.loc[("gjr-t", 1), "dm_p"] < 0.01
        vb = res.var_backtest().set_index(["model", "alpha"])
        # Rolling vol ignores clustering, so its violations bunch up
        assert (
            vb.loc[("gaussian", 0.05), "christoffersen_p"]
            < vb.loc[("gjr-t", 0.05), "christoffersen_p"]
        )

    def test_non_overlapping_pit_reported_with_eval_every(self, gjr_series):
        res = walk_forward(
            gjr_series, [LaplaceForecaster()], horizons=(5,), eval_every=5, n_samples=200
        )
        assert not np.isnan(res.summary()["pit_ks_p"].iloc[0])
