"""Tests for the distributional forecasters in synthquant.forecasters."""

from __future__ import annotations

import numpy as np
import pytest
from arch import arch_model
from scipy import stats

from synthquant.forecasters import (
    FilteredHistoricalSimulation,
    GaussianForecaster,
    GJRGarchForecaster,
    HARForecaster,
    HistoricalSimulation,
    LaplaceForecaster,
    PathModelForecaster,
    ReturnForecast,
)

from .conftest import GJR_TRUE, simulate_gjr

HORIZONS = (1, 5, 21)


@pytest.fixture(scope="module")
def gjr_data():
    return simulate_gjr(3000, seed=3)


@pytest.fixture(scope="module")
def fitted_skewt(gjr_data):
    r, _ = gjr_data
    return GJRGarchForecaster("skewt").fit(r[:2500])


class TestReturnForecast:
    def test_shape_validated(self):
        with pytest.raises(ValueError, match="shape"):
            ReturnForecast((1, 5), np.zeros((10, 3)))

    def test_accessors(self):
        s = np.column_stack([np.linspace(-1, 1, 101), np.linspace(-2, 2, 101)])
        fc = ReturnForecast((1, 5), s)
        np.testing.assert_array_equal(fc.at(5), s[:, 1])
        q = fc.quantiles([0.5])
        np.testing.assert_allclose(q, [[0.0, 0.0]], atol=1e-12)
        np.testing.assert_allclose(fc.price_quantiles(100.0, [0.5]), [[100.0, 100.0]])

    def test_horizons_sorted_and_validated(self):
        fc = LaplaceForecaster().predict(np.random.default_rng(0).normal(size=100), (21, 1, 5))
        assert fc.horizons == (1, 5, 21)
        with pytest.raises(ValueError):
            LaplaceForecaster().predict(np.ones(100), (0, 1))


class TestBaselines:
    def test_laplace_is_mle(self):
        rng = np.random.default_rng(1)
        r = stats.laplace.rvs(scale=0.01, size=5000, random_state=rng)
        fc = LaplaceForecaster(window=None)
        loc, scale = fc.fit_params(r)
        assert loc == 0.0
        assert scale == pytest.approx(np.mean(np.abs(r)))
        _, mle_scale = stats.laplace.fit(r, floc=0.0)
        assert scale == pytest.approx(mle_scale, rel=1e-6)
        med_loc, _ = LaplaceForecaster(window=None, loc="median").fit_params(r)
        assert med_loc == pytest.approx(np.median(r))

    def test_laplace_sqrt_time(self):
        r = stats.laplace.rvs(scale=0.01, size=1000, random_state=2)
        fc = LaplaceForecaster(window=504).predict(r, HORIZONS, n_samples=4000)
        b = np.mean(np.abs(r[-504:]))
        for h in HORIZONS:
            assert fc.at(h).std() == pytest.approx(np.sqrt(2) * b * np.sqrt(h), rel=0.02)

    def test_gaussian_volatility(self):
        rng = np.random.default_rng(3)
        r = rng.normal(0, 0.015, size=800)
        fc = GaussianForecaster(window=None).predict(r, HORIZONS, n_samples=4000)
        for h in HORIZONS:
            assert fc.at(h).std() == pytest.approx(np.std(r, ddof=1) * np.sqrt(h), rel=0.01)
            assert fc.at(h).mean() == pytest.approx(0.0, abs=1e-12)

    def test_ewma_reacts_faster_than_rolling(self):
        rng = np.random.default_rng(4)
        r = np.concatenate([rng.normal(0, 0.01, 500), rng.normal(0, 0.04, 10)])
        rolling = GaussianForecaster().volatility(r)
        ewma = GaussianForecaster(ewma_lambda=0.94).volatility(r)
        assert ewma > 2 * rolling
        assert GaussianForecaster(ewma_lambda=0.94).name == "ewma"

    def test_historical_bootstraps_window(self):
        r = np.arange(1, 1001, dtype=float) / 1e4
        fc = HistoricalSimulation(window=100).predict(r, (1, 2), n_samples=500, random_state=0)
        assert set(fc.at(1)).issubset(set(r[-100:]))
        assert fc.at(2).min() >= 2 * r[-100] - 1e-12

    def test_reproducible(self):
        r = np.random.default_rng(5).normal(size=600) / 100
        a = HistoricalSimulation().predict(r, HORIZONS, random_state=42).samples
        b = HistoricalSimulation().predict(r, HORIZONS, random_state=42).samples
        np.testing.assert_array_equal(a, b)


class TestGJRGarch:
    def test_unfitted_raises(self):
        with pytest.raises(RuntimeError, match="fit"):
            GJRGarchForecaster().predict(np.zeros(100), (1,))

    def test_recovers_leverage(self, fitted_skewt):
        p = fitted_skewt.params
        assert p["gamma"] > 0.05
        assert p["beta"] == pytest.approx(GJR_TRUE["beta"], abs=0.06)
        assert p["eta"] < 15  # fat tails detected (true df = 5)

    def test_conditional_variance_matches_arch(self, gjr_data):
        """Filtering the fit window reproduces arch's own one-step forecast."""
        r, _ = gjr_data
        window = r[:1500]
        model = GJRGarchForecaster("t").fit(window)
        res = arch_model(window * 100, mean="Zero", p=1, o=1, q=1, dist="t", rescale=False).fit(
            disp="off"
        )
        arch_var = float(res.forecast(horizon=1).variance.values[-1, 0]) / 100**2
        assert model.conditional_variance(window) == pytest.approx(arch_var, rel=1e-3)

    def test_tracks_true_conditional_variance(self, gjr_data, fitted_skewt):
        r, true_var = gjr_data
        est = [fitted_skewt.conditional_variance(r[t - 504 : t]) for t in range(2500, 3000, 25)]
        truth = true_var[2500:3000:25]
        assert np.corrcoef(est, truth)[0, 1] > 0.9

    def test_one_step_distribution(self, gjr_data, fitted_skewt):
        r, _ = gjr_data
        hist = r[:2600]
        fc = fitted_skewt.predict(hist, HORIZONS, n_samples=5000, random_state=0)
        sigma = np.sqrt(fitted_skewt.conditional_variance(hist))
        assert fc.at(1).std() == pytest.approx(sigma, rel=0.03)
        # Stratified first step: the 1% quantile matches the skew-t ppf closely
        dist, params = fitted_skewt._dist_obj, fitted_skewt._dist_params
        q01 = sigma * float(dist.ppf(0.01, params))
        assert np.quantile(fc.at(1), 0.01) == pytest.approx(q01, rel=0.03)

    def test_multi_step_variance_matches_arch(self, gjr_data):
        r, _ = gjr_data
        window = r[:1500]
        model = GJRGarchForecaster("normal").fit(window)
        fc = model.predict(window, (21,), n_samples=40_000, random_state=1)
        res = arch_model(window * 100, mean="Zero", p=1, o=1, q=1, rescale=False).fit(disp="off")
        var21 = float(res.forecast(horizon=21).variance.values[-1].sum()) / 100**2
        assert fc.at(21).var() == pytest.approx(var21, rel=0.05)

    def test_leverage_creates_negative_skew_at_horizon(self, fitted_skewt, gjr_data):
        r, _ = gjr_data
        fc = fitted_skewt.predict(r[:2500], (21,), n_samples=20_000, random_state=2)
        assert stats.skew(fc.at(21)) < -0.1

    def test_reproducible(self, fitted_skewt, gjr_data):
        r, _ = gjr_data
        a = fitted_skewt.predict(r[:2500], HORIZONS, random_state=7).samples
        b = fitted_skewt.predict(r[:2500], HORIZONS, random_state=7).samples
        np.testing.assert_array_equal(a, b)

    def test_names(self):
        assert GJRGarchForecaster("skewt").name == "gjr-skewt"
        assert GJRGarchForecaster("normal", asymmetric=False).name == "garch-normal"


class TestFHS:
    def test_innovations_bounded_by_residuals(self, gjr_data):
        r, _ = gjr_data
        model = FilteredHistoricalSimulation().fit(r[:1000])
        z = model._innovations(np.random.default_rng(0).random(10_000))
        resid = model._std_resid
        assert resid is not None
        assert z.min() >= resid.min() and z.max() <= resid.max()
        assert resid.mean() == pytest.approx(0.0, abs=1e-12)
        assert resid.std() == pytest.approx(1.0)

    def test_one_step_scale(self, gjr_data):
        r, _ = gjr_data
        model = FilteredHistoricalSimulation().fit(r[:1500])
        fc = model.predict(r[:1600], (1,), n_samples=5000, random_state=0)
        assert fc.at(1).std() == pytest.approx(
            np.sqrt(model.conditional_variance(r[:1600])), rel=0.03
        )


class TestHAR:
    def test_variance_tracks_truth(self, gjr_data):
        r, true_var = gjr_data
        model = HARForecaster().fit(r[:2000])
        est = [model.variance_forecast(r[:t], 1) for t in range(2000, 3000, 10)]
        assert np.corrcoef(est, true_var[2000:3000:10])[0, 1] > 0.6

    def test_standardized_distribution_is_calibrated_in_sample(self, gjr_data):
        r, _ = gjr_data
        model = HARForecaster().fit(r[:2000])
        _, z, _ = model._fit_horizon(5)
        assert z.mean() == pytest.approx(0.0, abs=1e-12)
        assert z.std() == pytest.approx(1.0, abs=0.25)

    def test_shapes_and_errors(self, gjr_data):
        r, _ = gjr_data
        model = HARForecaster()
        with pytest.raises(RuntimeError):
            model.predict(r[:100], (1,))
        model.fit(r[:600])
        fc = model.predict(r[:700], HORIZONS, n_samples=300)
        assert fc.samples.shape == (300, 3)
        with pytest.raises(ValueError, match="too short"):
            HARForecaster().fit(r[:60]).predict(r[:60], (21,))


class TestPathModelAdapter:
    def test_wraps_probabilistic_forecaster(self, gjr_data):
        from synthquant.forecast import ProbabilisticForecaster

        r, _ = gjr_data
        fc = PathModelForecaster(
            lambda: ProbabilisticForecaster(n_paths=300, random_state=0), name="regime"
        )
        with pytest.raises(RuntimeError):
            fc.predict(r[:500], (1,))
        out = fc.fit(r[:500]).predict(r[:500], (1, 5))
        assert out.samples.shape == (300, 2)
        assert np.all(np.isfinite(out.samples))
        assert out.at(5).std() > out.at(1).std()


class TestGJRSimulate:
    def test_predict_is_cumulative_simulate(self, fitted_skewt, gjr_data):
        r, _ = gjr_data
        hist = r[:2500]
        u = np.random.default_rng(0).random((400, 21))
        daily = fitted_skewt.simulate(hist, 21, uniforms=u)
        assert daily.shape == (400, 21)
        # same uniforms, same paths
        np.testing.assert_array_equal(daily, fitted_skewt.simulate(hist, 21, uniforms=u))
        fc = fitted_skewt.predict(hist, (1, 21), n_samples=4000, random_state=1)
        sim = fitted_skewt.simulate(hist, 21, n_samples=4000, random_state=2).sum(axis=1)
        assert sim.std() == pytest.approx(fc.at(21).std(), rel=0.08)

    def test_variance_scale_shocks_then_decays(self, fitted_skewt, gjr_data):
        r, _ = gjr_data
        hist = r[:2500]
        base = fitted_skewt.simulate(hist, 63, n_samples=20_000, random_state=0)
        shocked = fitted_skewt.simulate(
            hist, 63, n_samples=20_000, random_state=0, variance_scale=4.0
        )
        ratio_day1 = shocked[:, 0].std() / base[:, 0].std()
        ratio_day63 = shocked[:, -1].std() / base[:, -1].std()
        assert ratio_day1 == pytest.approx(2.0, rel=1e-9)  # same uniforms: exact
        assert 1.0 <= ratio_day63 < ratio_day1

    def test_uniforms_drive_correlation(self, gjr_data):
        """Correlated uniforms (Gaussian copula) give correlated asset paths."""
        r, _ = gjr_data
        a = GJRGarchForecaster("t").fit(r[:1500])
        b = GJRGarchForecaster("t").fit(r[1500:3000])
        rng = np.random.default_rng(3)
        z = rng.multivariate_normal([0, 0], [[1, 0.8], [0.8, 1]], size=(5000, 5))
        u = stats.norm.cdf(z)
        ra = a.simulate(r[:1500], 5, uniforms=u[:, :, 0])
        rb = b.simulate(r[1500:3000], 5, uniforms=u[:, :, 1])
        assert np.corrcoef(ra[:, 0], rb[:, 0])[0, 1] > 0.7

    def test_residuals_time_ordered_and_validation(self, fitted_skewt, gjr_data):
        z = fitted_skewt.standardized_residuals()
        assert z.size == 2500
        assert not np.all(np.diff(z) >= 0)  # not sorted
        assert np.nanstd(z) == pytest.approx(1.0)
        r, _ = gjr_data
        with pytest.raises(ValueError, match="variance_scale"):
            fitted_skewt.simulate(r[:100], 5, variance_scale=0.0)
        with pytest.raises(ValueError, match="shape"):
            fitted_skewt.simulate(r[:100], 5, uniforms=np.zeros((10, 4)))
