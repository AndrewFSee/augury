"""Tests for volatility views: target_vol and ImpliedVolForecaster."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from augury.evaluation import walk_forward
from augury.forecasters import GJRGarchForecaster, ImpliedVolForecaster, forecast_prices


@pytest.fixture(scope="module")
def returns() -> pd.Series:
    rng = np.random.default_rng(7)
    n = 900
    vol = 0.01 * np.exp(0.4 * np.sin(np.arange(n) / 60))
    idx = pd.bdate_range("2020-01-01", periods=n)
    return pd.Series(rng.standard_t(6, n) / np.sqrt(1.5) * vol, index=idx)


def test_variance_scale_for_hits_target(returns):
    x = returns.to_numpy()
    m = GJRGarchForecaster().fit(x)
    for target_vol in (0.08, 0.2, 0.5):
        target = target_vol**2 * 21 / 252
        s = m.variance_scale_for(x, 21, target)
        assert m.expected_variance(x, 21, variance_scale=s).sum() == pytest.approx(target)


def test_variance_scale_is_floored(returns):
    x = returns.to_numpy()
    m = GJRGarchForecaster().fit(x)
    assert m.variance_scale_for(x, 21, 1e-12) == pytest.approx(0.01)


def test_forecast_prices_target_vol(returns):
    prices = 100 * np.exp(returns.cumsum())
    fc = forecast_prices(prices, 21, target_vol=0.3, n_samples=40_000, random_state=0)
    realized = fc.returns.samples[:, -1].std() * np.sqrt(252 / 21)
    assert realized == pytest.approx(0.3, rel=0.03)
    base = forecast_prices(prices, 21, n_samples=40_000, random_state=0)
    assert base.returns.samples[:, -1].std() * np.sqrt(252 / 21) != pytest.approx(0.3, rel=0.1)


def test_target_vol_works_for_any_forecaster(returns):
    from augury.forecasters import GaussianForecaster

    prices = 100 * np.exp(returns.cumsum())
    fc = forecast_prices(prices, 5, forecaster=GaussianForecaster(), target_vol=0.2)
    assert fc.returns.samples[:, -1].std() * np.sqrt(252 / 5) == pytest.approx(0.2)


def test_target_vol_scales_all_horizons_equally(returns):
    prices = 100 * np.exp(returns.cumsum())
    base = forecast_prices(prices, 21, n_samples=20_000, random_state=3)
    view = forecast_prices(prices, 21, target_vol=0.4, n_samples=20_000, random_state=3)
    ratio = view.returns.samples.std(axis=0) / base.returns.samples.std(axis=0)
    np.testing.assert_allclose(ratio, ratio[-1], rtol=1e-9)


def _iv(returns: pd.Series, ratio: float) -> pd.Series:
    """Implied vol = ratio x the true 21-day-ahead realized vol, plus noise."""
    rv = (returns**2)[::-1].rolling(21).sum()[::-1].shift(-1)
    iv = ratio * np.sqrt(rv * 252 / 21)
    return iv.ffill().bfill()


def test_calibrated_k_recovers_variance_premium(returns):
    m = ImpliedVolForecaster(_iv(returns, 1.5), weight=1.0, calibrate=True).fit(returns)
    assert m.k == pytest.approx(1 / 1.5**2, rel=0.05)


def test_weight_zero_matches_base(returns):
    iv = _iv(returns, 1.2)
    a = ImpliedVolForecaster(iv, weight=0.0).fit(returns)
    b = GJRGarchForecaster().fit(returns.to_numpy())
    fa = a.predict(returns, (1, 21), n_samples=500, random_state=1)
    fb = b.predict(returns.to_numpy(), (1, 21), n_samples=500, random_state=1)
    np.testing.assert_allclose(fa.samples, fb.samples)


@pytest.mark.parametrize("scaling", ["proportional", "start"])
def test_weight_one_matches_scaled_implied_variance(returns, scaling):
    iv = _iv(returns, 1.3)
    m = ImpliedVolForecaster(iv, weight=1.0, scaling=scaling, calibrate=True).fit(returns)
    fc = m.predict(returns, (1, 21), n_samples=40_000, random_state=0)
    x = returns.to_numpy()
    want = m.k * iv.loc[returns.index[-1]] ** 2 * 21 / 252
    garch = m.base.expected_variance(x, 21)
    if scaling == "start":
        got = m.base.expected_variance(x, 21, variance_scale=m.last_scale).sum()
    else:
        got = m.last_scale * garch.sum()
        # every horizon moves by the same factor
        assert fc.samples[:, 0].var() == pytest.approx(m.last_scale * garch[0], rel=0.05)
    assert got == pytest.approx(want)
    assert fc.samples[:, 1].var() == pytest.approx(want, rel=0.05)


def test_future_implied_vol_is_not_used(returns):
    iv = _iv(returns, 1.2)
    hist = returns.iloc[:600]
    shocked = iv.copy()
    shocked.loc[shocked.index > hist.index[-1]] *= 5
    a = ImpliedVolForecaster(iv).fit(hist).predict(hist, (5,), n_samples=300, random_state=0)
    b = ImpliedVolForecaster(shocked).fit(hist).predict(hist, (5,), n_samples=300, random_state=0)
    np.testing.assert_allclose(a.samples, b.samples)


def test_needs_dated_returns(returns):
    m = ImpliedVolForecaster(_iv(returns, 1.2))
    with pytest.raises(TypeError, match="DatetimeIndex"):
        m.fit(returns.to_numpy())
    with pytest.raises(ValueError, match="DatetimeIndex"):
        walk_forward(returns.to_numpy(), [m], horizons=(1,), lookback=504, eval_every=100)


def test_walk_forward_runs_with_implied_vol(returns):
    iv = _iv(returns, 1.2)
    res = walk_forward(
        returns,
        [GJRGarchForecaster(), ImpliedVolForecaster(iv, weight=0.5, calibrate=True)],
        horizons=(1, 5),
        lookback=504,
        eval_every=20,
        n_samples=500,
    )
    s = res.summary()
    assert set(s["model"]) == {"gjr-skewt", "gjr-skewt+iv0.5"}
    # IV here is a scaled oracle of future vol, so blending must help
    assert s.loc[(s.model == "gjr-skewt+iv0.5") & (s.horizon == 5), "skill"].item() > 0


def test_forecast_prices_with_implied_vol(returns):
    prices = 100 * np.exp(returns.cumsum())
    fc = forecast_prices(prices, 10, forecaster=ImpliedVolForecaster(_iv(returns, 1.2)))
    assert fc.model == "gjr-skewt+iv0.5"
    with pytest.raises(ValueError, match="DatetimeIndex"):
        forecast_prices(prices.to_numpy(), 10, forecaster=ImpliedVolForecaster(_iv(returns, 1.2)))
