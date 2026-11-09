"""Tests for the default forecaster and the forecast_prices entry point."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import synthquant
from synthquant.forecasters import (
    DEFAULT_LOOKBACK,
    GJRGarchForecaster,
    LaplaceForecaster,
    default_forecaster,
    forecast_prices,
)

from .conftest import simulate_gjr


@pytest.fixture(scope="module")
def prices() -> pd.Series:
    r, _ = simulate_gjr(900, seed=21)
    p = 400.0 * np.exp(np.cumsum(r))
    return pd.Series(p, index=pd.bdate_range("2023-01-02", periods=p.size))


@pytest.fixture(scope="module")
def fc(prices):
    return forecast_prices(prices, horizon=21, n_samples=5000, random_state=0)


def test_default_is_gjr_skewt():
    model = default_forecaster()
    assert isinstance(model, GJRGarchForecaster)
    assert model.name == "gjr-skewt"
    assert model.mean == "zero"


def test_top_level_exports():
    assert synthquant.forecast_prices is forecast_prices
    assert synthquant.default_forecaster is default_forecaster


def test_uses_default_model_and_shapes(fc, prices):
    assert fc.model == "gjr-skewt"
    assert fc.horizon == 21
    assert fc.prices.shape == (5000, 21)
    assert fc.last_price == pytest.approx(prices.iloc[-1])
    assert fc.as_of == prices.index[-1]


def test_dates_are_following_business_days(fc, prices):
    assert fc.dates is not None
    assert len(fc.dates) == 21
    assert fc.dates[0] == prices.index[-1] + pd.offsets.BDay(1)
    assert all(d.weekday() < 5 for d in fc.dates)


def test_weekend_origin_and_daily_freq():
    p = pd.Series(
        100 * np.exp(np.cumsum(np.random.default_rng(0).normal(0, 0.01, 300))),
        index=pd.date_range(end="2025-10-26", periods=300, freq="D"),
    )
    assert p.index[-1].weekday() == 6  # Sunday
    out = forecast_prices(p, horizon=3, freq="D", forecaster=LaplaceForecaster())
    assert out.dates is not None
    assert list(out.dates) == list(pd.date_range(p.index[-1] + pd.Timedelta(days=1), periods=3))
    bday = forecast_prices(p, horizon=3, forecaster=LaplaceForecaster())
    assert bday.dates is not None and bday.dates[0].weekday() == 0  # Monday


def test_bands_monotone_and_widening(fc):
    b = fc.bands()
    assert list(b.columns) == ["q5", "q25", "q50", "q75", "q95"]
    assert (b.diff(axis=1).iloc[:, 1:] > 0).all().all()
    width = b["q95"] - b["q5"]
    assert width.iloc[-1] > 3 * width.iloc[0]


def test_risk_numbers_consistent(fc):
    var = fc.value_at_risk(0.05, 5)
    es = fc.expected_shortfall(0.05, 5)
    assert es < var < 0
    assert fc.prob_above(0.0) == 1.0
    assert 0.3 < fc.prob_above(fc.last_price, 21) < 0.7
    s = fc.summary()
    assert list(s.index) == [1, 5, 10, 21]
    assert s.loc[5, "var_95"] == pytest.approx(var)
    with pytest.raises(ValueError):
        fc.value_at_risk(0.05, 22)


def test_uses_only_lookback_window(prices):
    """Changing prices before the lookback window must not change the forecast."""
    altered = prices.copy()
    altered.iloc[: len(prices) - DEFAULT_LOOKBACK - 1] *= 1.5
    a = forecast_prices(prices, horizon=5, n_samples=500, random_state=1)
    b = forecast_prices(altered, horizon=5, n_samples=500, random_state=1)
    np.testing.assert_allclose(a.prices, b.prices)


def test_array_input_and_validation():
    p = 50 * np.exp(np.cumsum(np.random.default_rng(2).normal(0, 0.01, 400)))
    out = forecast_prices(p, horizon=5, forecaster=LaplaceForecaster())
    assert out.dates is None and out.as_of is None
    assert "date" not in out.summary().columns
    with pytest.raises(ValueError, match="positive"):
        forecast_prices(np.r_[p, -1.0])
    with pytest.raises(ValueError, match="at least"):
        forecast_prices(p[:50])
    with pytest.raises(ValueError):
        forecast_prices(p, horizon=0)


def test_annual_drift_shifts_log_returns(prices):
    base = forecast_prices(prices, horizon=21, n_samples=2000, random_state=3)
    tilted = forecast_prices(prices, horizon=21, n_samples=2000, random_state=3, annual_drift=0.252)
    shift = np.log(tilted.prices / base.prices)
    np.testing.assert_allclose(shift, np.broadcast_to(0.001 * np.arange(1, 22), shift.shape))
    assert tilted.prob_above(tilted.last_price, 21) > base.prob_above(base.last_price, 21)
    crypto = forecast_prices(
        prices, horizon=5, n_samples=200, random_state=3, annual_drift=0.365, periods_per_year=365
    )
    ref = forecast_prices(prices, horizon=5, n_samples=200, random_state=3)
    np.testing.assert_allclose(np.log(crypto.prices / ref.prices)[:, 0], 0.001)
