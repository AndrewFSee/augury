"""Tests for financial feature engineering."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from synthquant.data.features import FeatureEngine


@pytest.fixture()
def engine() -> FeatureEngine:
    return FeatureEngine()


@pytest.fixture()
def ohlcv() -> pd.DataFrame:
    """Synthetic 300-row OHLCV DataFrame with DatetimeIndex."""
    rng = np.random.default_rng(42)
    n = 300
    dates = pd.date_range("2022-01-01", periods=n, freq="B")
    close = 100.0 * np.cumprod(1 + rng.normal(0.0003, 0.012, n))
    spread = rng.uniform(0.002, 0.006, n)
    return pd.DataFrame(
        {
            "open": close * (1 - spread / 2),
            "high": close * (1 + spread),
            "low": close * (1 - spread),
            "close": close,
            "volume": rng.integers(1_000_000, 5_000_000, n).astype(float),
        },
        index=dates,
    )


# ── Rolling Returns ───────────────────────────────────────────────────────────


def test_rolling_returns_length(engine: FeatureEngine, ohlcv: pd.DataFrame) -> None:
    """rolling_returns() output has same length as input."""
    result = engine.rolling_returns(ohlcv["close"])
    assert len(result) == len(ohlcv)


def test_rolling_returns_first_value_nan(engine: FeatureEngine, ohlcv: pd.DataFrame) -> None:
    """First value of rolling_returns(window=1) is NaN."""
    result = engine.rolling_returns(ohlcv["close"], window=1)
    assert np.isnan(result.iloc[0])


def test_rolling_returns_window_5(engine: FeatureEngine, ohlcv: pd.DataFrame) -> None:
    """rolling_returns(window=5) has first 5 values as NaN."""
    result = engine.rolling_returns(ohlcv["close"], window=5)
    assert np.all(np.isnan(result.iloc[:5]))


# ── Realized Volatility ───────────────────────────────────────────────────────


def test_realized_vol_close_to_close(engine: FeatureEngine, ohlcv: pd.DataFrame) -> None:
    """close_to_close realized vol returns a Series of correct length."""
    rv = engine.realized_volatility(ohlcv, window=21, estimator="close_to_close")
    assert isinstance(rv, pd.Series)
    assert len(rv) == len(ohlcv)


def test_realized_vol_non_negative(engine: FeatureEngine, ohlcv: pd.DataFrame) -> None:
    """Realized volatility values are non-negative where not NaN."""
    rv = engine.realized_volatility(ohlcv, window=21, estimator="close_to_close")
    valid = rv.dropna()
    assert np.all(valid >= 0)


def test_realized_vol_parkinson(engine: FeatureEngine, ohlcv: pd.DataFrame) -> None:
    """Parkinson estimator returns valid non-negative values."""
    rv = engine.realized_volatility(ohlcv, window=21, estimator="parkinson")
    valid = rv.dropna()
    assert np.all(valid >= 0)


def test_realized_vol_garman_klass(engine: FeatureEngine, ohlcv: pd.DataFrame) -> None:
    """Garman-Klass estimator returns a Series without errors."""
    rv = engine.realized_volatility(ohlcv, window=21, estimator="garman_klass")
    assert isinstance(rv, pd.Series)


def test_realized_vol_yang_zhang(engine: FeatureEngine, ohlcv: pd.DataFrame) -> None:
    """Yang-Zhang estimator returns a Series without errors."""
    rv = engine.realized_volatility(ohlcv, window=30, estimator="yang_zhang")
    assert isinstance(rv, pd.Series)


def test_realized_vol_unknown_estimator_raises(engine: FeatureEngine, ohlcv: pd.DataFrame) -> None:
    """Unknown estimator raises ValueError."""
    with pytest.raises(ValueError, match="estimator"):
        engine.realized_volatility(ohlcv, estimator="unknown")


def test_realized_vol_annualized_larger_than_non_annualized(
    engine: FeatureEngine, ohlcv: pd.DataFrame
) -> None:
    """Annualized vol is larger than non-annualized vol."""
    rv_ann = engine.realized_volatility(ohlcv, window=21, annualize=True)
    rv_raw = engine.realized_volatility(ohlcv, window=21, annualize=False)
    valid_ann = rv_ann.dropna()
    valid_raw = rv_raw.dropna()
    assert valid_ann.mean() > valid_raw.mean()


# ── RSI ───────────────────────────────────────────────────────────────────────


def test_rsi_length(engine: FeatureEngine, ohlcv: pd.DataFrame) -> None:
    """RSI output has same length as input."""
    result = engine.rsi(ohlcv["close"])
    assert len(result) == len(ohlcv)


def test_rsi_name(engine: FeatureEngine, ohlcv: pd.DataFrame) -> None:
    """RSI series is named rsi_{period}."""
    result = engine.rsi(ohlcv["close"], period=14)
    assert result.name == "rsi_14"


def test_rsi_bounded_0_100(engine: FeatureEngine, ohlcv: pd.DataFrame) -> None:
    """RSI values are in [0, 100] where not NaN."""
    result = engine.rsi(ohlcv["close"]).dropna()
    assert (result >= 0).all()
    assert (result <= 100).all()


def test_rsi_custom_period(engine: FeatureEngine, ohlcv: pd.DataFrame) -> None:
    """RSI with different period returns different values."""
    rsi_7 = engine.rsi(ohlcv["close"], period=7).dropna()
    rsi_21 = engine.rsi(ohlcv["close"], period=21).dropna()
    # Shorter period RSI should be more volatile
    assert rsi_7.std() > rsi_21.std()


# ── MACD ──────────────────────────────────────────────────────────────────────


def test_macd_returns_dataframe(engine: FeatureEngine, ohlcv: pd.DataFrame) -> None:
    """MACD returns a DataFrame with the expected columns."""
    result = engine.macd(ohlcv["close"])
    assert isinstance(result, pd.DataFrame)
    assert set(result.columns) == {"macd", "signal", "histogram"}


def test_macd_length(engine: FeatureEngine, ohlcv: pd.DataFrame) -> None:
    """MACD output has same length as input."""
    result = engine.macd(ohlcv["close"])
    assert len(result) == len(ohlcv)


def test_macd_histogram_is_difference(engine: FeatureEngine, ohlcv: pd.DataFrame) -> None:
    """Histogram = MACD - signal."""
    result = engine.macd(ohlcv["close"])
    expected = result["macd"] - result["signal"]
    pd.testing.assert_series_equal(result["histogram"], expected, check_names=False)


def test_macd_custom_periods(engine: FeatureEngine, ohlcv: pd.DataFrame) -> None:
    """MACD with custom parameters produces values."""
    result = engine.macd(ohlcv["close"], fast=8, slow=21, signal=5)
    assert result["macd"].dropna().shape[0] > 0


# ── Bollinger Bands ───────────────────────────────────────────────────────────


def test_bollinger_bands_columns(engine: FeatureEngine, ohlcv: pd.DataFrame) -> None:
    """Bollinger Bands returns the expected columns."""
    result = engine.bollinger_bands(ohlcv["close"])
    assert set(result.columns) == {"middle", "upper", "lower", "pct_b", "bandwidth"}


def test_bollinger_bands_length(engine: FeatureEngine, ohlcv: pd.DataFrame) -> None:
    """Bollinger Bands output has same length as input."""
    result = engine.bollinger_bands(ohlcv["close"])
    assert len(result) == len(ohlcv)


def test_bollinger_upper_above_lower(engine: FeatureEngine, ohlcv: pd.DataFrame) -> None:
    """Upper band is always above lower band where valid."""
    result = engine.bollinger_bands(ohlcv["close"]).dropna()
    assert (result["upper"] > result["lower"]).all()


def test_bollinger_middle_is_sma(engine: FeatureEngine, ohlcv: pd.DataFrame) -> None:
    """Middle band equals the rolling mean of close."""
    result = engine.bollinger_bands(ohlcv["close"], window=20)
    expected_middle = ohlcv["close"].rolling(20).mean()
    pd.testing.assert_series_equal(result["middle"], expected_middle, check_names=False)


def test_bollinger_bandwidth_positive(engine: FeatureEngine, ohlcv: pd.DataFrame) -> None:
    """Bandwidth is positive where valid."""
    result = engine.bollinger_bands(ohlcv["close"]).dropna()
    assert (result["bandwidth"] > 0).all()


def test_bollinger_pct_b_within_unit_interval(engine: FeatureEngine, ohlcv: pd.DataFrame) -> None:
    """%B values are generally within [0, 1] for typical price data."""
    result = engine.bollinger_bands(ohlcv["close"], window=20).dropna()
    # Most values should be in [0, 1]; allow some excursions for volatile data
    in_range = ((result["pct_b"] >= 0) & (result["pct_b"] <= 1)).mean()
    assert in_range > 0.85


# ── Rolling Skew ──────────────────────────────────────────────────────────────


def test_rolling_skew_length(engine: FeatureEngine, ohlcv: pd.DataFrame) -> None:
    """Rolling skew output has same length as input."""
    result = engine.rolling_skew(ohlcv["close"])
    assert len(result) == len(ohlcv)


def test_rolling_skew_name(engine: FeatureEngine, ohlcv: pd.DataFrame) -> None:
    """Series is named skew_{window}."""
    result = engine.rolling_skew(ohlcv["close"], window=30)
    assert result.name == "skew_30"


def test_rolling_skew_values_finite(engine: FeatureEngine, ohlcv: pd.DataFrame) -> None:
    """Non-NaN rolling skew values are finite."""
    result = engine.rolling_skew(ohlcv["close"]).dropna()
    assert np.all(np.isfinite(result))


def test_rolling_skew_symmetric_data() -> None:
    """Symmetric returns should have skewness near 0."""
    rng = np.random.default_rng(42)
    # Generate symmetric price series from normal returns
    returns = rng.normal(0, 0.01, 500)
    close = pd.Series(100.0 * np.cumprod(1 + returns))
    engine = FeatureEngine()
    result = engine.rolling_skew(close, window=100).dropna()
    assert abs(result.mean()) < 0.5


# ── Rolling Kurtosis ─────────────────────────────────────────────────────────


def test_rolling_kurtosis_length(engine: FeatureEngine, ohlcv: pd.DataFrame) -> None:
    """Rolling kurtosis output has same length as input."""
    result = engine.rolling_kurtosis(ohlcv["close"])
    assert len(result) == len(ohlcv)


def test_rolling_kurtosis_name(engine: FeatureEngine, ohlcv: pd.DataFrame) -> None:
    """Series is named kurt_{window}."""
    result = engine.rolling_kurtosis(ohlcv["close"], window=60)
    assert result.name == "kurt_60"


def test_rolling_kurtosis_values_finite(engine: FeatureEngine, ohlcv: pd.DataFrame) -> None:
    """Non-NaN rolling kurtosis values are finite."""
    result = engine.rolling_kurtosis(ohlcv["close"]).dropna()
    assert np.all(np.isfinite(result))


def test_rolling_kurtosis_normal_data_near_zero() -> None:
    """Normal returns should have excess kurtosis near 0."""
    rng = np.random.default_rng(42)
    returns = rng.normal(0, 0.01, 1000)
    close = pd.Series(100.0 * np.cumprod(1 + returns))
    engine = FeatureEngine()
    result = engine.rolling_kurtosis(close, window=200).dropna()
    # Pandas .kurt() computes excess kurtosis, should be near 0 for normal data
    assert abs(result.mean()) < 1.5
