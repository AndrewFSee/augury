"""Tests for Markov-Switching GARCH regime detector."""

from __future__ import annotations

import numpy as np
import pytest

from augury.regime.ms_garch import MarkovSwitchingGARCH


@pytest.fixture()
def two_regime_returns() -> np.ndarray:
    """Synthetic returns with two distinct regimes (low-vol then high-vol)."""
    rng = np.random.default_rng(42)
    low_vol = rng.normal(0.001, 0.005, 200)  # calm market
    high_vol = rng.normal(-0.001, 0.020, 200)  # stressed market
    return np.concatenate([low_vol, high_vol])


@pytest.fixture()
def fitted_model(two_regime_returns) -> MarkovSwitchingGARCH:
    """Pre-fitted 2-regime model."""
    model = MarkovSwitchingGARCH(k_regimes=2)
    model.fit(two_regime_returns)
    return model


def test_constructor_defaults() -> None:
    """Default parameters are stored correctly."""
    m = MarkovSwitchingGARCH()
    assert m.k_regimes == 2
    assert m.switching_variance is True
    assert m.switching_mean is True


def test_constructor_custom() -> None:
    """Custom parameters are stored correctly."""
    m = MarkovSwitchingGARCH(k_regimes=3, switching_variance=False, switching_mean=False)
    assert m.k_regimes == 3
    assert m.switching_variance is False
    assert m.switching_mean is False


def test_fit_returns_self(two_regime_returns) -> None:
    """fit() returns self for chaining."""
    m = MarkovSwitchingGARCH()
    result = m.fit(two_regime_returns)
    assert result is m


def test_predict_before_fit_raises() -> None:
    """predict() before fit() raises RuntimeError."""
    m = MarkovSwitchingGARCH()
    with pytest.raises(RuntimeError, match="not been fitted"):
        m.predict(np.zeros(100))


def test_predict_proba_before_fit_raises() -> None:
    """predict_proba() before fit() raises RuntimeError."""
    m = MarkovSwitchingGARCH()
    with pytest.raises(RuntimeError, match="not been fitted"):
        m.predict_proba(np.zeros(100))


def test_get_regime_params_before_fit_raises() -> None:
    """get_regime_params() before fit() raises RuntimeError."""
    m = MarkovSwitchingGARCH()
    with pytest.raises(RuntimeError, match="not been fitted"):
        m.get_regime_params()


def test_predict_returns_correct_length(fitted_model, two_regime_returns) -> None:
    labels = fitted_model.predict(two_regime_returns)
    assert len(labels) == len(two_regime_returns)


def test_predict_labels_valid_range(fitted_model, two_regime_returns) -> None:
    labels = fitted_model.predict(two_regime_returns)
    assert set(np.unique(labels)).issubset({0, 1})


def test_predict_proba_shape(fitted_model, two_regime_returns) -> None:
    proba = fitted_model.predict_proba(two_regime_returns)
    assert proba.shape == (len(two_regime_returns), 2)


def test_predict_proba_sums_to_one(fitted_model, two_regime_returns) -> None:
    proba = fitted_model.predict_proba(two_regime_returns)
    row_sums = proba.sum(axis=1)
    np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)


def test_predict_proba_in_unit_interval(fitted_model, two_regime_returns) -> None:
    proba = fitted_model.predict_proba(two_regime_returns)
    assert np.all(proba >= -1e-10)
    assert np.all(proba <= 1 + 1e-10)


def test_get_regime_params_structure(fitted_model) -> None:
    """Returns dict with correct structure for each regime."""
    params = fitted_model.get_regime_params()
    assert len(params) == 2
    for i in range(2):
        assert "mu" in params[i]
        assert "sigma" in params[i]
        assert params[i]["sigma"] >= 0


def test_regimes_have_different_volatilities(fitted_model) -> None:
    """The two fitted regimes should have meaningfully different sigmas."""
    params = fitted_model.get_regime_params()
    sigmas = [params[i]["sigma"] for i in range(2)]
    ratio = max(sigmas) / min(sigmas)
    assert ratio > 1.5, f"Regime sigmas not distinct enough: {sigmas}"


def test_model_detects_regime_shift(fitted_model, two_regime_returns) -> None:
    """Regime labels should mostly differ between the two halves."""
    labels = fitted_model.predict(two_regime_returns)
    first_half_mode = np.bincount(labels[:200]).argmax()
    second_half_mode = np.bincount(labels[200:]).argmax()
    assert first_half_mode != second_half_mode, "Model should detect the regime shift"
