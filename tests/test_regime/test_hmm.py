"""Tests for HMM regime detector."""

from __future__ import annotations

import numpy as np
import pytest

from synthquant.regime.hmm import HMMRegimeDetector


@pytest.fixture()
def fitted_detector(sample_returns: np.ndarray) -> HMMRegimeDetector:
    """Return a fitted HMMRegimeDetector."""
    detector = HMMRegimeDetector(n_components=2, random_state=42)
    detector.fit(sample_returns)
    return detector


def test_fit_returns_self(sample_returns: np.ndarray) -> None:
    """fit() returns the detector itself for method chaining."""
    detector = HMMRegimeDetector(n_components=2, random_state=42)
    result = detector.fit(sample_returns)
    assert result is detector


def test_predict_returns_correct_length(
    fitted_detector: HMMRegimeDetector, sample_returns: np.ndarray
) -> None:
    """predict() returns an array of the same length as input."""
    labels = fitted_detector.predict(sample_returns)
    assert len(labels) == len(sample_returns)


def test_predict_labels_are_valid(
    fitted_detector: HMMRegimeDetector, sample_returns: np.ndarray
) -> None:
    """predict() returns labels in {0, ..., n_components-1}."""
    labels = fitted_detector.predict(sample_returns)
    assert set(labels).issubset({0, 1})


def test_predict_proba_sums_to_one(
    fitted_detector: HMMRegimeDetector, sample_returns: np.ndarray
) -> None:
    """predict_proba() returns probability rows that sum to 1."""
    proba = fitted_detector.predict_proba(sample_returns)
    assert proba.shape == (len(sample_returns), 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_predict_proba_values_in_0_1(
    fitted_detector: HMMRegimeDetector, sample_returns: np.ndarray
) -> None:
    """All probabilities are in [0, 1]."""
    proba = fitted_detector.predict_proba(sample_returns)
    assert np.all(proba >= 0.0)
    assert np.all(proba <= 1.0)


def test_get_regime_params_structure(fitted_detector: HMMRegimeDetector) -> None:
    """get_regime_params() returns a dict with 'mu' and 'sigma' per regime."""
    params = fitted_detector.get_regime_params()
    assert len(params) == 2
    for _regime_id, p in params.items():
        assert "mu" in p
        assert "sigma" in p
        assert isinstance(p["mu"], float)
        assert p["sigma"] > 0.0


def test_predict_before_fit_raises() -> None:
    """predict() raises RuntimeError if model not fitted."""
    detector = HMMRegimeDetector(n_components=2)
    with pytest.raises(RuntimeError, match="fitted"):
        detector.predict(np.array([0.01, -0.02, 0.005]))


def test_different_n_components() -> None:
    """Detector works with n_components=3 on clearly separated synthetic data."""
    rng = np.random.default_rng(7)
    # Generate clearly separated 3-regime data
    seg1 = rng.normal(-0.02, 0.005, 300)
    seg2 = rng.normal(0.00, 0.005, 300)
    seg3 = rng.normal(0.02, 0.005, 300)
    returns_large = np.concatenate([seg1, seg2, seg3])
    detector = HMMRegimeDetector(n_components=3, covariance_type="diag", random_state=0)
    detector.fit(returns_large)
    labels = detector.predict(returns_large)
    assert set(labels).issubset({0, 1, 2})
    params = detector.get_regime_params()
    assert len(params) == 3


# ------------------------------------------------------------------ #
# Filtered prediction tests
# ------------------------------------------------------------------ #


def test_predict_proba_filtered_shape(
    fitted_detector: HMMRegimeDetector, sample_returns: np.ndarray
) -> None:
    """predict_proba_filtered() returns the same shape as predict_proba()."""
    filtered = fitted_detector.predict_proba_filtered(sample_returns)
    assert filtered.shape == (len(sample_returns), 2)


def test_predict_proba_filtered_sums_to_one(
    fitted_detector: HMMRegimeDetector, sample_returns: np.ndarray
) -> None:
    """Filtered probabilities sum to 1 at each time step."""
    filtered = fitted_detector.predict_proba_filtered(sample_returns)
    np.testing.assert_allclose(filtered.sum(axis=1), 1.0, atol=1e-6)


def test_predict_proba_filtered_values_in_0_1(
    fitted_detector: HMMRegimeDetector, sample_returns: np.ndarray
) -> None:
    """All filtered probabilities are in [0, 1]."""
    filtered = fitted_detector.predict_proba_filtered(sample_returns)
    assert np.all(filtered >= 0.0)
    assert np.all(filtered <= 1.0)


def test_predict_filtered_returns_labels(
    fitted_detector: HMMRegimeDetector, sample_returns: np.ndarray
) -> None:
    """predict_filtered() returns valid integer labels."""
    labels = fitted_detector.predict_filtered(sample_returns)
    assert len(labels) == len(sample_returns)
    assert set(labels).issubset({0, 1})


def test_filtered_no_look_ahead(sample_returns: np.ndarray) -> None:
    """Adding more data doesn't change earlier filtered probabilities.

    If we compute filtered probs on returns[:T] and returns[:T+50],
    the first T rows should be identical (no backward pass).
    """
    detector = HMMRegimeDetector(n_components=2, random_state=42)
    detector.fit(sample_returns)

    T = 150
    proba_short = detector.predict_proba_filtered(sample_returns[:T])
    proba_long = detector.predict_proba_filtered(sample_returns)

    # First T rows should be identical — no look-ahead
    np.testing.assert_allclose(proba_short, proba_long[:T], atol=1e-12)


# ------------------------------------------------------------------ #
# Label anchoring tests
# ------------------------------------------------------------------ #


def test_label_anchoring_low_vol_is_state_zero() -> None:
    """After fitting, state 0 should be the lowest-volatility regime."""
    rng = np.random.default_rng(99)
    calm = rng.normal(0.001, 0.005, 300)
    crisis = rng.normal(-0.002, 0.025, 300)
    returns = np.concatenate([calm, crisis])

    detector = HMMRegimeDetector(n_components=2, random_state=42)
    detector.fit(returns)
    params = detector.get_regime_params()

    # State 0 should have lower sigma than state 1
    assert params[0]["sigma"] < params[1]["sigma"]


def test_label_anchoring_consistent_across_seeds() -> None:
    """Label anchoring produces consistent state ordering regardless of seed."""
    rng = np.random.default_rng(99)
    calm = rng.normal(0.001, 0.005, 300)
    crisis = rng.normal(-0.002, 0.025, 300)
    returns = np.concatenate([calm, crisis])

    sigmas_0 = []
    for seed in [0, 1, 42, 123]:
        det = HMMRegimeDetector(n_components=2, random_state=seed)
        det.fit(returns)
        sigmas_0.append(det.get_regime_params()[0]["sigma"])

    # All state-0 sigmas should be the low-vol one
    for s in sigmas_0:
        assert s < 0.015, f"State 0 sigma {s:.4f} > 0.015 — anchoring failed"


# ------------------------------------------------------------------
# Multivariate HMM tests
# ------------------------------------------------------------------


class TestMultivariateHMM:
    """Tests for multivariate (feature-based) HMM regime detection."""

    @pytest.fixture()
    def bimodal_features(self) -> np.ndarray:
        """2D feature matrix with well-separated calm and crisis regimes."""
        rng = np.random.default_rng(42)
        # Calm: low-vol returns, near-zero signed vol, positive momentum
        calm_ret = rng.normal(0.001, 0.005, 200)
        calm_sv = rng.normal(0.0, 0.001, 200)
        calm_mom = rng.normal(0.005, 0.002, 200)
        calm = np.column_stack([calm_ret, calm_sv, calm_mom])
        # Crisis: high-vol returns, negative signed vol, negative momentum
        crisis_ret = rng.normal(-0.003, 0.025, 200)
        crisis_sv = rng.normal(-0.01, 0.005, 200)
        crisis_mom = rng.normal(-0.01, 0.005, 200)
        crisis = np.column_stack([crisis_ret, crisis_sv, crisis_mom])
        return np.vstack([calm, crisis])

    def test_fit_multivariate(self, bimodal_features: np.ndarray) -> None:
        """HMM fits a 2D feature matrix without error."""
        det = HMMRegimeDetector(n_components=2, random_state=42)
        result = det.fit(bimodal_features)
        assert result is det
        assert det._n_features == 3

    def test_predict_multivariate_shape(self, bimodal_features: np.ndarray) -> None:
        """predict() returns correct shape for multivariate input."""
        det = HMMRegimeDetector(n_components=2, random_state=42)
        det.fit(bimodal_features)
        labels = det.predict(bimodal_features)
        assert labels.shape == (400,)
        assert set(np.unique(labels)).issubset({0, 1})

    def test_predict_proba_multivariate(self, bimodal_features: np.ndarray) -> None:
        """predict_proba() returns valid probabilities for multivariate input."""
        det = HMMRegimeDetector(n_components=2, random_state=42)
        det.fit(bimodal_features)
        proba = det.predict_proba(bimodal_features)
        assert proba.shape == (400, 2)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_filtered_multivariate(self, bimodal_features: np.ndarray) -> None:
        """Filtered forward pass works for multivariate observations."""
        det = HMMRegimeDetector(n_components=2, random_state=42)
        det.fit(bimodal_features)
        proba = det.predict_proba_filtered(bimodal_features)
        assert proba.shape == (400, 2)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)
        # Values should be valid probabilities
        assert np.all(proba >= 0)
        assert np.all(proba <= 1)

    def test_multivariate_anchoring(self, bimodal_features: np.ndarray) -> None:
        """State 0 should have lower total variance in multivariate mode."""
        det = HMMRegimeDetector(n_components=2, random_state=42)
        det.fit(bimodal_features)
        covars = det._model.covars_
        trace_0 = float(np.trace(covars[0]))
        trace_1 = float(np.trace(covars[1]))
        assert trace_0 <= trace_1, f"State 0 total var ({trace_0:.6f}) > State 1 ({trace_1:.6f})"

    def test_get_regime_params_multivariate(
        self,
        bimodal_features: np.ndarray,
    ) -> None:
        """get_regime_params() returns first-feature mu/sigma in MV mode."""
        det = HMMRegimeDetector(n_components=2, random_state=42)
        det.fit(bimodal_features)
        params = det.get_regime_params()
        assert 0 in params and 1 in params
        assert "mu" in params[0] and "sigma" in params[0]
        # State 0 (calm) should have lower sigma than state 1 (crisis)
        assert params[0]["sigma"] < params[1]["sigma"]


class TestBuildHmmFeatures:
    """Tests for the feature builder function."""

    def test_output_shape(self, sample_returns: np.ndarray) -> None:
        from synthquant.regime.features import build_hmm_features

        features = build_hmm_features(sample_returns)
        assert features.shape == (len(sample_returns), 3)

    def test_z_scored(self, sample_returns: np.ndarray) -> None:
        """Each column should be approximately zero-mean, unit-variance."""
        from synthquant.regime.features import build_hmm_features

        features = build_hmm_features(sample_returns)
        for col in range(3):
            assert abs(np.mean(features[:, col])) < 0.01
            assert abs(np.std(features[:, col]) - 1.0) < 0.01

    def test_no_nans(self, sample_returns: np.ndarray) -> None:
        from synthquant.regime.features import build_hmm_features

        features = build_hmm_features(sample_returns)
        assert not np.any(np.isnan(features))

    def test_crash_vs_recovery_asymmetry(self) -> None:
        """Signed vol should be negative during crash, positive during recovery."""
        from synthquant.regime.features import build_hmm_features

        rng = np.random.default_rng(42)
        # Crash: 50 days of large negative returns
        crash = rng.normal(-0.03, 0.03, 50)
        # Recovery: 50 days of large positive returns
        recovery = rng.normal(0.03, 0.03, 50)
        combined = np.concatenate([crash, recovery])

        features = build_hmm_features(combined, vol_lookback=10, momentum_lookback=5)
        # Raw signed_vol (before z-score) — check via column 1
        # In crash section, returns are negative → signed_vol negative
        crash_sv_mean = np.mean(features[:50, 1])
        recovery_sv_mean = np.mean(features[50:, 1])
        assert crash_sv_mean < recovery_sv_mean, (
            f"Crash signed_vol ({crash_sv_mean:.3f}) >= Recovery ({recovery_sv_mean:.3f})"
        )

    def test_custom_feature_names(self, sample_returns: np.ndarray) -> None:
        """Can select specific features via feature_names."""
        from synthquant.regime.features import build_hmm_features

        feats = build_hmm_features(
            sample_returns,
            feature_names=["returns", "realized_vol"],
        )
        assert feats.shape == (len(sample_returns), 2)
        assert not np.any(np.isnan(feats))

    def test_all_features_no_nans(self, sample_returns: np.ndarray) -> None:
        """Every registered feature produces no NaNs."""
        from synthquant.regime.features import ALL_FEATURE_NAMES, build_hmm_features

        feats = build_hmm_features(
            sample_returns,
            feature_names=ALL_FEATURE_NAMES,
        )
        assert feats.shape == (len(sample_returns), len(ALL_FEATURE_NAMES))
        assert not np.any(np.isnan(feats))

    def test_unknown_feature_raises(self, sample_returns: np.ndarray) -> None:
        from synthquant.regime.features import build_hmm_features

        with pytest.raises(ValueError, match="Unknown feature"):
            build_hmm_features(sample_returns, feature_names=["bogus"])

    def test_feature_registry_completeness(self) -> None:
        from synthquant.regime.features import (
            ALL_FEATURE_NAMES,
            FEATURE_REGISTRY,
        )

        assert set(ALL_FEATURE_NAMES) == set(FEATURE_REGISTRY.keys())
        assert len(ALL_FEATURE_NAMES) >= 10  # at least 10 candidates
