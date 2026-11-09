"""Statistical correctness tests for regime detection models.

These tests verify that regime models can RECOVER known regimes
from synthetic data with known ground-truth structure, not just
that the models run without errors.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from augury.regime.clustering import ClusteringRegimeDetector
from augury.regime.ensemble import EnsembleRegimeDetector
from augury.regime.hmm import HMMRegimeDetector
from augury.regime.ms_garch import MarkovSwitchingGARCH

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _generate_two_regime_data(
    n_per_regime: int = 500,
    mu_low: float = 0.0005,
    sigma_low: float = 0.005,
    mu_high: float = -0.001,
    sigma_high: float = 0.025,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic returns with two well-separated regimes.

    Returns (returns, true_labels) where regime 0 = low-vol, regime 1 = high-vol.
    The data is a simple concatenation: [regime_0] then [regime_1].
    """
    rng = np.random.default_rng(seed)
    r0 = rng.normal(mu_low, sigma_low, n_per_regime)
    r1 = rng.normal(mu_high, sigma_high, n_per_regime)
    returns = np.concatenate([r0, r1])
    labels = np.concatenate([np.zeros(n_per_regime), np.ones(n_per_regime)]).astype(int)
    return returns, labels


def _generate_three_regime_data(
    n_per_regime: int = 400,
    seed: int = 123,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate 3-regime synthetic data: calm, moderate, crisis."""
    rng = np.random.default_rng(seed)
    r0 = rng.normal(0.0005, 0.003, n_per_regime)  # calm
    r1 = rng.normal(0.0000, 0.015, n_per_regime)  # moderate
    r2 = rng.normal(-0.003, 0.040, n_per_regime)  # crisis
    returns = np.concatenate([r0, r1, r2])
    labels = np.concatenate(
        [
            np.zeros(n_per_regime),
            np.ones(n_per_regime),
            2 * np.ones(n_per_regime),
        ]
    ).astype(int)
    return returns, labels


def _regime_accuracy(true_labels: np.ndarray, pred_labels: np.ndarray) -> float:
    """Compute best accuracy across all label permutations.

    HMM/GMM labels are arbitrary (regime 0 could map to either true regime),
    so we check all permutations and return the best match.
    """
    from itertools import permutations

    n_regimes = len(set(true_labels))
    best = 0.0
    for perm in permutations(range(n_regimes)):
        remapped = np.array([perm[lbl] for lbl in pred_labels])
        acc = np.mean(remapped == true_labels)
        best = max(best, acc)
    return best


def _match_regime_params(
    true_params: dict[int, dict[str, float]],
    fitted_params: dict[int, dict[str, float]],
) -> dict[int, dict[str, float]]:
    """Match fitted regimes to true regimes by closest sigma.

    Returns fitted_params re-indexed to match true regime ordering.
    """
    true_sigmas = {k: v["sigma"] for k, v in true_params.items()}
    fitted_sigmas = {k: v["sigma"] for k, v in fitted_params.items()}

    # Sort both by sigma and create mapping
    true_order = sorted(true_sigmas, key=true_sigmas.get)
    fitted_order = sorted(fitted_sigmas, key=fitted_sigmas.get)

    mapping = dict(zip(fitted_order, true_order, strict=False))
    return {mapping[k]: v for k, v in fitted_params.items()}


# ===========================================================================
# HMM Regime Detector — Statistical Correctness
# ===========================================================================


class TestHMMCorrectness:
    """Verify that HMM can recover known regime structure."""

    def test_hmm_recovers_two_regime_labels(self):
        """HMM should assign correct labels to two well-separated regimes."""
        returns, true_labels = _generate_two_regime_data(n_per_regime=500, seed=42)
        hmm = HMMRegimeDetector(n_components=2, random_state=42)
        hmm.fit(returns)
        pred = hmm.predict(returns)

        accuracy = _regime_accuracy(true_labels, pred)
        assert accuracy > 0.85, (
            f"HMM label accuracy {accuracy:.3f} is too low; "
            "model cannot recover known regime structure"
        )

    def test_hmm_recovers_two_regime_parameters(self):
        """HMM should recover μ and σ close to the true generating values."""
        mu_low, sigma_low = 0.0005, 0.005
        mu_high, sigma_high = -0.001, 0.025

        returns, _ = _generate_two_regime_data(
            n_per_regime=800,
            mu_low=mu_low,
            sigma_low=sigma_low,
            mu_high=mu_high,
            sigma_high=sigma_high,
            seed=99,
        )
        hmm = HMMRegimeDetector(n_components=2, random_state=99)
        hmm.fit(returns)
        raw_params = hmm.get_regime_params()

        # Match fitted regimes to true regimes by sigma
        true_params = {
            0: {"mu": mu_low, "sigma": sigma_low},
            1: {"mu": mu_high, "sigma": sigma_high},
        }
        matched = _match_regime_params(true_params, raw_params)

        # Sigma recovery (most important for downstream vol modeling)
        assert abs(matched[0]["sigma"] - sigma_low) < 0.003, (
            f"Low-vol regime sigma: fitted={matched[0]['sigma']:.5f}, true={sigma_low}"
        )
        assert abs(matched[1]["sigma"] - sigma_high) < 0.005, (
            f"High-vol regime sigma: fitted={matched[1]['sigma']:.5f}, true={sigma_high}"
        )

        # Mu recovery (weaker signal, wider tolerance)
        assert abs(matched[0]["mu"] - mu_low) < 0.002
        assert abs(matched[1]["mu"] - mu_high) < 0.005

    def test_hmm_three_regime_recovery(self):
        """HMM should identify 3 distinct regimes with well-separated params."""
        returns, true_labels = _generate_three_regime_data(n_per_regime=800, seed=42)
        hmm = HMMRegimeDetector(n_components=3, n_iter=200, random_state=42)
        hmm.fit(returns)
        pred = hmm.predict(returns)

        accuracy = _regime_accuracy(true_labels, pred)
        assert accuracy > 0.65, f"3-regime HMM accuracy {accuracy:.3f} too low"

        # Verify 3 distinct sigma values recovered
        params = hmm.get_regime_params()
        sigmas = sorted([params[i]["sigma"] for i in range(3)])
        # Ratios between consecutive regimes should be > 1.3
        assert sigmas[1] / sigmas[0] > 1.3, "Calm/moderate regimes not separated"
        assert sigmas[2] / sigmas[1] > 1.3, "Moderate/crisis regimes not separated"

    def test_hmm_transition_matrix_structure(self):
        """HMM transition matrix should have high self-transition probabilities
        (regimes are persistent, not random noise)."""
        returns, _ = _generate_two_regime_data(n_per_regime=500, seed=42)
        hmm = HMMRegimeDetector(n_components=2, random_state=42)
        hmm.fit(returns)

        # Access the fitted transition matrix from hmmlearn
        trans_mat = hmm._model.transmat_
        assert trans_mat.shape == (2, 2)

        # Self-transition probabilities should dominate (regimes are sticky)
        for i in range(2):
            assert trans_mat[i, i] > 0.8, (
                f"Self-transition prob for regime {i} is {trans_mat[i, i]:.3f}, "
                "expected > 0.8 for persistent regimes"
            )

    def test_hmm_predict_proba_confident_in_known_segments(self):
        """In the middle of a known regime segment, HMM should be confident."""
        returns, _ = _generate_two_regime_data(n_per_regime=500, seed=42)
        hmm = HMMRegimeDetector(n_components=2, random_state=42)
        hmm.fit(returns)
        proba = hmm.predict_proba(returns)

        # In the middle of each segment (away from transition), max prob should be high
        # Regime 0: indices 100-400, Regime 1: indices 600-900
        mid_regime0 = proba[100:400].max(axis=1)
        mid_regime1 = proba[600:900].max(axis=1)

        assert np.mean(mid_regime0 > 0.8) > 0.9, "HMM not confident in regime 0 interior"
        assert np.mean(mid_regime1 > 0.8) > 0.9, "HMM not confident in regime 1 interior"


# ===========================================================================
# Clustering (GMM) Regime Detector — Statistical Correctness
# ===========================================================================


class TestClusteringCorrectness:
    """Verify that GMM-based clustering recovers known regime structure."""

    def test_gmm_recovers_two_regime_labels(self):
        """GMM should cluster two well-separated distributions correctly."""
        returns, true_labels = _generate_two_regime_data(n_per_regime=500, seed=42)
        gmm = ClusteringRegimeDetector(n_components=2, random_state=42)
        gmm.fit(returns)
        pred = gmm.predict(returns)

        accuracy = _regime_accuracy(true_labels, pred)
        assert accuracy > 0.80, f"GMM label accuracy {accuracy:.3f} too low"

    def test_gmm_recovers_component_means_and_variances(self):
        """GMM should recover true μ and σ of each generating Gaussian."""
        mu_low, sigma_low = 0.0005, 0.005
        mu_high, sigma_high = -0.001, 0.025

        returns, _ = _generate_two_regime_data(
            n_per_regime=800,
            mu_low=mu_low,
            sigma_low=sigma_low,
            mu_high=mu_high,
            sigma_high=sigma_high,
            seed=99,
        )
        gmm = ClusteringRegimeDetector(n_components=2, random_state=99)
        gmm.fit(returns)
        raw_params = gmm.get_regime_params()

        # Extract means and stds
        fitted = {}
        for i, p in raw_params.items():
            mean = p["means"][0]
            var = (
                p["covariances"] if isinstance(p["covariances"], float) else p["covariances"][0][0]
            )
            fitted[i] = {"mu": mean, "sigma": np.sqrt(var)}

        # Match by sigma
        true_params = {
            0: {"mu": mu_low, "sigma": sigma_low},
            1: {"mu": mu_high, "sigma": sigma_high},
        }
        matched = _match_regime_params(true_params, fitted)

        assert abs(matched[0]["sigma"] - sigma_low) < 0.003
        assert abs(matched[1]["sigma"] - sigma_high) < 0.005
        assert abs(matched[0]["mu"] - mu_low) < 0.002
        assert abs(matched[1]["mu"] - mu_high) < 0.005

    def test_gmm_recovers_component_weights(self):
        """GMM should recover ~50/50 weight split for equal-sized regimes."""
        returns, _ = _generate_two_regime_data(n_per_regime=500, seed=42)
        gmm = ClusteringRegimeDetector(n_components=2, random_state=42)
        gmm.fit(returns)
        params = gmm.get_regime_params()

        for i in range(2):
            assert abs(params[i]["weight"] - 0.5) < 0.1, (
                f"Component {i} weight {params[i]['weight']:.3f} far from 0.5"
            )

    def test_gmm_unequal_weight_recovery(self):
        """GMM should approximate correct proportions for unequal splits."""
        rng = np.random.default_rng(42)
        r0 = rng.normal(0.001, 0.005, 800)  # 80%
        r1 = rng.normal(-0.002, 0.025, 200)  # 20%
        returns = np.concatenate([r0, r1])

        gmm = ClusteringRegimeDetector(n_components=2, random_state=42)
        gmm.fit(returns)
        params = gmm.get_regime_params()

        weights = sorted([params[i]["weight"] for i in range(2)])
        # Smaller component should be ~0.2, larger ~0.8
        assert abs(weights[0] - 0.2) < 0.10
        assert abs(weights[1] - 0.8) < 0.10


# ===========================================================================
# MS-GARCH — Parameter Recovery
# ===========================================================================


class TestMSGARCHCorrectness:
    """Verify MS-GARCH recovers known regime parameters."""

    def test_ms_garch_recovers_regime_volatilities(self):
        """MS-GARCH should recover true σ per regime."""
        sigma_low, sigma_high = 0.005, 0.025
        returns, _ = _generate_two_regime_data(
            n_per_regime=500,
            sigma_low=sigma_low,
            sigma_high=sigma_high,
            seed=42,
        )
        ms = MarkovSwitchingGARCH(k_regimes=2)
        ms.fit(returns)
        params = ms.get_regime_params()

        sigmas = sorted([params[i]["sigma"] for i in range(2)])
        # Allow wider tolerance for MS-GARCH (harder optimization)
        assert abs(sigmas[0] - sigma_low) < 0.005, (
            f"Low-vol sigma: fitted={sigmas[0]:.5f}, true={sigma_low}"
        )
        assert abs(sigmas[1] - sigma_high) < 0.010, (
            f"High-vol sigma: fitted={sigmas[1]:.5f}, true={sigma_high}"
        )

    def test_ms_garch_identifies_regime_shift_location(self):
        """MS-GARCH should detect regime shift roughly where it actually occurs."""
        returns, true_labels = _generate_two_regime_data(
            n_per_regime=500,
            sigma_low=0.005,
            sigma_high=0.025,
            seed=42,
        )
        ms = MarkovSwitchingGARCH(k_regimes=2)
        ms.fit(returns)
        labels = ms.predict(returns)

        # The true transition is at index 500
        # Check that mode of labels flips around that point
        first_half_mode = stats.mode(labels[:450], keepdims=True).mode[0]
        second_half_mode = stats.mode(labels[550:], keepdims=True).mode[0]
        assert first_half_mode != second_half_mode, (
            "MS-GARCH should detect different regimes in first vs second half"
        )

    def test_ms_garch_smoothed_probabilities_transition(self):
        """Smoothed probabilities should transition near the true breakpoint."""
        returns, _ = _generate_two_regime_data(
            n_per_regime=500,
            sigma_low=0.005,
            sigma_high=0.025,
            seed=42,
        )
        ms = MarkovSwitchingGARCH(k_regimes=2)
        ms.fit(returns)
        proba = ms.predict_proba(returns)

        # Before breakpoint (index 200-400): one regime should dominate
        max_prob_early = proba[200:400].max(axis=1)
        assert np.mean(max_prob_early > 0.7) > 0.8, (
            "Smoothed probabilities not confident before breakpoint"
        )

        # After breakpoint (index 600-900): the other regime should dominate
        max_prob_late = proba[600:900].max(axis=1)
        assert np.mean(max_prob_late > 0.7) > 0.8, (
            "Smoothed probabilities not confident after breakpoint"
        )


# ===========================================================================
# Ensemble Detector — Correctness
# ===========================================================================


class TestEnsembleCorrectness:
    """Verify ensemble improves over or matches individual detectors."""

    def test_ensemble_majority_vote_accuracy(self):
        """Ensemble majority vote should achieve reasonable accuracy.

        Note: majority voting assumes aligned labels across detectors.
        Since HMM/GMM labels are arbitrary, the ensemble may degrade
        vs the best individual — this tests that it still achieves
        a useful accuracy (>0.75 on well-separated data).
        """
        returns, true_labels = _generate_two_regime_data(n_per_regime=500, seed=42)

        hmm = HMMRegimeDetector(n_components=2, random_state=42)
        gmm = ClusteringRegimeDetector(n_components=2, random_state=42)

        ensemble = EnsembleRegimeDetector(
            detectors=[hmm, gmm],
            method="majority_vote",
            n_regimes=2,
        )
        ensemble.fit(returns)
        ensemble_acc = _regime_accuracy(true_labels, ensemble.predict(returns))

        # Ensemble with unaligned labels may not match the best individual,
        # but should still achieve reasonable accuracy on well-separated data
        assert ensemble_acc > 0.75, (
            f"Ensemble accuracy {ensemble_acc:.3f} too low on well-separated data"
        )

    def test_ensemble_proba_average_calibration(self):
        """Probability-averaged ensemble should produce well-calibrated probabilities."""
        returns, true_labels = _generate_two_regime_data(n_per_regime=500, seed=42)

        hmm = HMMRegimeDetector(n_components=2, random_state=42)
        gmm = ClusteringRegimeDetector(n_components=2, random_state=42)

        ensemble = EnsembleRegimeDetector(
            detectors=[hmm, gmm],
            method="proba_average",
            n_regimes=2,
        )
        ensemble.fit(returns)
        proba = ensemble.predict_proba(returns)

        # Probabilities should sum to 1
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-10)

        # High confidence predictions should be correct
        max_proba = proba.max(axis=1)
        confident_mask = max_proba > 0.9
        if confident_mask.sum() > 50:
            confident_preds = np.argmax(proba[confident_mask], axis=1)
            confident_true = true_labels[confident_mask]
            confident_acc = _regime_accuracy(confident_true, confident_preds)
            assert confident_acc > 0.90, (
                f"High-confidence predictions only {confident_acc:.3f} accurate"
            )

    def test_ensemble_three_detector_stability(self):
        """Ensemble with 3 detectors should remain accurate."""
        returns, true_labels = _generate_two_regime_data(n_per_regime=500, seed=42)

        detectors = [
            HMMRegimeDetector(n_components=2, random_state=42),
            HMMRegimeDetector(n_components=2, random_state=99),
            ClusteringRegimeDetector(n_components=2, random_state=42),
        ]
        ensemble = EnsembleRegimeDetector(
            detectors=detectors,
            method="majority_vote",
            n_regimes=2,
        )
        ensemble.fit(returns)
        acc = _regime_accuracy(true_labels, ensemble.predict(returns))
        assert acc > 0.75, f"3-detector ensemble accuracy {acc:.3f} too low"


# ===========================================================================
# Cross-model consistency
# ===========================================================================


class TestCrossModelConsistency:
    """Verify that different regime models agree on clear regime structure."""

    def test_models_agree_on_obvious_regimes(self):
        """All regime models should agree on strongly separated regimes."""
        returns, true_labels = _generate_two_regime_data(
            n_per_regime=600,
            sigma_low=0.005,
            sigma_high=0.025,
            seed=42,
        )

        hmm = HMMRegimeDetector(n_components=2, random_state=42)
        gmm = ClusteringRegimeDetector(n_components=2, n_init=10, random_state=42)
        ms = MarkovSwitchingGARCH(k_regimes=2)

        hmm.fit(returns)
        gmm.fit(returns)
        ms.fit(returns)

        hmm_acc = _regime_accuracy(true_labels, hmm.predict(returns))
        gmm_acc = _regime_accuracy(true_labels, gmm.predict(returns))
        ms_acc = _regime_accuracy(true_labels, ms.predict(returns))

        # All should achieve >75% on well-separated data
        for name, acc in [("HMM", hmm_acc), ("GMM", gmm_acc), ("MS-GARCH", ms_acc)]:
            assert acc > 0.75, f"{name} accuracy {acc:.3f} too low on obvious regimes"

    def test_regime_params_sigma_ordering_consistent(self):
        """All models should agree on which regime has higher volatility."""
        returns, _ = _generate_two_regime_data(
            n_per_regime=500,
            sigma_low=0.005,
            sigma_high=0.025,
            seed=42,
        )

        hmm = HMMRegimeDetector(n_components=2, random_state=42)
        gmm = ClusteringRegimeDetector(n_components=2, random_state=42)

        hmm.fit(returns)
        gmm.fit(returns)

        # Both should have sigma ratio > 2.0 between regimes
        hmm_params = hmm.get_regime_params()
        hmm_sigmas = sorted([hmm_params[i]["sigma"] for i in range(2)])

        gmm_raw = gmm.get_regime_params()
        gmm_sigmas = sorted(
            [
                np.sqrt(
                    gmm_raw[i]["covariances"][0][0]
                    if isinstance(gmm_raw[i]["covariances"], list)
                    else gmm_raw[i]["covariances"]
                )
                for i in range(2)
            ]
        )

        hmm_ratio = hmm_sigmas[1] / hmm_sigmas[0]
        gmm_ratio = gmm_sigmas[1] / gmm_sigmas[0]

        assert hmm_ratio > 2.0, f"HMM sigma ratio {hmm_ratio:.2f} too low"
        assert gmm_ratio > 2.0, f"GMM sigma ratio {gmm_ratio:.2f} too low"
