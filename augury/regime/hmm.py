"""Hidden Markov Model regime detector."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from scipy.stats import multivariate_normal, norm

logger = logging.getLogger(__name__)

__all__ = ["HMMRegimeDetector"]


class HMMRegimeDetector:
    """Gaussian HMM-based regime detector wrapping hmmlearn.

    Fits a Gaussian HMM to either a univariate return series or a
    multivariate feature matrix and identifies distinct market regimes.

    **Multivariate mode**: when given a 2-D feature matrix (e.g. from
    :func:`~augury.regime.features.build_hmm_features`), the HMM
    learns a multivariate Gaussian emission per state.  Interaction
    features such as *signed volatility* break the crash/recovery
    symmetry that plagues univariate HMMs.

    After fitting, states are re-ordered by ascending volatility so that
    state 0 is always the calm (low-vol) regime and the highest-index
    state is the crisis (high-vol) regime.  This anchoring eliminates
    the arbitrary label-swapping problem across consecutive fits.

    Args:
        n_components: Number of hidden states (regimes).
        covariance_type: HMM covariance type ('full', 'diag', 'spherical', 'tied').
        n_iter: Maximum EM iterations.
        tol: EM convergence tolerance.
        random_state: Seed for reproducibility.
    """

    def __init__(
        self,
        n_components: int = 2,
        covariance_type: str = "full",
        n_iter: int = 100,
        tol: float = 1e-4,
        random_state: int | None = None,
    ) -> None:
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.tol = tol
        self.random_state = random_state
        self._model: Any = None
        self._n_features: int = 1  # set after fit()

    @staticmethod
    def _to_obs(data: np.ndarray) -> np.ndarray:
        """Normalise input to (n_samples, n_features) observation matrix."""
        data = np.asarray(data, dtype=float)
        if data.ndim == 1:
            return data.reshape(-1, 1)
        return data

    def fit(self, returns: np.ndarray) -> HMMRegimeDetector:
        """Fit the HMM to a return series or feature matrix.

        Accepts either a 1-D return array (univariate mode) or a 2-D
        feature matrix of shape ``(T, d)`` (multivariate mode, e.g.
        from :func:`~augury.regime.features.build_hmm_features`).

        After EM convergence the internal state ordering is permuted so
        that state 0 has the lowest volatility (calm) and the last state
        the highest (crisis).  This makes labels stable across refits.

        Args:
            returns: 1-D array of log returns *or* 2-D ``(T, d)`` feature
                matrix whose first column is returns.

        Returns:
            self (fitted detector).

        Raises:
            ImportError: If hmmlearn is not installed.
        """
        try:
            from hmmlearn.hmm import GaussianHMM
        except ImportError as e:
            raise ImportError("hmmlearn is required. Install with: pip install augury") from e

        X = self._to_obs(returns)
        self._n_features = X.shape[1]

        self._model = GaussianHMM(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            n_iter=self.n_iter,
            tol=self.tol,
            random_state=self.random_state,
        )
        self._model.fit(X)
        self._anchor_labels()
        logger.info(
            f"HMMRegimeDetector fitted: {self.n_components} states, "
            f"{self._n_features} features, "
            f"converged={self._model.monitor_.converged}"
        )
        return self

    # ------------------------------------------------------------------ #
    # Smoothed prediction (original – uses full backward pass)
    # ------------------------------------------------------------------ #

    def predict(self, returns: np.ndarray) -> np.ndarray:
        """Predict the most likely regime sequence (Viterbi, smoothed).

        Note: uses the full observation window bidirectionally.  For
        online/real-time use, prefer :meth:`predict_filtered`.

        Args:
            returns: 1-D array of log returns *or* 2-D feature matrix.

        Returns:
            Integer array of regime labels, shape (n,).

        Raises:
            RuntimeError: If the model has not been fitted.
        """
        self._check_fitted()
        X = self._to_obs(returns)
        labels = self._model.predict(X)
        logger.debug(f"Predicted {len(labels)} regime labels")
        return labels

    def predict_proba(self, returns: np.ndarray) -> np.ndarray:
        """Compute smoothed posterior state probabilities (forward-backward).

        Note: uses the full observation window bidirectionally.  For
        online/real-time use, prefer :meth:`predict_proba_filtered`.

        Args:
            returns: 1-D array of log returns *or* 2-D feature matrix.

        Returns:
            Array of shape (n, n_components) where each row sums to 1.
        """
        self._check_fitted()
        X = self._to_obs(returns)
        _, posteriors = self._model.score_samples(X)
        return posteriors

    # ------------------------------------------------------------------ #
    # Filtered prediction (forward-only – no look-ahead)
    # ------------------------------------------------------------------ #

    def predict_filtered(self, returns: np.ndarray) -> np.ndarray:
        """Predict regime labels using only past data (forward filter).

        At each time step *t* the regime is chosen as
        ``argmax P(state_t | x_1 ... x_t)`` — no future observations
        are used.  This is suitable for online/real-time trading.

        Args:
            returns: 1-D array of log returns *or* 2-D feature matrix.

        Returns:
            Integer array of regime labels, shape (n,).
        """
        proba = self.predict_proba_filtered(returns)
        return np.argmax(proba, axis=1)

    def predict_proba_filtered(self, returns: np.ndarray) -> np.ndarray:
        """Compute filtered posterior P(state_t | x_1 ... x_t).

        Runs only the forward pass of the HMM (no backward pass),
        so the probability at time *t* depends only on observations
        up to and including *t*.  This eliminates the look-ahead
        bias inherent in the standard forward-backward smoother.

        Supports both univariate (1-D) and multivariate (2-D) inputs.

        Args:
            returns: 1-D array of log returns *or* 2-D feature matrix.

        Returns:
            Array of shape (n, n_components) where each row sums to 1.
        """
        self._check_fitted()
        X = self._to_obs(returns)
        n, d = X.shape
        k = self.n_components

        startprob = self._model.startprob_  # (k,)
        transmat = self._model.transmat_  # (k, k)

        if d == 1:
            return self._forward_univariate(X.ravel(), n, k, startprob, transmat)
        return self._forward_multivariate(X, n, k, d, startprob, transmat)

    def _forward_univariate(
        self,
        x: np.ndarray,
        n: int,
        k: int,
        startprob: np.ndarray,
        transmat: np.ndarray,
    ) -> np.ndarray:
        """Forward pass for univariate observations."""
        means = self._model.means_.ravel()  # (k,)
        variances = np.array([abs(float(self._model.covars_[i].flat[0])) for i in range(k)])
        stds = np.sqrt(variances)

        alpha = np.empty((n, k), dtype=float)

        emission_0 = norm.pdf(x[0], loc=means, scale=stds)
        alpha[0] = startprob * emission_0
        row_sum = alpha[0].sum()
        if row_sum > 0:
            alpha[0] /= row_sum

        for t in range(1, n):
            predicted = alpha[t - 1] @ transmat
            emission_t = norm.pdf(x[t], loc=means, scale=stds)
            alpha[t] = predicted * emission_t
            row_sum = alpha[t].sum()
            if row_sum > 0:
                alpha[t] /= row_sum

        return alpha

    def _forward_multivariate(
        self,
        X: np.ndarray,
        n: int,
        k: int,
        d: int,
        startprob: np.ndarray,
        transmat: np.ndarray,
    ) -> np.ndarray:
        """Forward pass for multivariate observations."""
        means = self._model.means_  # (k, d)
        covars = self._model.covars_  # (k, d, d) expanded

        # Pre-build frozen multivariate normal distributions per state
        mvn_pdfs = []
        for i in range(k):
            cov_i = covars[i]
            # Regularise: add small diagonal to prevent singular covariance
            cov_i = cov_i + np.eye(d) * 1e-8
            mvn_pdfs.append(multivariate_normal(mean=means[i], cov=cov_i))

        alpha = np.empty((n, k), dtype=float)

        emission_0 = np.array([mvn_pdfs[i].pdf(X[0]) for i in range(k)])
        alpha[0] = startprob * emission_0
        row_sum = alpha[0].sum()
        if row_sum > 0:
            alpha[0] /= row_sum

        for t in range(1, n):
            predicted = alpha[t - 1] @ transmat
            emission_t = np.array([mvn_pdfs[i].pdf(X[t]) for i in range(k)])
            alpha[t] = predicted * emission_t
            row_sum = alpha[t].sum()
            if row_sum > 0:
                alpha[t] /= row_sum

        return alpha

    # ------------------------------------------------------------------ #
    # Parameter access
    # ------------------------------------------------------------------ #

    def get_regime_params(self) -> dict[int, dict[str, float]]:
        """Return fitted parameters for each regime.

        Returns mu/sigma of the **first feature** (returns) for backward
        compatibility.  In multivariate mode, the full mean vector is
        available via ``self._model.means_``.

        Returns:
            Dict mapping regime index to {'mu': float, 'sigma': float}.
            State 0 is always the lowest-volatility (calm) regime.

        Raises:
            RuntimeError: If the model has not been fitted.
        """
        self._check_fitted()
        params: dict[int, dict[str, float]] = {}
        for i in range(self.n_components):
            mu = float(self._model.means_[i, 0])
            cov = self._model.covars_[i]
            sigma = float(np.sqrt(abs(float(cov.flat[0]))))
            params[i] = {"mu": mu, "sigma": sigma}
        return params

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _anchor_labels(self) -> None:
        """Re-order HMM states by ascending volatility.

        After EM fitting, state labels are arbitrary.  This permutes
        the model parameters so that state 0 always corresponds to
        the lowest-variance (calm) regime.

        For multivariate observations, uses the **trace** of each
        state's covariance matrix (total variance across all features)
        as the ordering criterion.

        Handles all hmmlearn covariance types: full, diag, spherical, tied.
        """
        k = self.n_components
        cov_type = self._model.covariance_type

        if cov_type == "tied":
            pass

        # Use the public getter to get expanded covariances for sorting
        covars_expanded = self._model.covars_  # (k, d, d) for all types

        if self._n_features == 1:
            # Univariate: use scalar variance
            variances = np.array([abs(float(covars_expanded[i].flat[0])) for i in range(k)])
        else:
            # Multivariate: use trace of covariance (total variance)
            variances = np.array([float(np.trace(covars_expanded[i])) for i in range(k)])

        order = np.argsort(variances)
        if np.array_equal(order, np.arange(k)):
            return  # already in ascending-vol order

        # Permute means and start probs
        self._model.means_ = self._model.means_[order]
        self._model.startprob_ = self._model.startprob_[order]
        self._model.transmat_ = self._model.transmat_[np.ix_(order, order)]

        # Permute the raw internal covariance representation
        # (_covars_ stores the compact form; covars_ getter expands it)
        raw_covars = self._model._covars_
        self._model._covars_ = raw_covars[order]

    def _check_fitted(self) -> None:
        if self._model is None:
            raise RuntimeError("Model has not been fitted. Call fit() first.")
