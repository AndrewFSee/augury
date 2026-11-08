"""Markov-Switching GARCH regime detector."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["MarkovSwitchingGARCH"]


class MarkovSwitchingGARCH:
    """Markov-Switching model with regime-conditional volatility.

    Uses statsmodels MarkovRegression to identify regimes with
    distinct mean and variance parameters.

    After fitting, states are re-ordered by ascending variance so that
    state 0 is always the calm (low-vol) regime, matching the HMM
    anchoring convention.

    Args:
        k_regimes: Number of regimes.
        switching_variance: Allow variance to switch between regimes.
        switching_mean: Allow mean to switch between regimes.
    """

    def __init__(
        self,
        k_regimes: int = 2,
        switching_variance: bool = True,
        switching_mean: bool = True,
    ) -> None:
        self.k_regimes = k_regimes
        self.switching_variance = switching_variance
        self.switching_mean = switching_mean
        self._result: Any = None
        self._label_order: np.ndarray | None = None

    def fit(self, returns: np.ndarray) -> MarkovSwitchingGARCH:
        """Fit the Markov-Switching model.

        Args:
            returns: 1-D array of log returns.

        Returns:
            self (fitted detector).

        Raises:
            ImportError: If statsmodels is not installed.
        """
        try:
            from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
        except ImportError as e:
            raise ImportError(
                "statsmodels is required. Install with: pip install synthquant"
            ) from e

        endog = np.asarray(returns, dtype=float)
        model = MarkovRegression(
            endog,
            k_regimes=self.k_regimes,
            switching_variance=self.switching_variance,
            switching_trend=self.switching_mean,
        )
        self._result = model.fit(disp=False)
        self._anchor_labels()
        logger.info(
            f"MarkovSwitchingGARCH fitted: {self.k_regimes} regimes, AIC={self._result.aic:.4f}"
        )
        return self

    def predict(self, returns: np.ndarray) -> np.ndarray:
        """Predict the most likely regime at each point in time.

        Args:
            returns: 1-D array of log returns (must match training data length).

        Returns:
            Integer array of regime labels.

        Raises:
            RuntimeError: If the model has not been fitted.
        """
        self._check_fitted()
        proba = self.predict_proba(returns)
        labels = np.argmax(proba, axis=1)
        logger.debug(f"MarkovSwitching predicted {len(labels)} regime labels")
        return labels

    def predict_proba(self, returns: np.ndarray) -> np.ndarray:  # noqa: ARG002
        """Return smoothed regime probabilities.

        Args:
            returns: 1-D array of log returns (ignored; smoothed probs from training used).

        Returns:
            Array of shape (n, k_regimes) with smoothed probabilities.
        """
        self._check_fitted()
        probs = np.asarray(self._result.smoothed_marginal_probabilities)
        return probs[:, self._label_order]

    def predict_proba_filtered(self, returns: np.ndarray) -> np.ndarray:  # noqa: ARG002
        """Return Hamilton-filtered regime probabilities (no look-ahead).

        Uses ``filtered_marginal_probabilities`` from the statsmodels
        result, where P(state_t | x_1 ... x_t) depends only on past
        and current observations.

        Args:
            returns: 1-D array of log returns (ignored; filtered probs
                from training used — statsmodels does not support
                out-of-sample filtering).

        Returns:
            Array of shape (n, k_regimes) with filtered probabilities.
        """
        self._check_fitted()
        probs = np.asarray(self._result.filtered_marginal_probabilities)
        return probs[:, self._label_order]

    def predict_filtered(self, returns: np.ndarray) -> np.ndarray:
        """Predict regime labels using Hamilton-filtered probabilities.

        Args:
            returns: 1-D array of log returns.

        Returns:
            Integer array of regime labels.
        """
        proba = self.predict_proba_filtered(returns)
        return np.argmax(proba, axis=1)

    def get_regime_params(self) -> dict[int, dict[str, float]]:
        """Return fitted mean and variance for each regime.

        Returns:
            Dict mapping regime index to {'mu': float, 'sigma': float}.
            State 0 is always the lowest-variance (calm) regime.
        """
        self._check_fitted()
        if self._label_order is None:
            self._anchor_labels()
        assert self._label_order is not None
        param_map = dict(
            zip(self._result.model.data.param_names, self._result.params, strict=False)
        )
        params: dict[int, dict[str, float]] = {}
        for new_idx, orig_idx in enumerate(self._label_order):
            mu = float(param_map.get(f"const[{orig_idx}]", 0.0))
            var = float(param_map.get(f"sigma2[{orig_idx}]", 1.0))
            params[new_idx] = {"mu": mu, "sigma": float(np.sqrt(var))}
        return params

    def _anchor_labels(self) -> None:
        """Compute state ordering by ascending variance."""
        param_map = dict(
            zip(self._result.model.data.param_names, self._result.params, strict=False)
        )
        variances = np.array(
            [float(param_map.get(f"sigma2[{i}]", 1.0)) for i in range(self.k_regimes)]
        )
        self._label_order = np.argsort(variances)

    def _check_fitted(self) -> None:
        if self._result is None:
            raise RuntimeError("Model has not been fitted. Call fit() first.")
