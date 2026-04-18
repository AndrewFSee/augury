"""Ensemble regime detector combining multiple base detectors."""

from __future__ import annotations

import logging
from itertools import permutations
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

logger = logging.getLogger(__name__)

__all__ = ["EnsembleRegimeDetector"]


class EnsembleRegimeDetector:
    """Combines multiple regime detectors via voting or probability averaging.

    All base detectors must be pre-fitted before passing to this class,
    OR fitting will be delegated to each detector's own ``fit()`` method.

    Args:
        detectors: List of fitted (or unfitted) regime detector instances.
            Each must implement ``predict(x)`` and ``predict_proba(x)``.
        method: Combination method: 'majority_vote' or 'proba_average'.
        n_regimes: Expected number of regimes. Required for 'proba_average'.
    """

    def __init__(
        self,
        detectors: list[Any],
        method: str = "majority_vote",
        n_regimes: int = 2,
    ) -> None:
        if method not in ("majority_vote", "proba_average"):
            raise ValueError("method must be 'majority_vote' or 'proba_average'")
        self.detectors = detectors
        self.method = method
        self.n_regimes = n_regimes

    def fit(self, features: np.ndarray) -> EnsembleRegimeDetector:
        """Fit all base detectors.

        Args:
            features: Feature array passed directly to each detector's fit().

        Returns:
            self
        """
        for det in self.detectors:
            det.fit(features)
        logger.info(f"EnsembleRegimeDetector fitted {len(self.detectors)} detectors")
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Predict regime labels using the configured combination method.

        Args:
            features: Feature array passed to each detector.

        Returns:
            Integer array of regime labels, shape (n,).
        """
        if self.method == "majority_vote":
            all_labels = [d.predict(features) for d in self.detectors]
            # Align labels to the first detector before voting
            aligned = self._align_labels(all_labels)
            stacked = np.stack(aligned, axis=1)
            labels = np.apply_along_axis(
                lambda row: np.bincount(row, minlength=self.n_regimes).argmax(),
                axis=1,
                arr=stacked,
            )
        else:
            avg_proba = self.predict_proba(features)
            labels = np.argmax(avg_proba, axis=1)
        logger.debug(f"Ensemble predicted {len(labels)} labels via {self.method}")
        return labels

    @staticmethod
    def _align_labels(label_sets: list[np.ndarray]) -> list[np.ndarray]:
        """Align regime labels across detectors to a common ordering.

        Since HMM/GMM label assignments are arbitrary (regime 0 in one model
        may correspond to regime 1 in another), this finds the permutation
        of each detector's labels that maximises agreement with the first
        detector (used as the reference).

        Args:
            label_sets: List of label arrays from each detector.

        Returns:
            List of label arrays with labels re-mapped for maximum agreement.
        """
        if len(label_sets) <= 1:
            return label_sets

        reference = label_sets[0]
        n_regimes = len(set(reference))
        if n_regimes == 0:
            return label_sets
        aligned = [reference]

        for labels in label_sets[1:]:
            if n_regimes > 5:
                # Use Hungarian algorithm for large k — O(k^3) vs O(k!)
                confusion = np.zeros((n_regimes, n_regimes), dtype=int)
                for ref_l, det_l in zip(reference, labels, strict=False):
                    confusion[ref_l, det_l] += 1
                # Maximize agreement = minimize negative agreement
                row_ind, col_ind = linear_sum_assignment(-confusion)
                mapping = np.arange(n_regimes)
                mapping[col_ind] = row_ind
                aligned.append(mapping[labels])
            else:
                best_perm: tuple[int, ...] = tuple(range(n_regimes))
                best_acc = -1.0
                for perm in permutations(range(n_regimes)):
                    remapped = np.array([perm[lbl] for lbl in labels])
                    acc = np.mean(remapped == reference)
                    if acc > best_acc:
                        best_acc = acc
                        best_perm = perm
                aligned.append(np.array([best_perm[lbl] for lbl in labels]))

        return aligned

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Average posterior probabilities across all detectors.

        Columns are aligned to the first detector's label ordering
        before averaging, so that regime 0 means the same thing
        across all detectors.

        Args:
            features: Feature array passed to each detector.

        Returns:
            Array of shape (n, n_regimes) with averaged probabilities.
        """
        return self._predict_proba_impl(features, filtered=False)

    def predict_proba_filtered(self, features: np.ndarray) -> np.ndarray:
        """Average *filtered* posterior probabilities (no look-ahead).

        Falls back to smoothed ``predict_proba`` for any detector that
        does not implement ``predict_proba_filtered``.

        Args:
            features: Feature array passed to each detector.

        Returns:
            Array of shape (n, n_regimes) with averaged filtered probabilities.
        """
        return self._predict_proba_impl(features, filtered=True)

    def predict_filtered(self, features: np.ndarray) -> np.ndarray:
        """Predict regime labels using filtered probabilities.

        Args:
            features: Feature array passed to each detector.

        Returns:
            Integer array of regime labels, shape (n,).
        """
        avg_proba = self.predict_proba_filtered(features)
        return np.argmax(avg_proba, axis=1)

    def _predict_proba_impl(self, features: np.ndarray, *, filtered: bool) -> np.ndarray:
        """Shared implementation for smoothed and filtered probability averaging."""
        probas = []
        for d in self.detectors:
            if filtered and hasattr(d, "predict_proba_filtered"):
                probas.append(d.predict_proba_filtered(features))
            else:
                probas.append(d.predict_proba(features))

        # Determine the column permutation by aligning hard labels
        labels = [d.predict(features) for d in self.detectors]
        if len(labels) > 1:
            reference = labels[0]
            n_regimes = self.n_regimes
            aligned_probas = [probas[0]]
            for idx in range(1, len(labels)):
                if n_regimes > 5:
                    confusion = np.zeros((n_regimes, n_regimes), dtype=int)
                    for ref_l, det_l in zip(reference, labels[idx], strict=False):
                        confusion[ref_l, det_l] += 1
                    row_ind, col_ind = linear_sum_assignment(-confusion)
                    perm_list = list(range(n_regimes))
                    for r_i, c_i in zip(row_ind, col_ind, strict=False):
                        perm_list[c_i] = r_i
                    aligned_probas.append(probas[idx][:, perm_list])
                else:
                    best_perm: tuple[int, ...] = tuple(range(n_regimes))
                    best_acc = -1.0
                    for perm in permutations(range(n_regimes)):
                        remapped = np.array([perm[lbl] for lbl in labels[idx]])
                        acc = np.mean(remapped == reference)
                        if acc > best_acc:
                            best_acc = acc
                            best_perm = perm
                    # Reorder columns of the probability matrix
                    aligned_probas.append(probas[idx][:, list(best_perm)])
            probas = aligned_probas
        avg = np.mean(np.stack(probas, axis=0), axis=0)
        return avg

    def get_regime_params(self) -> dict[int, dict[str, Any]]:
        """Return regime parameters from the first detector that supports it.

        Returns:
            Dict mapping regime index to parameter dict.

        Raises:
            AttributeError: If no detector has get_regime_params.
        """
        for det in self.detectors:
            if hasattr(det, "get_regime_params"):
                return det.get_regime_params()
        raise AttributeError("None of the ensemble detectors implement get_regime_params()")
