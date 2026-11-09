"""Bayesian adjustment of HMM regime probabilities using market-implied data.

The HMM estimates ``P(regime | returns)`` from equity returns alone.
Market-implied data — VIX term structure, credit spreads, MOVE index,
SKEW index — provides **independent** evidence about the current market
regime.  These are prices set by options and credit traders who may have
information the HMM doesn't see.

This module combines both sources via a log-odds Bayesian update:

    adjusted_log_odds = hmm_log_odds + strength × crisis_score

Where ``crisis_score ∈ [-1, +1]`` is a weighted composite of market-implied
signals.  A positive score means market data says "more crisis than the
HMM thinks"; negative means "calmer".

The adjusted probabilities feed into :class:`EnsembleForecaster` to control:

    1. **Engine weights**: More crisis → more Merton JD (jump) paths.
    2. **Regime parameter selection**: Flipped argmax → crisis-calibrated
       drift, vol, and jump params drive the simulation.
    3. **Allocation**: ``RegimeAllocator`` / ``PredictiveAllocator`` use the
       adjusted posteriors for exposure sizing.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from synthquant.data.market_data import MarketSnapshot

from synthquant.strategy.predictive_signals import (
    _clip_signal,
    _credit_spread_signal,
    _move_index_signal,
    _skew_signal,
    _vix_term_structure_signal,
)

logger = logging.getLogger(__name__)

__all__ = ["MarketRegimeAdjuster"]

DEFAULT_SIGNAL_WEIGHTS: dict[str, float] = {
    "vix_term_structure": 2.0,
    "credit_spread": 1.8,
    "move": 1.2,
    "skew": 1.0,
}

# Maximum log-odds shift when strength=1.0 and crisis_score=±1.0.
# 2.0 shifts P(crisis) from 15% → ~45% at maximum; moderate enough
# to be informative without overwhelming the HMM.
_MAX_LOG_ODDS_SHIFT = 2.0


class MarketRegimeAdjuster:
    """Adjust HMM regime probabilities using market-implied evidence.

    Args:
        strength: How much to trust market data vs HMM alone.
            0.0 = no adjustment (pass-through).
            1.0 = full Bayesian update (max shift ≈ 2.0 log-odds).
            Default 0.5.
        signal_weights: Relative importance of each market signal.
            Only signals with weight > 0 and available data are used.
    """

    def __init__(
        self,
        strength: float = 0.5,
        signal_weights: dict[str, float] | None = None,
    ) -> None:
        if not 0.0 <= strength <= 1.0:
            raise ValueError(f"strength must be in [0, 1], got {strength}")
        self.strength = strength
        self.signal_weights = dict(signal_weights or DEFAULT_SIGNAL_WEIGHTS)

    def compute_crisis_score(
        self,
        market_data: MarketSnapshot,
    ) -> float:
        """Compute composite crisis evidence from market-implied data.

        Returns a score in [-1, +1]:
            Positive → market signals say "crisis" (more than HMM sees).
            Negative → market signals say "calm".
            Zero → no evidence or no data.
        """
        components: dict[str, float] = {}
        active_weights: dict[str, float] = {}

        w = self.signal_weights.get("vix_term_structure", 0)
        vix = market_data.vix if market_data is not None else None
        vix3m = market_data.vix3m if market_data is not None else None
        if w > 0 and vix is not None and vix3m is not None:
            # Signal is positive = bullish (contango). We want positive = crisis.
            # So negate: crisis_evidence = -signal
            sig = _vix_term_structure_signal(vix, vix3m)
            components["vix_term_structure"] = -sig
            active_weights["vix_term_structure"] = w

        w = self.signal_weights.get("skew", 0)
        skew_arr = market_data.skew if market_data is not None else None
        if w > 0 and skew_arr is not None:
            sig = _skew_signal(skew_arr)
            # Signal is negative when SKEW high (bearish). Negate for crisis.
            components["skew"] = -sig
            active_weights["skew"] = w

        w = self.signal_weights.get("credit_spread", 0)
        hyg = market_data.hyg_close if market_data is not None else None
        ief = market_data.ief_close if market_data is not None else None
        if w > 0 and hyg is not None and ief is not None:
            sig = _credit_spread_signal(hyg, ief)
            # Signal is positive = HY outperforming = risk-on. Negate for crisis.
            components["credit_spread"] = -sig
            active_weights["credit_spread"] = w

        w = self.signal_weights.get("move", 0)
        move_arr = market_data.move if market_data is not None else None
        if w > 0 and move_arr is not None:
            sig = _move_index_signal(move_arr)
            # Signal is negative when MOVE high (bearish). Negate for crisis.
            components["move"] = -sig
            active_weights["move"] = w

        if not active_weights:
            return 0.0

        total_weight = sum(active_weights.values())
        composite = sum(components[k] * active_weights[k] for k in active_weights) / total_weight

        return float(_clip_signal(composite))

    def adjust(
        self,
        hmm_proba: np.ndarray,
        market_data: MarketSnapshot | None,
    ) -> np.ndarray:
        """Adjust HMM regime probabilities using market-implied evidence.

        Uses a log-odds Bayesian update on the crisis state (last regime,
        highest volatility by HMM convention).

        Args:
            hmm_proba: HMM posterior, shape ``(n_regimes,)``, sums to 1.
            market_data: Market snapshot aligned to the observation window.
                If None, returns ``hmm_proba`` unchanged.

        Returns:
            Adjusted probabilities, same shape, sums to 1.
        """
        if market_data is None or self.strength == 0.0:
            return hmm_proba

        crisis_score = self.compute_crisis_score(market_data)
        if abs(crisis_score) < 1e-6:
            return hmm_proba

        n = len(hmm_proba)
        if n < 2:
            return hmm_proba

        # For 2-regime case: log-odds update on crisis state (index n-1)
        # For N-regime: shift probability mass between calmest and most
        # crisis-like states, leaving middle states' relative weights fixed.
        crisis_idx = n - 1
        calm_idx = 0

        adjusted = hmm_proba.copy().astype(float)

        # Clamp probabilities away from 0/1 before log-odds transform
        eps = 1e-4
        adjusted = np.clip(adjusted, eps, 1.0 - eps)
        adjusted /= adjusted.sum()

        # Log-odds shift: positive crisis_score → increase P(crisis)
        log_odds_shift = self.strength * crisis_score * _MAX_LOG_ODDS_SHIFT

        if n == 2:
            # Simple 2-state log-odds update
            log_odds = np.log(adjusted[crisis_idx] / adjusted[calm_idx])
            new_log_odds = log_odds + log_odds_shift
            p_crisis = 1.0 / (1.0 + np.exp(-new_log_odds))
            adjusted[crisis_idx] = p_crisis
            adjusted[calm_idx] = 1.0 - p_crisis
        else:
            # Multi-regime: shift mass from calm ↔ crisis, keep middle
            # states proportionally fixed relative to each other.
            p_crisis = adjusted[crisis_idx]
            p_calm = adjusted[calm_idx]
            p_middle = 1.0 - p_crisis - p_calm

            log_odds = np.log(p_crisis / max(p_calm, eps))
            new_log_odds = log_odds + log_odds_shift

            new_ratio = np.exp(new_log_odds)
            # p_crisis / p_calm = new_ratio
            # p_calm + p_crisis = 1 - p_middle
            # p_calm = (1 - p_middle) / (1 + new_ratio)
            remaining = 1.0 - p_middle
            new_calm = remaining / (1.0 + new_ratio)
            new_crisis = remaining - new_calm

            adjusted[calm_idx] = new_calm
            adjusted[crisis_idx] = new_crisis
            # Middle states keep their relative proportions

        # Final safety: ensure valid probability distribution
        adjusted = np.clip(adjusted, 0.0, 1.0)
        adjusted /= adjusted.sum()

        logger.debug(
            f"MarketRegimeAdjuster: score={crisis_score:+.3f}, "
            f"strength={self.strength:.2f}, shift={log_odds_shift:+.3f}, "
            f"P(crisis) {hmm_proba[crisis_idx]:.3f} → {adjusted[crisis_idx]:.3f}"
        )

        return adjusted
