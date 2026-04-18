"""Predictive allocator — combines regime detection with leading indicators.

The :class:`RegimeAllocator` is reactive: it maps the *current* regime
posterior to an exposure level, so it can only match buy-and-hold at best.

The :class:`PredictiveAllocator` layers **forward-looking signals** on top
of the regime posterior.  The idea:

    exposure = base_regime_exposure + adjustment × predictive_composite

Where ``base_regime_exposure`` comes from the same regime-weighted formula
as :class:`RegimeAllocator`, and ``adjustment`` shifts exposure toward
positions that *anticipate* regime transitions.

When the predictive composite is positive (recovery / calm expected), the
allocator increases exposure *before* the HMM catches up.  When negative
(crisis building), it reduces exposure *before* vol spikes show up in
the posterior.

This is where the actual edge over buy-and-hold should come from:
anticipating rather than reacting.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from synthquant.forecast import ForecastResult
from synthquant.strategy.predictive_signals import (
    PredictiveSignals,
    SignalOutput,
)

logger = logging.getLogger(__name__)

__all__ = ["PredictiveAllocator", "PredictiveAllocationResult"]


@dataclass(frozen=True)
class PredictiveAllocationResult:
    """Output of predictive allocation.

    Attributes:
        exposure: Final target equity exposure in [0, 1].
        base_exposure: Regime-only exposure (before predictive adjustment).
        predictive_adjustment: Change in exposure from predictive signals.
        signal: The full SignalOutput with component scores.
        regime: Current regime index.
        regime_confidence: Probability of the current regime.
        regime_probabilities: Full posterior.
        vol_scale: Volatility targeting scale (1.0 if disabled).
    """

    exposure: float
    base_exposure: float
    predictive_adjustment: float
    signal: SignalOutput
    regime: int
    regime_confidence: float
    regime_probabilities: np.ndarray
    vol_scale: float

    def summary(self) -> dict[str, Any]:
        return {
            "exposure": round(self.exposure, 3),
            "base_exposure": round(self.base_exposure, 3),
            "predictive_adj": round(self.predictive_adjustment, 3),
            "composite_signal": round(self.signal.composite, 4),
            "regime": self.regime,
            "regime_confidence": round(self.regime_confidence, 3),
            "vol_scale": round(self.vol_scale, 3),
        }


class PredictiveAllocator:
    """Regime allocation enhanced with predictive leading indicators.

    Args:
        calm_weight: Exposure when fully in calm regime.
        crisis_weight: Exposure when fully in crisis regime.
        predictive_strength: How much the predictive signal can shift
            exposure.  At 1.0, a maximal predictive signal can shift
            exposure by the full (calm_weight - crisis_weight) range.
            At 0.5, it can shift by half that range.  Default 0.5.
        vol_target: Annualised vol target for vol scaling (None to disable).
        vol_lookback: Days for realised vol estimate.
        min_exposure: Floor on exposure.
        signal_weights: Weights for individual predictive signals.
            Passed to :class:`PredictiveSignals`.
        sensitivity: Sensitivity multiplier for the predictive composite.
    """

    def __init__(
        self,
        calm_weight: float = 1.0,
        crisis_weight: float = 0.25,
        predictive_strength: float = 0.5,
        vol_target: float | None = None,
        vol_lookback: int = 21,
        min_exposure: float = 0.0,
        signal_weights: dict[str, float] | None = None,
        sensitivity: float = 1.0,
    ) -> None:
        if not 0 <= crisis_weight <= calm_weight <= 1.0:
            raise ValueError(
                f"Need 0 <= crisis_weight ({crisis_weight}) <= calm_weight ({calm_weight}) <= 1"
            )
        self.calm_weight = calm_weight
        self.crisis_weight = crisis_weight
        self.predictive_strength = predictive_strength
        self.vol_target = vol_target
        self.vol_lookback = vol_lookback
        self.min_exposure = min_exposure
        self._signals = PredictiveSignals(
            signal_weights=signal_weights,
            sensitivity=sensitivity,
        )

    def allocate(
        self,
        forecast: ForecastResult,
        close: np.ndarray,
        returns: np.ndarray,
        high: np.ndarray | None = None,
        low: np.ndarray | None = None,
        volume: np.ndarray | None = None,
        safe_haven_returns: np.ndarray | None = None,
        market_data: Any = None,
    ) -> PredictiveAllocationResult:
        """Compute target exposure using regime + predictive signals.

        Args:
            forecast: A ForecastResult from EnsembleForecaster.
            close: Daily close prices (most recent last).
            returns: Daily log returns (most recent last).
            high: Daily high prices (optional).
            low: Daily low prices (optional).
            volume: Daily volume (optional).
            safe_haven_returns: Safe-haven asset returns (optional).

        Returns:
            PredictiveAllocationResult with exposure and diagnostics.
        """
        proba = forecast.regime_probabilities
        regime = forecast.current_regime
        confidence = float(proba[regime])

        # ---- 1. Base regime-weighted exposure (same as RegimeAllocator) ----
        n_regimes = len(proba)
        if n_regimes == 2:
            base_exposure = self.calm_weight * float(proba[0]) + self.crisis_weight * float(
                proba[1]
            )
        else:
            weights = np.linspace(self.calm_weight, self.crisis_weight, n_regimes)
            base_exposure = float(np.dot(weights, proba))

        # ---- 2. Compute predictive signals ----
        signal = self._signals.compute(
            close=close,
            returns=returns,
            high=high,
            low=low,
            volume=volume,
            safe_haven_returns=safe_haven_returns,
            market_data=market_data,
        )

        # ---- 3. Apply predictive adjustment ----
        # The adjustment range is proportional to the gap between calm/crisis
        max_adjustment = (self.calm_weight - self.crisis_weight) * self.predictive_strength
        adjustment = signal.composite * max_adjustment

        exposure = base_exposure + adjustment

        # ---- 4. Volatility targeting overlay ----
        vol_scale = 1.0
        if self.vol_target is not None and len(returns) >= 5:
            lookback_rets = returns[-self.vol_lookback :]
            if len(lookback_rets) >= 5:
                realised_vol = float(np.std(lookback_rets)) * np.sqrt(252)
                if realised_vol > 0:
                    vol_scale = min(self.vol_target / realised_vol, 1.0)
                    exposure *= vol_scale

        # ---- 5. Clamp to valid range ----
        exposure = max(self.min_exposure, min(1.0, exposure))

        result = PredictiveAllocationResult(
            exposure=exposure,
            base_exposure=base_exposure,
            predictive_adjustment=adjustment,
            signal=signal,
            regime=regime,
            regime_confidence=confidence,
            regime_probabilities=proba,
            vol_scale=vol_scale,
        )

        logger.debug(
            "PredictiveAllocator: exposure=%.2f base=%.2f adj=%.3f composite=%.3f",
            exposure,
            base_exposure,
            adjustment,
            signal.composite,
        )
        return result
