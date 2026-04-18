"""Regime-conditional position sizing.

A simpler alternative to the composite :class:`ForecastSignal` approach.
Instead of scoring six signal components and applying a conviction
threshold, this allocator maps regime probabilities directly to equity
exposure:

    exposure = w_calm × P(calm) + w_crisis × P(crisis)

In calm markets you stay (nearly) fully invested to capture the equity
risk premium.  In crisis markets you reduce exposure to limit drawdowns.
The allocation varies continuously with the regime posterior — no binary
long/flat decision and no conviction dampening.

**Momentum re-entry**: By default the allocator detects when exposure
has been reduced (crisis regime) but recent returns are positive
(recovery under way) and ramps exposure back up faster than the raw
regime probabilities alone would suggest.  This addresses the common
HMM lag problem where volatility stays elevated during V-shaped
recoveries, keeping allocation low long after the bottom.

Typical usage::

    allocator = RegimeAllocator(calm_weight=1.0, crisis_weight=0.25)
    fc = EnsembleForecaster(use_filtered=True)
    fc.fit(window)
    forecast = fc.forecast(horizon=21, current_price=price)
    result = allocator.allocate(forecast, recent_returns=rets)
    # result.exposure ∈ [0.25, 1.0]
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from synthquant.forecast import ForecastResult

logger = logging.getLogger(__name__)

__all__ = ["RegimeAllocator", "AllocationResult"]


@dataclass(frozen=True)
class AllocationResult:
    """Output of regime-conditional allocation.

    Attributes:
        exposure: Target equity exposure in [0, 1].
        regime: Current regime index (0 = calm, 1 = crisis with anchoring).
        regime_confidence: Probability assigned to the current regime.
        regime_probabilities: Full regime probability vector.
        vol_scale: GARCH-based volatility scaling factor applied.
        momentum_boost: Exposure added by momentum re-entry (0 if disabled).
    """

    exposure: float
    regime: int
    regime_confidence: float
    regime_probabilities: np.ndarray
    vol_scale: float
    momentum_boost: float = 0.0

    def summary(self) -> dict[str, Any]:
        return {
            "exposure": round(self.exposure, 3),
            "regime": self.regime,
            "regime_confidence": round(self.regime_confidence, 3),
            "vol_scale": round(self.vol_scale, 3),
            "momentum_boost": round(self.momentum_boost, 3),
        }


class RegimeAllocator:
    """Maps regime probabilities to equity exposure.

    With label-anchored detectors (state 0 = calm, state 1 = crisis),
    this produces a continuous exposure between ``crisis_weight`` and
    ``calm_weight`` based on the posterior probability of each regime.

    **Momentum re-entry** addresses the classic HMM lag: during a
    V-shaped recovery volatility stays elevated so the HMM still reports
    "crisis", but prices are already rising.  When enabled, if exposure
    has been reduced by the regime signal *and* trailing returns are
    positive (i.e. a recovery is under way), exposure is boosted back
    towards ``calm_weight`` proportionally to the strength of the
    recent rally.  This makes de-risking fast (follows HMM immediately)
    but re-risking adaptive (momentum accelerates the return to full
    investment).

    An optional volatility overlay scales exposure down when realised
    vol exceeds a threshold (vol targeting).

    Args:
        calm_weight: Equity exposure when fully in the calm regime.
        crisis_weight: Equity exposure when fully in the crisis regime.
        vol_target: Annualised volatility target for vol scaling.
            Set to *None* to disable vol targeting.  When enabled,
            exposure is further scaled by min(vol_target / realised_vol, 1).
        vol_lookback: Number of days for realised vol estimation.
        min_exposure: Floor on exposure (never go below this).
        reentry_lookback: Number of days of trailing returns used to
            detect recovery momentum.  Set to 0 to disable momentum
            re-entry.
        reentry_scale: The trailing cumulative return over the lookback
            window at which the exposure gap is fully closed.  For
            example, 0.04 means a +4% trailing return restores full
            ``calm_weight`` exposure even if the HMM still reports
            crisis.
    """

    def __init__(
        self,
        calm_weight: float = 1.0,
        crisis_weight: float = 0.25,
        vol_target: float | None = None,
        vol_lookback: int = 21,
        min_exposure: float = 0.0,
        reentry_lookback: int = 20,
        reentry_scale: float = 0.04,
    ) -> None:
        if not 0 <= crisis_weight <= calm_weight <= 1.0:
            raise ValueError(
                f"Need 0 <= crisis_weight ({crisis_weight}) <= calm_weight ({calm_weight}) <= 1"
            )
        self.calm_weight = calm_weight
        self.crisis_weight = crisis_weight
        self.vol_target = vol_target
        self.vol_lookback = vol_lookback
        self.min_exposure = min_exposure
        self.reentry_lookback = reentry_lookback
        self.reentry_scale = reentry_scale

    def allocate(
        self,
        forecast: ForecastResult,
        recent_returns: np.ndarray | None = None,
    ) -> AllocationResult:
        """Compute target exposure from a forecast's regime probabilities.

        Args:
            forecast: A ForecastResult from EnsembleForecaster.
            recent_returns: Recent daily log-returns (most recent last).
                Used for vol targeting when ``vol_target`` is set.

        Returns:
            AllocationResult with continuous exposure.
        """
        proba = forecast.regime_probabilities
        regime = forecast.current_regime
        confidence = float(proba[regime])

        # Regime-weighted exposure:
        # With 2 regimes and anchoring, state 0 = calm, state 1 = crisis.
        # Generalises to N regimes by interpolating linearly with vol rank.
        n_regimes = len(proba)
        if n_regimes == 2:
            exposure = self.calm_weight * float(proba[0]) + self.crisis_weight * float(proba[1])
        else:
            # For >2 regimes, interpolate weights by state index
            # (states ordered by ascending vol thanks to anchoring)
            weights = np.linspace(self.calm_weight, self.crisis_weight, n_regimes)
            exposure = float(np.dot(weights, proba))

        # Momentum re-entry: if exposure is reduced (crisis detected) but
        # trailing returns are positive (recovery under way), ramp exposure
        # back toward calm_weight proportionally.  De-risking stays fast
        # (follows HMM directly); re-risking is accelerated by momentum.
        momentum_boost = 0.0
        if self.reentry_lookback > 0 and recent_returns is not None and exposure < self.calm_weight:
            trail = recent_returns[-self.reentry_lookback :]
            if len(trail) >= 5:
                trailing_return = float(np.sum(trail))
                if trailing_return > 0 and self.reentry_scale > 0:
                    recovery = min(trailing_return / self.reentry_scale, 1.0)
                    gap = self.calm_weight - exposure
                    momentum_boost = recovery * gap
                    exposure += momentum_boost

        # Volatility targeting overlay
        vol_scale = 1.0
        if self.vol_target is not None and recent_returns is not None:
            lookback_rets = recent_returns[-self.vol_lookback :]
            if len(lookback_rets) >= 5:
                realised_vol = float(np.std(lookback_rets)) * np.sqrt(252)
                if realised_vol > 0:
                    vol_scale = min(self.vol_target / realised_vol, 1.0)
                    exposure *= vol_scale

        exposure = max(exposure, self.min_exposure)
        exposure = min(exposure, 1.0)

        result = AllocationResult(
            exposure=exposure,
            regime=regime,
            regime_confidence=confidence,
            regime_probabilities=proba,
            vol_scale=vol_scale,
            momentum_boost=momentum_boost,
        )

        logger.debug(
            "RegimeAllocator: exposure=%.2f regime=%d vol_scale=%.2f momentum_boost=%.3f",
            exposure,
            regime,
            vol_scale,
            momentum_boost,
        )
        return result
