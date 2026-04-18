"""Forecast-driven trading signal generation.

Combines regime detection with probabilistic forecast distribution
properties to produce composite long/flat/short signals with conviction.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.stats import skew

from synthquant.forecast import ForecastResult

logger = logging.getLogger(__name__)

__all__ = ["ForecastSignal", "SignalResult"]


@dataclass(frozen=True)
class SignalResult:
    """Output of the forecast signal generator.

    The (direction, size) pair is the preferred interpretation: direction
    should come from a signal that actually predicts return sign (e.g.
    momentum); size should come from forecast-distribution uncertainty.
    The legacy (signal, conviction) pair is retained for backwards
    compatibility: by default ``direction = signal`` and ``size = conviction``.

    Attributes:
        signal: +1 (long), 0 (flat), -1 (short). Composite-mode sign.
        conviction: Strength of the signal in [0, 1]. Composite-mode magnitude.
        direction: Directional component in {-1, 0, +1}. In decomposed mode
            this comes from a directional signal (e.g. momentum) and is NOT
            a function of vol/skew/CVaR.
        size: Position size in [0, 1] derived from regime confidence and
            forecast uncertainty. Independent of direction.
        regime: Current regime index.
        regime_confidence: Probability assigned to the current regime.
        components: Individual signal components and their contributions.
    """

    signal: int
    conviction: float
    regime: int
    regime_confidence: float
    components: dict[str, float]
    direction: int | None = None
    size: float | None = None

    def __post_init__(self) -> None:
        if self.direction is None:
            object.__setattr__(self, "direction", self.signal)
        if self.size is None:
            object.__setattr__(self, "size", self.conviction)

    @property
    def position(self) -> float:
        """Signed target exposure in [-1, +1] = direction × size."""
        return float((self.direction or 0) * (self.size or 0.0))

    @property
    def is_long(self) -> bool:
        return self.signal == 1

    @property
    def is_short(self) -> bool:
        return self.signal == -1

    @property
    def is_flat(self) -> bool:
        return self.signal == 0

    def summary(self) -> dict[str, Any]:
        return {
            "signal": self.signal,
            "conviction": round(self.conviction, 3),
            "direction": self.direction,
            "size": round(self.size or 0.0, 3),
            "regime": self.regime,
            "regime_confidence": round(self.regime_confidence, 3),
            **{k: round(v, 3) for k, v in self.components.items()},
        }


class ForecastSignal:
    """Generates a composite trading signal from a ForecastResult.

    The signal combines five components, each scored on [-1, +1]:

    1. **Regime** — Is the current regime bullish (calm) or bearish (crisis)?
       Based on the regime's drift sign and the detector's confidence.

    2. **Distribution skew** — Is the forecast return distribution positively
       or negatively skewed?  Positive skew = more upside surprise potential.

    3. **Probability of profit** — What fraction of Monte Carlo paths end
       positive?  >50% favours long, <50% favours short.

    4. **Tail risk** — How severe is the downside?  Uses CVaR-95 relative to
       expected return.  Mild tails favour long, fat left tails penalise.

    5. **Volatility regime** — Is current vol elevated relative to the calm
       regime baseline?  High vol = uncertainty = lower conviction.

    6. **Momentum** — Are recent returns trending up or down?  Uses a dual
       timeframe comparison (short vs long lookback) to detect directional
       drift that the HMM may miss when volatility stays low.  Requires
       ``recent_returns`` to be passed to :meth:`generate`.

    Args:
        weights: Dict of component weights (default: balanced).
            Keys: 'regime', 'skew', 'prob_profit', 'tail_risk', 'vol',
            'momentum'.
        conviction_threshold: Minimum absolute composite score to trigger
            a non-flat signal (default: 0.15).
        allow_short: Whether short signals are permitted (default: False).
        momentum_short: Short lookback window in days (default: 21).
        momentum_long: Long lookback window in days (default: 63).
    """

    DEFAULT_WEIGHTS: dict[str, float] = {
        "regime": 0.25,
        "skew": 0.10,
        "prob_profit": 0.20,
        "tail_risk": 0.15,
        "vol": 0.10,
        "momentum": 0.20,
    }

    def __init__(
        self,
        weights: dict[str, float] | None = None,
        conviction_threshold: float = 0.15,
        allow_short: bool = False,
        momentum_short: int = 21,
        momentum_long: int = 63,
    ) -> None:
        self.weights = dict(self.DEFAULT_WEIGHTS)
        if weights:
            for k, v in weights.items():
                if k not in self.DEFAULT_WEIGHTS:
                    raise ValueError(f"Unknown weight component: {k!r}")
                self.weights[k] = v

        total = sum(self.weights.values())
        if total <= 0:
            raise ValueError("Weights must sum to a positive number")
        self.weights = {k: v / total for k, v in self.weights.items()}

        self.conviction_threshold = conviction_threshold
        self.allow_short = allow_short
        self.momentum_short = momentum_short
        self.momentum_long = momentum_long

    def generate(
        self,
        forecast: ForecastResult,
        recent_returns: np.ndarray | None = None,
    ) -> SignalResult:
        """Produce a trading signal from a forecast result.

        Args:
            forecast: A ForecastResult from ProbabilisticForecaster or
                EnsembleForecaster.
            recent_returns: Optional array of recent daily log-returns
                (most recent last).  Used for the momentum component.
                Should contain at least ``momentum_long`` observations.
                When *None*, the momentum component scores 0 (neutral).

        Returns:
            SignalResult with signal, conviction, and component breakdown.
        """
        components: dict[str, float] = {}

        # --- 1. Regime component ---
        components["regime"] = self._score_regime(forecast)

        # --- 2. Skew component ---
        components["skew"] = self._score_skew(forecast)

        # --- 3. Probability of profit ---
        components["prob_profit"] = self._score_prob_profit(forecast)

        # --- 4. Tail risk ---
        components["tail_risk"] = self._score_tail_risk(forecast)

        # --- 5. Volatility regime ---
        components["vol"] = self._score_volatility(forecast)

        # --- 6. Momentum ---
        components["momentum"] = self._score_momentum(recent_returns)

        # Weighted composite in [-1, +1]
        composite = sum(self.weights[k] * components[k] for k in self.weights)

        # Determine signal
        if composite >= self.conviction_threshold:
            signal = 1
        elif composite <= -self.conviction_threshold and self.allow_short:
            signal = -1
        else:
            signal = 0

        conviction = min(abs(composite), 1.0)

        regime_probs = forecast.regime_probabilities
        regime_confidence = float(regime_probs[forecast.current_regime])

        result = SignalResult(
            signal=signal,
            conviction=conviction,
            regime=forecast.current_regime,
            regime_confidence=regime_confidence,
            components=components,
        )

        logger.debug(
            "ForecastSignal: signal=%+d conviction=%.2f regime=%d components=%s",
            signal,
            conviction,
            forecast.current_regime,
            {k: f"{v:+.2f}" for k, v in components.items()},
        )
        return result

    def generate_decomposed(
        self,
        forecast: ForecastResult,
        direction: int,
        recent_returns: np.ndarray | None = None,
    ) -> SignalResult:
        """Produce a signal with *direction* taken externally and *size*
        derived only from non-directional forecast properties.

        Rationale: a volatility forecast tells you how wide the future
        return distribution is, not which way it will move. Mixing vol,
        skew, CVaR, and prob-positive into a directional score conflates
        uncertainty with expected return. This method keeps them separate:

        - Direction comes from the caller (e.g. a momentum signal from
          :mod:`synthquant.strategy.directional`).
        - Size in [0, 1] is regime_confidence × vol_penalty × tail_penalty.
          High vol or fat left tail shrinks size toward zero.

        Args:
            forecast: A ForecastResult from a fitted forecaster.
            direction: Externally supplied direction in {-1, 0, +1}.
            recent_returns: Used only for component diagnostics; does not
                affect direction in this method.

        Returns:
            SignalResult where ``direction`` is the caller-supplied value
            and ``size`` reflects forecast uncertainty.
        """
        if direction not in (-1, 0, 1):
            raise ValueError("direction must be one of {-1, 0, +1}")

        regime_probs = forecast.regime_probabilities
        regime_conf = float(regime_probs[forecast.current_regime])

        # Uncertainty penalties — each in [0, 1], higher = safer to size up.
        # Vol: 10%/yr → 1.0, 40%/yr → 0.0 (linear, clipped).
        ann_vol = float(np.mean(forecast.realized_vol_samples))
        vol_penalty = float(np.clip(1.0 - (ann_vol - 0.10) / 0.30, 0.0, 1.0))

        # Tail: if CVaR-95 is much worse than expected, shrink size.
        tail_drag = forecast.expected_return - forecast.cvar_95
        tail_penalty = float(np.clip(1.0 - (tail_drag - 0.03) / 0.12, 0.0, 1.0))

        size = regime_conf * vol_penalty * tail_penalty
        size = float(np.clip(size, 0.0, 1.0))

        components = {
            "regime_confidence": regime_conf,
            "vol_penalty": vol_penalty,
            "tail_penalty": tail_penalty,
            "ann_vol": ann_vol,
            "tail_drag": float(tail_drag),
            "momentum": self._score_momentum(recent_returns),
        }

        return SignalResult(
            signal=int(direction),
            conviction=size,
            regime=forecast.current_regime,
            regime_confidence=regime_conf,
            components=components,
            direction=int(direction),
            size=size,
        )

    # ------------------------------------------------------------------
    # Component scorers — each returns a value in [-1, +1]
    # ------------------------------------------------------------------

    @staticmethod
    def _score_regime(fc: ForecastResult) -> float:
        """Score based on current regime drift and confidence.

        Calm regime with positive mu → +1.
        Crisis regime with negative mu → -1.
        Scaled by regime probability (confidence).
        """
        regime = fc.current_regime
        rp = fc.regime_params.get(regime, {})
        mu = rp.get("mu", 0.0)

        # Regime probability as confidence
        confidence = float(fc.regime_probabilities[regime])

        # Map mu to [-1, +1] using tanh — saturates around ±0.15/day
        # (|mu| * 252 ≈ 38%/yr maps to ~0.9)
        score = float(np.tanh(mu * 252 / 0.40))
        return score * confidence

    @staticmethod
    def _score_skew(fc: ForecastResult) -> float:
        """Score based on forecast return distribution skewness.

        Positive skew → more upside surprise potential → bullish.
        Negative skew → fatter left tail → bearish.
        """
        sk = float(skew(fc.terminal_returns))
        # tanh saturates: skew ±2 → score ±0.96
        return float(np.tanh(sk / 2.0))

    @staticmethod
    def _score_prob_profit(fc: ForecastResult) -> float:
        """Score based on fraction of paths ending positive.

        Linearly maps [0, 1] → [-1, +1].
        50% → 0, 70% → +0.4, 90% → +0.8.
        """
        return 2.0 * fc.prob_positive - 1.0

    @staticmethod
    def _score_tail_risk(fc: ForecastResult) -> float:
        """Score based on downside severity (CVaR-95 vs expected return).

        If CVaR is mild relative to the expected return, that's positive.
        If CVaR is extreme relative to expected return, that's negative.
        """
        expected = fc.expected_return
        cvar = fc.cvar_95  # negative number = loss

        # Tail drag = how much worse CVaR is than the mean
        # For SPY over 21d: expected ~+1%, CVaR ~-5% → drag = 6%
        tail_drag = expected - cvar  # positive number

        # Map: small drag (< 3%) → +1, large drag (> 15%) → -1
        # Linear scale centred at 8% drag → 0
        score = 1.0 - (tail_drag - 0.03) / 0.06
        return float(np.clip(score, -1.0, 1.0))

    @staticmethod
    def _score_volatility(fc: ForecastResult) -> float:
        """Score based on current vol level.

        Low vol → favours conviction in directional trades → positive.
        High vol → uncertainty → negative (reduces conviction).
        """
        vol = fc.return_std
        horizon_days = fc.horizon

        # Annualise
        ann_vol = vol * np.sqrt(252 / horizon_days) if horizon_days > 0 else vol

        # Map: 10% vol → +0.8, 20% → +0.2, 30% → -0.4, 40% → -1.0
        score = 1.0 - (ann_vol - 0.10) / 0.15
        return float(np.clip(score, -1.0, 1.0))

    def _score_momentum(self, recent_returns: np.ndarray | None) -> float:
        """Score based on recent price trend using dual-timeframe comparison.

        Compares the short-lookback cumulative return against the
        long-lookback cumulative return.  A positive short-term trend
        scores positively; a negative trend scores negatively.  The
        long-term trend provides confirmation: agreement amplifies the
        score, disagreement dampens it.

        Returns 0.0 (neutral) when *recent_returns* is None or too short.
        """
        if recent_returns is None or len(recent_returns) < self.momentum_short:
            return 0.0

        rets = np.asarray(recent_returns, dtype=float)

        short_ret = float(np.sum(rets[-self.momentum_short :]))
        long_n = min(self.momentum_long, len(rets))
        long_ret = float(np.sum(rets[-long_n:]))

        # Annualise for consistent scaling (21d cumulative → annualised)
        short_ann = short_ret * (252 / self.momentum_short)
        long_ann = long_ret * (252 / long_n)

        # Primary signal: short-term trend, tanh-saturated at ~±40%/yr
        primary = float(np.tanh(short_ann / 0.40))

        # Confirmation multiplier: same sign → 1.0, opposite → 0.5
        confirm = 1.0 if short_ann * long_ann >= 0 else 0.5

        return float(np.clip(primary * confirm, -1.0, 1.0))
