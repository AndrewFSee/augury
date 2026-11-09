"""Predictive leading indicators for regime-change anticipation.

The regime allocator detects regimes *concurrently* — it identifies a crisis
after it has already begun, and declares calm only after volatility has
already subsided.  This module computes **leading indicators** from OHLCV
data that tend to move *before* regime transitions, giving the allocator
a predictive edge over pure reactive regime detection.

Signals are scaled to [-1, +1] where:
  - Positive = bullish / risk-on (increase exposure)
  - Negative = bearish / risk-off (decrease exposure)
  - Zero = neutral

Categories of predictive signals:

**Volatility anticipation** — detect vol building before it spikes:
  - Vol term structure (short/long vol ratio)
  - Vol regime change rate (acceleration)
  - Range expansion (intraday High-Low widening)

**Trend exhaustion / mean reversion** — detect when moves are overextended:
  - Price distance from MA (z-score)
  - Trend deceleration (momentum of momentum)
  - RSI-style overbought/oversold

**Volume confirmation** — price moves without volume support are fragile:
  - Volume trend (rising/falling)
  - Price-volume divergence (prices up but volume fading)

**Cross-asset regime lead** — risk-off assets move before equities:
  - Equity/bond relative strength
  - Gold momentum (flight-to-safety signal)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from synthquant.data.market_data import MarketSnapshot

__all__ = [
    "PredictiveSignals",
    "SignalOutput",
    "compute_predictive_signals",
]


@dataclass(frozen=True)
class SignalOutput:
    """Output from predictive signal computation.

    Attributes:
        composite: Combined signal in [-1, +1].  Positive = bullish.
        components: Individual signal values keyed by name.
        weights: Weight used for each component in the composite.
    """

    composite: float
    components: dict[str, float]
    weights: dict[str, float]

    def summary(self) -> dict[str, Any]:
        return {
            "composite": round(self.composite, 4),
            **{k: round(v, 4) for k, v in self.components.items()},
        }


# ------------------------------------------------------------------ #
# Helper functions
# ------------------------------------------------------------------ #


def _rolling_std(arr: np.ndarray, window: int) -> np.ndarray:
    """Rolling std with expanding warmup.  Returns array of same length."""
    n = len(arr)
    out = np.empty(n, dtype=float)
    for t in range(n):
        w = arr[max(0, t - window + 1) : t + 1]
        out[t] = np.std(w, ddof=0) if len(w) > 1 else 0.0
    return out


def _rolling_mean(arr: np.ndarray, window: int) -> np.ndarray:
    n = len(arr)
    out = np.empty(n, dtype=float)
    for t in range(n):
        out[t] = np.mean(arr[max(0, t - window + 1) : t + 1])
    return out


def _ema(arr: np.ndarray, span: int) -> np.ndarray:
    alpha = 2.0 / (span + 1)
    out = np.empty(len(arr), dtype=float)
    out[0] = arr[0]
    for t in range(1, len(arr)):
        out[t] = alpha * arr[t] + (1 - alpha) * out[t - 1]
    return out


def _clip_signal(x: float) -> float:
    """Clip to [-1, +1]."""
    return max(-1.0, min(1.0, x))


# ------------------------------------------------------------------ #
# Individual signal builders
# ------------------------------------------------------------------ #


def _vol_term_structure(returns: np.ndarray, short: int = 5, long: int = 63) -> float:
    """Short-term vol vs long-term vol.

    When short vol >> long vol, a regime change is building.
    When short vol << long vol, crisis is fading → recovery signal.

    Returns signal in [-1, +1]:  positive = short vol below long (bullish),
    negative = short vol above long (bearish).
    """
    if len(returns) < long:
        return 0.0
    short_vol = np.std(returns[-short:])
    long_vol = np.std(returns[-long:])
    if long_vol < 1e-10:
        return 0.0
    ratio = short_vol / long_vol
    # ratio > 1 = short vol elevated = bearish
    # ratio < 1 = short vol compressed = bullish
    # Map: ratio=0.5 → +1, ratio=1.0 → 0, ratio=2.0 → -1
    signal = -(ratio - 1.0)
    return _clip_signal(signal)


def _vol_acceleration(returns: np.ndarray, window: int = 21) -> float:
    """Rate of change of volatility — is vol increasing or decreasing?

    Positive signal = vol declining (bullish for risk assets).
    Negative signal = vol accelerating (bearish).
    """
    if len(returns) < window + 5:
        return 0.0
    vols = _rolling_std(returns, 5)
    recent_vol = np.mean(vols[-5:])
    prior_vol = np.mean(vols[-(window):-5]) if len(vols) > window else np.mean(vols[:-5])
    if prior_vol < 1e-10:
        return 0.0
    change = (recent_vol - prior_vol) / prior_vol
    # Vol declining → positive signal; vol rising → negative
    signal = -change * 2.0
    return _clip_signal(signal)


def _range_expansion(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    short: int = 5,
    long: int = 63,
) -> float:
    """Intraday range expansion — widening ranges precede breakouts/breakdowns.

    Compares recent average true range to historical average.
    Expanding range + down direction = bearish signal.
    Contracting range = calm signal (bullish for risk).
    """
    if len(high) < long:
        return 0.0
    ranges = (high - low) / np.maximum(close, 1e-8)
    recent_range = np.mean(ranges[-short:])
    long_range = np.mean(ranges[-long:])
    long_std = np.std(ranges[-long:])
    if long_std < 1e-10:
        return 0.0
    z = (recent_range - long_range) / long_std
    # Expanding range = bearish (negative); contracting = bullish (positive)
    signal = -z * 0.3
    return _clip_signal(signal)


def _price_distance_from_ma(
    close: np.ndarray,
    ma_window: int = 50,
    std_window: int = 252,
) -> float:
    """Z-score of price relative to moving average.

    Extreme negative → mean-reversion buy signal.
    Extreme positive → potential mean-reversion sell.
    Moderate → trend-following (stays with direction).
    """
    if len(close) < ma_window + 10:
        return 0.0
    ma = _rolling_mean(close, ma_window)
    dev = close - ma
    window = min(std_window, len(close))
    dev_std = np.std(dev[-window:])
    if dev_std < 1e-10:
        return 0.0
    z = dev[-1] / dev_std
    # Mild z (|z| < 1.5): trend following (positive z → bullish)
    # Extreme z (|z| > 2.5): mean-reversion (invert)
    abs_z = abs(z)
    if abs_z < 1.5:
        signal = z * 0.3  # mild trend-following
    elif abs_z < 2.5:
        signal = z * 0.15  # fading trend signal
    else:
        signal = -z * 0.1  # mean-reversion contrarian
    return _clip_signal(signal)


def _trend_deceleration(returns: np.ndarray, short: int = 5, long: int = 21) -> float:
    """Momentum of momentum — is the trend accelerating or stalling?

    Positive = momentum accelerating in upward direction.
    Negative = momentum decelerating / reversing.
    """
    if len(returns) < long + 5:
        return 0.0
    mom_short = np.sum(returns[-short:])
    mom_long = np.sum(returns[-long:]) / (long / short)
    # If short mom > long mom → accelerating; if short < long → decelerating
    diff = mom_short - mom_long
    # Scale by recent vol for normalization
    vol = np.std(returns[-long:])
    if vol < 1e-10:
        return 0.0
    signal = diff / (vol * np.sqrt(short)) * 0.5
    return _clip_signal(signal)


def _rsi_signal(returns: np.ndarray, window: int = 14) -> float:
    """RSI-based mean-reversion signal.

    RSI < 30 → oversold (bullish signal).
    RSI > 70 → overbought (bearish signal).
    30-70 → neutral, slight trend-following.
    """
    if len(returns) < window + 1:
        return 0.0
    r = returns[-window:]
    up = np.where(r > 0, r, 0.0)
    down = np.where(r < 0, -r, 0.0)
    avg_up = np.mean(up)
    avg_down = np.mean(down)
    if avg_down < 1e-10:
        rsi = 100.0
    else:
        rs = avg_up / avg_down
        rsi = 100.0 - (100.0 / (1.0 + rs))
    # Map RSI to [-1, +1]
    if rsi < 30:
        signal = (30 - rsi) / 30  # 0 to +1 (oversold → buy)
    elif rsi > 70:
        signal = -(rsi - 70) / 30  # 0 to -1 (overbought → sell)
    else:
        signal = (rsi - 50) / 50 * 0.2  # mild trend-following
    return _clip_signal(signal)


def _volume_trend(volume: np.ndarray, short: int = 5, long: int = 21) -> float:
    """Volume trend — rising volume confirms moves, falling volume weakens them.

    Returns a volume momentum score, not directional itself.
    Needs to be combined with price direction for the signal.
    """
    if len(volume) < long:
        return 0.0
    vol_short = np.mean(volume[-short:])
    vol_long = np.mean(volume[-long:])
    if vol_long < 1e-10:
        return 0.0
    return float((vol_short / vol_long) - 1.0)


def _price_volume_divergence(
    returns: np.ndarray,
    volume: np.ndarray,
    window: int = 21,
) -> float:
    """Detect price-volume divergence.

    Price rising + volume falling → weak rally → bearish signal.
    Price falling + volume falling → weak selloff → bullish signal.
    Price rising + volume rising → confirmed rally → bullish.
    """
    if len(returns) < window or len(volume) < window:
        return 0.0
    price_trend = np.sum(returns[-window:])
    vol_ratio = _volume_trend(volume, 5, window)

    if price_trend > 0 and vol_ratio < -0.1:
        # Prices rising but volume fading — bearish divergence
        signal = -min(abs(vol_ratio), 0.5) * 2
    elif price_trend < 0 and vol_ratio < -0.1:
        # Prices falling but volume fading — selling exhaustion (bullish)
        signal = min(abs(vol_ratio), 0.5) * 1.5
    elif price_trend > 0 and vol_ratio > 0.1:
        # Confirmed rally
        signal = min(vol_ratio, 0.5) * 1.0
    elif price_trend < 0 and vol_ratio > 0.1:
        # Confirmed selloff — bearish
        signal = -min(vol_ratio, 0.5) * 1.0
    else:
        signal = 0.0
    return _clip_signal(signal)


# ------------------------------------------------------------------ #
# Cross-asset signals  (optional — need extra ticker data)
# ------------------------------------------------------------------ #


def _cross_asset_signal(
    equity_returns: np.ndarray,
    safe_haven_returns: np.ndarray | None,
    window: int = 21,
) -> float:
    """Relative strength: equity vs safe-haven.

    When safe-haven outperforms equity → flight-to-safety → bearish for equity.
    When equity outperforms safe-haven → risk-on → bullish for equity.
    """
    if safe_haven_returns is None or len(safe_haven_returns) < window:
        return 0.0
    if len(equity_returns) < window:
        return 0.0
    eq_mom = np.sum(equity_returns[-window:])
    sh_mom = np.sum(safe_haven_returns[-window:])
    diff = eq_mom - sh_mom
    vol = np.std(equity_returns[-window:])
    if vol < 1e-10:
        return 0.0
    signal = diff / (vol * np.sqrt(window)) * 0.5
    return _clip_signal(signal)


# ------------------------------------------------------------------ #
# Market-implied signals  (options / credit / bond vol)
# ------------------------------------------------------------------ #


def _vix_term_structure_signal(
    vix: np.ndarray | None,
    vix3m: np.ndarray | None,
    lookback: int = 5,
) -> float:
    """VIX / VIX3M term structure — the single strongest leading indicator.

    Normal (contango): VIX < VIX3M → market complacent → bullish.
    Inverted (backwardation): VIX > VIX3M → near-term fear → bearish.
    The *change* in the ratio also matters: steepening = improving,
    flattening/inverting = deteriorating.

    Returns signal in [-1, +1].  Positive = contango (bullish).
    """
    if vix is None or vix3m is None:
        return 0.0
    if len(vix) < lookback + 1 or len(vix3m) < lookback + 1:
        return 0.0
    # Avoid division issues
    v3 = vix3m[-lookback:]
    v1 = vix[-lookback:]
    valid = v3 > 1.0  # filter out any zeros
    if not np.any(valid):
        return 0.0
    ratios = v1[valid] / v3[valid]
    current_ratio = ratios[-1]

    # Ratio < 1 = contango (bullish); > 1 = backwardation (bearish)
    # Map: 0.80 → +0.8, 1.00 → 0, 1.20 → -0.8
    level_signal = -(current_ratio - 1.0) * 4.0

    # Also consider the momentum of the ratio
    if len(ratios) >= 3:
        momentum = ratios[-1] - np.mean(ratios[:-1])
        momentum_signal = -momentum * 5.0  # rising ratio = bearish
    else:
        momentum_signal = 0.0

    signal = 0.7 * level_signal + 0.3 * momentum_signal
    return _clip_signal(signal)


def _skew_signal(
    skew: np.ndarray | None,
    lookback: int = 21,
) -> float:
    """CBOE SKEW index — measures demand for tail-risk protection.

    Normal range 100-150.  Higher = more tail hedging demand.
    Rising SKEW = increasing crash fear (bearish).
    Falling SKEW = complacency (bullish for equities, but watch out).

    We use z-score of SKEW relative to its own recent history, because
    the absolute level drifts over time.
    """
    if skew is None or len(skew) < lookback:
        return 0.0
    recent = skew[-lookback:]
    mu = np.mean(recent)
    sigma = np.std(recent)
    if sigma < 0.5:  # SKEW barely moving
        return 0.0
    z = (skew[-1] - mu) / sigma
    # High SKEW = bearish (traders buying puts); low SKEW = bullish
    # But don't over-weight — SKEW is noisy at daily frequency
    signal = -z * 0.3
    return _clip_signal(signal)


def _credit_spread_signal(
    hyg_close: np.ndarray | None,
    ief_close: np.ndarray | None,
    short: int = 5,
    long: int = 63,
) -> float:
    """Credit spread proxy from HYG/IEF ratio.

    HYG (high yield) outperforming IEF (treasury) = risk-on = bullish.
    HYG underperforming IEF = credit stress = bearish.

    This is one of the strongest leading indicators for equity drawdowns
    because credit markets typically crack before equity markets.
    """
    if hyg_close is None or ief_close is None:
        return 0.0
    if len(hyg_close) < long or len(ief_close) < long:
        return 0.0
    # Compute HYG/IEF ratio (credit spread proxy, inverted)
    ratio = hyg_close / np.maximum(ief_close, 1e-8)
    if len(ratio) < long:
        return 0.0

    # Z-score of current ratio vs recent history
    recent = ratio[-long:]
    mu = np.mean(recent)
    sigma = np.std(recent)
    if sigma < 1e-8:
        return 0.0

    # Level: how does current ratio compare to recent history
    z_level = (ratio[-1] - mu) / sigma

    # Momentum: is the ratio rising or falling over the short window
    short_change = np.mean(ratio[-short:]) - np.mean(ratio[-long:-short]) if long > short else 0.0
    z_mom = short_change / sigma if sigma > 1e-8 else 0.0

    # Rising ratio (HY outperforming) = bullish; falling = bearish
    signal = 0.5 * z_level * 0.4 + 0.5 * z_mom * 2.0
    return _clip_signal(signal)


def _move_index_signal(
    move: np.ndarray | None,
    lookback: int = 21,
) -> float:
    """MOVE index — bond market volatility.

    Spikes in MOVE often precede equity volatility.  Bond traders
    tend to be faster/smarter money than equity traders.

    Rising MOVE = bond vol spiking = bearish for equities.
    Falling MOVE = calming fixed income = bullish for equities.
    """
    if move is None or len(move) < lookback:
        return 0.0
    recent = move[-lookback:]
    mu = np.mean(recent)
    sigma = np.std(recent)
    if sigma < 0.5:
        return 0.0
    z = (move[-1] - mu) / sigma
    # High z = MOVE elevated = bearish; low z = MOVE compressed = bullish
    signal = -z * 0.4
    return _clip_signal(signal)


# ------------------------------------------------------------------ #
# Main class
# ------------------------------------------------------------------ #


class PredictiveSignals:
    """Compute forward-looking signals from OHLCV data.

    Each signal is independently scaled to [-1, +1] and then combined
    into a composite score using configurable weights.

    The composite score is meant to *adjust* allocation on top of
    reactive regime detection:
      - Composite > 0 → increase exposure (anticipate recovery / calm)
      - Composite < 0 → decrease exposure (anticipate crisis)
      - Composite ≈ 0 → trust the regime detector

    Args:
        signal_weights: Dict mapping signal name to its weight in
            the composite.  Weights are normalized to sum to 1.
            Defaults use equal weighting across available signals.
    """

    DEFAULT_WEIGHTS: dict[str, float] = {
        "vol_term_structure": 1.5,
        "vol_acceleration": 1.0,
        "range_expansion": 0.8,
        "price_ma_distance": 1.2,
        "trend_deceleration": 1.0,
        "rsi": 0.8,
        "price_volume_divergence": 1.0,
        "cross_asset": 1.0,
        # Market-implied signals (options / credit / bond vol)
        "vix_term_structure": 2.0,
        "skew": 1.0,
        "credit_spread": 1.8,
        "move_index": 1.2,
    }

    def __init__(
        self,
        signal_weights: dict[str, float] | None = None,
        sensitivity: float = 1.0,
    ) -> None:
        self.signal_weights = dict(signal_weights or self.DEFAULT_WEIGHTS)
        self.sensitivity = sensitivity

    def compute(
        self,
        close: np.ndarray,
        returns: np.ndarray,
        high: np.ndarray | None = None,
        low: np.ndarray | None = None,
        volume: np.ndarray | None = None,
        safe_haven_returns: np.ndarray | None = None,
        market_data: MarketSnapshot | None = None,
    ) -> SignalOutput:
        """Compute all predictive signals and combine into composite.

        Args:
            close: Daily close prices (most recent last).
            returns: Daily log returns (most recent last).
            high: Daily high prices (optional, needed for range signals).
            low: Daily low prices (optional, needed for range signals).
            volume: Daily volume (optional, needed for volume signals).
            safe_haven_returns: Returns of a safe-haven asset (e.g. TLT)
                for cross-asset signals.  Optional.
            market_data: MarketSnapshot with VIX, VIX3M, SKEW, MOVE,
                HYG, IEF data.  Optional — market-implied signals are
                skipped when not provided.

        Returns:
            SignalOutput with composite and per-component scores.
        """
        components: dict[str, float] = {}
        active_weights: dict[str, float] = {}

        # Volatility signals
        w = self.signal_weights.get("vol_term_structure", 0)
        if w > 0:
            components["vol_term_structure"] = _vol_term_structure(returns)
            active_weights["vol_term_structure"] = w

        w = self.signal_weights.get("vol_acceleration", 0)
        if w > 0:
            components["vol_acceleration"] = _vol_acceleration(returns)
            active_weights["vol_acceleration"] = w

        # Range expansion (requires high/low)
        w = self.signal_weights.get("range_expansion", 0)
        if w > 0 and high is not None and low is not None:
            components["range_expansion"] = _range_expansion(high, low, close)
            active_weights["range_expansion"] = w

        # Trend/mean-reversion signals
        w = self.signal_weights.get("price_ma_distance", 0)
        if w > 0:
            components["price_ma_distance"] = _price_distance_from_ma(close)
            active_weights["price_ma_distance"] = w

        w = self.signal_weights.get("trend_deceleration", 0)
        if w > 0:
            components["trend_deceleration"] = _trend_deceleration(returns)
            active_weights["trend_deceleration"] = w

        w = self.signal_weights.get("rsi", 0)
        if w > 0:
            components["rsi"] = _rsi_signal(returns)
            active_weights["rsi"] = w

        # Volume signals
        w = self.signal_weights.get("price_volume_divergence", 0)
        if w > 0 and volume is not None:
            components["price_volume_divergence"] = _price_volume_divergence(returns, volume)
            active_weights["price_volume_divergence"] = w

        # Cross-asset signals
        w = self.signal_weights.get("cross_asset", 0)
        if w > 0:
            components["cross_asset"] = _cross_asset_signal(returns, safe_haven_returns)
            active_weights["cross_asset"] = w

        # Market-implied signals (require MarketSnapshot)
        vix = market_data.vix if market_data is not None else None
        vix3m = market_data.vix3m if market_data is not None else None
        skew_arr = market_data.skew if market_data is not None else None
        move_arr = market_data.move if market_data is not None else None
        hyg_arr = market_data.hyg_close if market_data is not None else None
        ief_arr = market_data.ief_close if market_data is not None else None

        w = self.signal_weights.get("vix_term_structure", 0)
        if w > 0 and vix is not None and vix3m is not None:
            components["vix_term_structure"] = _vix_term_structure_signal(vix, vix3m)
            active_weights["vix_term_structure"] = w

        w = self.signal_weights.get("skew", 0)
        if w > 0 and skew_arr is not None:
            components["skew"] = _skew_signal(skew_arr)
            active_weights["skew"] = w

        w = self.signal_weights.get("credit_spread", 0)
        if w > 0 and hyg_arr is not None and ief_arr is not None:
            components["credit_spread"] = _credit_spread_signal(hyg_arr, ief_arr)
            active_weights["credit_spread"] = w

        w = self.signal_weights.get("move_index", 0)
        if w > 0 and move_arr is not None:
            components["move_index"] = _move_index_signal(move_arr)
            active_weights["move_index"] = w

        # Weighted composite
        if not active_weights:
            composite = 0.0
        else:
            total_weight = sum(active_weights.values())
            composite = sum(
                components[k] * active_weights[k] / total_weight for k in active_weights
            )
            composite *= self.sensitivity

        composite = _clip_signal(composite)

        return SignalOutput(
            composite=composite,
            components=components,
            weights=active_weights,
        )


def compute_predictive_signals(
    close: np.ndarray,
    returns: np.ndarray,
    high: np.ndarray | None = None,
    low: np.ndarray | None = None,
    volume: np.ndarray | None = None,
    safe_haven_returns: np.ndarray | None = None,
    market_data: MarketSnapshot | None = None,
    sensitivity: float = 1.0,
) -> SignalOutput:
    """Convenience function: compute predictive signals with default weights."""
    ps = PredictiveSignals(sensitivity=sensitivity)
    return ps.compute(close, returns, high, low, volume, safe_haven_returns, market_data)
