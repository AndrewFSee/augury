"""Feature engineering for multivariate HMM regime detection.

Provides a rich library of candidate features that can be computed from
a 1-D return series.  Each feature captures a different facet of market
dynamics — direction, volatility, asymmetry, tail risk, mean-reversion,
etc.  Features are z-scored so the HMM's Gaussian emission model treats
them on an equal footing.

The ``FEATURE_REGISTRY`` maps short names to builder functions.  Use
:func:`build_hmm_features` with a list of feature names to construct
any subset for experimentation.

Default features (the original 3):

- **returns** — raw log returns
- **signed_vol** — ``return × rolling_std`` (crash/recovery interaction)
- **momentum** — trailing cumulative return
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.signal import lfilter

__all__ = [
    "build_hmm_features",
    "FEATURE_REGISTRY",
    "ALL_FEATURE_NAMES",
    "DEFAULT_FEATURE_NAMES",
]

# ------------------------------------------------------------------ #
# Rolling helpers (pure numpy, no pandas dependency)
# ------------------------------------------------------------------ #

def _rolling_sum(r: np.ndarray, window: int) -> np.ndarray:
    """Rolling sum with expanding warmup. Vectorized via cumsum."""
    n = len(r)
    cs = np.cumsum(r)
    out = np.empty(n, dtype=float)
    # Expanding warmup: first `window` elements use all available data
    out[:window] = cs[:window]
    # Fixed window
    if n > window:
        out[window:] = cs[window:] - cs[:-window]
    return out


def _rolling_mean(r: np.ndarray, window: int) -> np.ndarray:
    """Rolling mean with expanding warmup. Vectorized via cumsum."""
    n = len(r)
    cs = np.cumsum(r)
    out = np.empty(n, dtype=float)
    # Expanding warmup
    counts = np.arange(1, min(window, n) + 1, dtype=float)
    out[:window] = cs[:window] / counts
    # Fixed window
    if n > window:
        out[window:] = (cs[window:] - cs[:-window]) / window
    return out


def _rolling_std(r: np.ndarray, window: int) -> np.ndarray:
    """Rolling standard deviation with expanding warmup. Vectorized."""
    n = len(r)
    out = np.empty(n, dtype=float)
    # Use cumsum-based E[X^2] - E[X]^2
    cs = np.cumsum(r)
    cs2 = np.cumsum(r ** 2)

    # Expanding warmup period
    for t in range(min(window, n)):
        cnt = t + 1
        if cnt > 1:
            mean = cs[t] / cnt
            var = cs2[t] / cnt - mean ** 2
            out[t] = np.sqrt(max(var, 0.0))
        else:
            out[t] = abs(r[t])

    # Fixed window: vectorized
    if n > window:
        s1 = cs[window:] - cs[:-window]
        s2 = cs2[window:] - cs2[:-window]
        mean = s1 / window
        var = s2 / window - mean ** 2
        out[window:] = np.sqrt(np.maximum(var, 0.0))

    return out


def _ewma(r: np.ndarray, span: int) -> np.ndarray:
    """Exponentially weighted moving average. Vectorized via scipy lfilter."""
    alpha = 2.0 / (span + 1)
    # IIR filter: y[t] = alpha * x[t] + (1-alpha) * y[t-1]
    return lfilter([alpha], [1, -(1 - alpha)], r, zi=[r[0] * (1 - alpha)])[0]


# ------------------------------------------------------------------ #
# Individual feature builders
# ------------------------------------------------------------------ #
# Each takes (r, params) where r is 1-D returns and params is the
# shared parameter dict.  Returns a 1-D array of length len(r).
# ------------------------------------------------------------------ #

FeatureBuilder = Callable[[np.ndarray, dict], np.ndarray]


def _feat_returns(r: np.ndarray, params: dict) -> np.ndarray:
    """Raw daily log returns."""
    return r.copy()


def _feat_abs_returns(r: np.ndarray, params: dict) -> np.ndarray:
    """Absolute returns — magnitude regardless of direction."""
    return np.abs(r)


def _feat_signed_vol(r: np.ndarray, params: dict) -> np.ndarray:
    """return × rolling vol.  Negative in crashes, positive in rallies."""
    rvol = _rolling_std(r, params.get("vol_lookback", 21))
    return r * rvol


def _feat_realized_vol(r: np.ndarray, params: dict) -> np.ndarray:
    """Rolling realised volatility (unsigned)."""
    return _rolling_std(r, params.get("vol_lookback", 21))


def _feat_momentum(r: np.ndarray, params: dict) -> np.ndarray:
    """Short-term trailing cumulative return."""
    return _rolling_sum(r, params.get("momentum_lookback", 5))


def _feat_momentum_long(r: np.ndarray, params: dict) -> np.ndarray:
    """Medium-term trailing cumulative return (21 day)."""
    return _rolling_sum(r, params.get("momentum_long_lookback", 21))


def _feat_vol_of_vol(r: np.ndarray, params: dict) -> np.ndarray:
    """Volatility of volatility — rolling std of daily vol changes."""
    rvol = _rolling_std(r, params.get("vol_lookback", 21))
    vol_changes = np.diff(rvol, prepend=rvol[0])
    return _rolling_std(vol_changes, params.get("vov_lookback", 21))


def _feat_realized_skew(r: np.ndarray, params: dict) -> np.ndarray:
    """Rolling realised skewness (asymmetry of returns)."""
    window = params.get("skew_lookback", 21)
    n = len(r)
    out = np.zeros(n, dtype=float)
    if n < 3:
        return out
    # Use sliding_window_view for fixed-window portion
    if n >= window:
        views = np.lib.stride_tricks.sliding_window_view(r, window)
        mu = np.mean(views, axis=1)
        sigma = np.std(views, axis=1)
        valid = sigma > 1e-12
        if np.any(valid):
            centered = views[valid] - mu[valid, None]
            out[window - 1 :][valid] = np.mean(
                (centered / sigma[valid, None]) ** 3, axis=1
            )
    # Expanding warmup for early elements
    for t in range(min(window - 1, n)):
        w = r[: t + 1]
        if len(w) < 3:
            continue
        mu_w = np.mean(w)
        sigma_w = np.std(w)
        if sigma_w > 1e-12:
            out[t] = np.mean(((w - mu_w) / sigma_w) ** 3)
    return out


def _feat_realized_kurt(r: np.ndarray, params: dict) -> np.ndarray:
    """Rolling realised excess kurtosis (tail heaviness)."""
    window = params.get("kurt_lookback", 21)
    n = len(r)
    out = np.zeros(n, dtype=float)
    if n < 4:
        return out
    if n >= window:
        views = np.lib.stride_tricks.sliding_window_view(r, window)
        mu = np.mean(views, axis=1)
        sigma = np.std(views, axis=1)
        valid = sigma > 1e-12
        if np.any(valid):
            centered = views[valid] - mu[valid, None]
            out[window - 1 :][valid] = (
                np.mean((centered / sigma[valid, None]) ** 4, axis=1) - 3.0
            )
    for t in range(min(window - 1, n)):
        w = r[: t + 1]
        if len(w) < 4:
            continue
        mu_w = np.mean(w)
        sigma_w = np.std(w)
        if sigma_w > 1e-12:
            out[t] = np.mean(((w - mu_w) / sigma_w) ** 4) - 3.0
    return out


def _feat_downside_vol(r: np.ndarray, params: dict) -> np.ndarray:
    """Rolling downside deviation (std of negative returns only)."""
    window = params.get("vol_lookback", 21)
    n = len(r)
    out = np.zeros(n, dtype=float)
    for t in range(n):
        w = r[max(0, t - window + 1) : t + 1]
        neg = w[w < 0]
        out[t] = np.std(neg) if len(neg) > 1 else 0.0
    return out


def _feat_up_down_vol_ratio(r: np.ndarray, params: dict) -> np.ndarray:
    """Ratio of upside vol to downside vol — >1 means more up dispersion."""
    window = params.get("vol_lookback", 21)
    n = len(r)
    out = np.ones(n, dtype=float)
    for t in range(n):
        w = r[max(0, t - window + 1) : t + 1]
        up = w[w > 0]
        dn = w[w < 0]
        up_vol = np.std(up) if len(up) > 1 else 1e-8
        dn_vol = np.std(dn) if len(dn) > 1 else 1e-8
        out[t] = up_vol / dn_vol
    return out


def _feat_return_autocorr(r: np.ndarray, params: dict) -> np.ndarray:
    """Rolling first-order autocorrelation of returns."""
    window = params.get("autocorr_lookback", 21)
    n = len(r)
    out = np.zeros(n, dtype=float)
    for t in range(n):
        w = r[max(0, t - window + 1) : t + 1]
        if len(w) < 5:
            continue
        mu = np.mean(w)
        var = np.var(w)
        if var > 1e-16:
            out[t] = np.mean((w[1:] - mu) * (w[:-1] - mu)) / var
    return out


def _feat_ewma_vol(r: np.ndarray, params: dict) -> np.ndarray:
    """EWMA volatility (exponentially weighted std)."""
    span = params.get("ewma_span", 21)
    sq = r ** 2
    ewma_var = _ewma(sq, span)
    return np.sqrt(np.maximum(ewma_var, 0.0))


def _feat_vol_regime_change(r: np.ndarray, params: dict) -> np.ndarray:
    """Short vol minus long vol — positive when vol is spiking."""
    short_w = params.get("vol_short", 5)
    long_w = params.get("vol_lookback", 21)
    return _rolling_std(r, short_w) - _rolling_std(r, long_w)


def _feat_max_drawdown(r: np.ndarray, params: dict) -> np.ndarray:
    """Rolling max drawdown over a trailing window."""
    window = params.get("mdd_lookback", 21)
    cum = np.cumsum(r)
    n = len(r)
    out = np.zeros(n, dtype=float)
    # Max drawdown requires sequential accumulate per window — keep loop
    # but avoid redundant slicing by reusing cumulative sums
    for t in range(n):
        w_start = max(0, t - window + 1)
        segment = cum[w_start : t + 1] - cum[w_start]
        peak = np.maximum.accumulate(segment)
        out[t] = np.min(segment - peak)
    return out


# ------------------------------------------------------------------ #
# Feature registry
# ------------------------------------------------------------------ #

FEATURE_REGISTRY: dict[str, FeatureBuilder] = {
    "returns": _feat_returns,
    "abs_returns": _feat_abs_returns,
    "signed_vol": _feat_signed_vol,
    "realized_vol": _feat_realized_vol,
    "momentum": _feat_momentum,
    "momentum_long": _feat_momentum_long,
    "vol_of_vol": _feat_vol_of_vol,
    "realized_skew": _feat_realized_skew,
    "realized_kurt": _feat_realized_kurt,
    "downside_vol": _feat_downside_vol,
    "up_down_vol_ratio": _feat_up_down_vol_ratio,
    "return_autocorr": _feat_return_autocorr,
    "ewma_vol": _feat_ewma_vol,
    "vol_regime_change": _feat_vol_regime_change,
    "max_drawdown": _feat_max_drawdown,
}

ALL_FEATURE_NAMES: list[str] = list(FEATURE_REGISTRY.keys())

DEFAULT_FEATURE_NAMES: list[str] = ["returns", "signed_vol", "momentum"]


# ------------------------------------------------------------------ #
# Public builder
# ------------------------------------------------------------------ #

def build_hmm_features(
    returns: np.ndarray,
    *,
    feature_names: list[str] | None = None,
    vol_lookback: int = 21,
    momentum_lookback: int = 5,
    **kwargs: int,
) -> np.ndarray:
    """Build a multivariate feature matrix from a 1-D return series.

    Args:
        returns: 1-D array of daily log returns (length *T*).
        feature_names: Which features to include.  Defaults to
            ``DEFAULT_FEATURE_NAMES`` (``['returns', 'signed_vol',
            'momentum']``).  Pass ``None`` for defaults or any subset
            of :data:`ALL_FEATURE_NAMES`.
        vol_lookback: Window for realised vol rolling std.
        momentum_lookback: Window for trailing cumulative return.
        **kwargs: Additional parameter overrides forwarded to individual
            feature builders (e.g. ``skew_lookback=42``).

    Returns:
        Array of shape ``(T, len(feature_names))`` with z-scored features.
    """
    if feature_names is None:
        feature_names = DEFAULT_FEATURE_NAMES

    r = np.asarray(returns, dtype=float).ravel()

    params: dict = {
        "vol_lookback": vol_lookback,
        "momentum_lookback": momentum_lookback,
        **kwargs,
    }

    cols = []
    for name in feature_names:
        builder = FEATURE_REGISTRY.get(name)
        if builder is None:
            raise ValueError(
                f"Unknown feature '{name}'. "
                f"Available: {sorted(FEATURE_REGISTRY.keys())}"
            )
        cols.append(builder(r, params))

    features = np.column_stack(cols)

    # Z-score each column
    for col in range(features.shape[1]):
        mu = np.mean(features[:, col])
        sigma = np.std(features[:, col])
        if sigma > 1e-12:
            features[:, col] = (features[:, col] - mu) / sigma
        else:
            features[:, col] = 0.0

    return features
