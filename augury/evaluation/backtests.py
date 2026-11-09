"""Statistical backtests for VaR forecasts and forecast comparison."""

from __future__ import annotations

from typing import Literal

import numpy as np
from scipy import stats
from scipy.special import xlogy

__all__ = [
    "kupiec_test",
    "christoffersen_test",
    "conditional_coverage_test",
    "basel_zone",
    "diebold_mariano",
]


def _bernoulli_loglik(n0: int, n1: int, p: float) -> float:
    """Log-likelihood of n0 zeros and n1 ones under Bernoulli(p), with 0*log(0) = 0."""
    return float(xlogy(n0, 1.0 - p) + xlogy(n1, p))


def kupiec_test(hits: np.ndarray, alpha: float) -> tuple[float, float]:
    """Kupiec (1995) proportion-of-failures test of unconditional coverage.

    H0: the VaR violation probability equals ``alpha``.

    Args:
        hits: Boolean/0-1 array, True where the loss exceeded VaR.
        alpha: Nominal violation probability (e.g. 0.01 for 99% VaR).

    Returns:
        Tuple ``(lr_statistic, p_value)``; LR ~ chi2(1) under H0.
    """
    v = np.asarray(hits, dtype=bool)
    n1 = int(v.sum())
    n0 = int(v.size - n1)
    if v.size == 0:
        return 0.0, 1.0
    pi_hat = n1 / v.size
    lr = -2.0 * (_bernoulli_loglik(n0, n1, alpha) - _bernoulli_loglik(n0, n1, pi_hat))
    lr = max(lr, 0.0)
    return lr, float(stats.chi2.sf(lr, 1))


def christoffersen_test(hits: np.ndarray) -> tuple[float, float]:
    """Christoffersen (1998) test of independence of VaR violations.

    H0: a violation today is no more likely after a violation yesterday
    (first-order Markov alternative). Detects clustered violations, the
    signature of a model that reacts too slowly to volatility changes.

    Args:
        hits: Boolean/0-1 array of violations in time order.

    Returns:
        Tuple ``(lr_statistic, p_value)``; LR ~ chi2(1) under H0.
    """
    v = np.asarray(hits, dtype=int)
    if v.size < 2:
        return 0.0, 1.0
    prev, cur = v[:-1], v[1:]
    n00 = int(np.sum((prev == 0) & (cur == 0)))
    n01 = int(np.sum((prev == 0) & (cur == 1)))
    n10 = int(np.sum((prev == 1) & (cur == 0)))
    n11 = int(np.sum((prev == 1) & (cur == 1)))
    total = n00 + n01 + n10 + n11
    pi = (n01 + n11) / total
    pi01 = n01 / (n00 + n01) if (n00 + n01) else 0.0
    pi11 = n11 / (n10 + n11) if (n10 + n11) else 0.0
    ll_null = _bernoulli_loglik(n00 + n10, n01 + n11, pi)
    ll_alt = _bernoulli_loglik(n00, n01, pi01) + _bernoulli_loglik(n10, n11, pi11)
    lr = max(-2.0 * (ll_null - ll_alt), 0.0)
    return lr, float(stats.chi2.sf(lr, 1))


def conditional_coverage_test(hits: np.ndarray, alpha: float) -> tuple[float, float]:
    """Christoffersen (1998) joint test of correct coverage and independence.

    Returns:
        Tuple ``(lr_statistic, p_value)``; LR = LR_uc + LR_ind ~ chi2(2) under H0.
    """
    lr_uc, _ = kupiec_test(hits, alpha)
    lr_ind, _ = christoffersen_test(hits)
    lr = lr_uc + lr_ind
    return lr, float(stats.chi2.sf(lr, 2))


def basel_zone(n_violations: int, n_obs: int, alpha: float = 0.01) -> str:
    """Basel traffic-light zone from the binomial distribution of violations.

    Green while ``P(X <= k) < 95%``, red once ``P(X <= k) >= 99.99%``, yellow
    in between. For 250 days at 1% this reproduces the regulatory table:
    0-4 green, 5-9 yellow, 10+ red.

    Args:
        n_violations: Observed number of VaR violations.
        n_obs: Number of backtest days.
        alpha: Nominal violation probability.

    Returns:
        ``"Green"``, ``"Yellow"`` or ``"Red"``.
    """
    cdf = float(stats.binom.cdf(n_violations, n_obs, alpha))
    if cdf < 0.95:
        return "Green"
    if cdf < 0.9999:
        return "Yellow"
    return "Red"


def diebold_mariano(
    loss_a: np.ndarray,
    loss_b: np.ndarray,
    lags: int = 0,
    alternative: Literal["two-sided", "less", "greater"] = "two-sided",
) -> tuple[float, float]:
    """Diebold-Mariano test of equal predictive accuracy.

    Tests ``E[loss_a - loss_b] = 0`` using a Newey-West (Bartlett) long-run
    variance, the Harvey-Leybourne-Newbold small-sample correction and a
    Student-t reference distribution.

    Args:
        loss_a: Per-period losses (e.g. CRPS) of model A.
        loss_b: Per-period losses of model B, aligned with ``loss_a``.
        lags: Autocorrelation lags in the loss differential. Use ``h - 1``
            for overlapping h-step forecasts made every period.
        alternative: ``"less"`` tests whether A has *lower* loss than B.

    Returns:
        Tuple ``(statistic, p_value)``. A negative statistic favours model A.
    """
    d = np.asarray(loss_a, dtype=float) - np.asarray(loss_b, dtype=float)
    d = d[np.isfinite(d)]
    n = d.size
    if n < 3:
        return float("nan"), float("nan")
    dc = d - d.mean()
    lrv = float(dc @ dc) / n
    for k in range(1, min(lags, n - 1) + 1):
        w = 1.0 - k / (lags + 1)
        lrv += 2.0 * w * float(dc[k:] @ dc[:-k]) / n
    if lrv <= 0:
        return float("nan"), float("nan")
    dm = d.mean() / np.sqrt(lrv / n)
    # Harvey, Leybourne & Newbold (1997) correction for horizon h = lags + 1
    h = lags + 1
    dm *= np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    if alternative == "less":
        p = stats.t.cdf(dm, n - 1)
    elif alternative == "greater":
        p = stats.t.sf(dm, n - 1)
    else:
        p = 2 * stats.t.sf(abs(dm), n - 1)
    return float(dm), float(p)
