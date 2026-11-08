"""Drift conventions: convert mean log returns into engine drift inputs.

Every simulation engine takes an *arithmetic* drift ``mu`` and applies the
Itô convexity correction itself, i.e. the expected log return per unit of
time is ``mu - 0.5 * E[variance]`` (minus a jump compensator for Merton).

Regime detectors and return statistics, however, estimate the mean of
*log* returns directly. Passing that number straight into an engine
subtracts the convexity term a second time and biases forecasts
downward by ``0.5 * sigma^2`` per year (about -5%/yr at 32% vol).

The helpers here map an annualised mean log return to the ``mu`` each
engine needs so that the simulated mean log return matches the estimate.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "gbm_drift",
    "heston_drift",
    "merton_drift",
    "rough_bergomi_drift",
]


def gbm_drift(mu_log: float, sigma: float) -> float:
    """Arithmetic drift for GBM / regime-switching GBM.

    Args:
        mu_log: Annualised mean log return.
        sigma: Annualised volatility.
    """
    return float(mu_log + 0.5 * sigma**2)


def heston_drift(
    mu_log: float, v0: float, theta: float, kappa: float, horizon_years: float
) -> float:
    """Arithmetic drift for Heston, using the horizon-average expected variance.

    ``E[v_t] = theta + (v0 - theta) * exp(-kappa t)``, averaged over
    ``[0, horizon_years]``.

    Args:
        mu_log: Annualised mean log return.
        v0: Initial annualised variance.
        theta: Long-run annualised variance.
        kappa: Mean-reversion speed.
        horizon_years: Forecast horizon in years.
    """
    kt = kappa * horizon_years
    decay = (1.0 - np.exp(-kt)) / kt if kt > 1e-10 else 1.0
    avg_var = theta + (v0 - theta) * decay
    return float(mu_log + 0.5 * avg_var)


def merton_drift(
    mu_log: float, sigma: float, lambda_j: float, mu_j: float, sigma_j: float
) -> float:
    """Arithmetic drift for Merton jump-diffusion.

    The engine's expected log return per year is
    ``mu - lambda_j * k - 0.5 * sigma^2 + lambda_j * mu_j`` with
    ``k = exp(mu_j + sigma_j^2 / 2) - 1``; this inverts that relation.

    Args:
        mu_log: Annualised mean log return (diffusion + jumps).
        sigma: Annualised diffusion volatility.
        lambda_j: Jump intensity per year.
        mu_j: Mean log jump size.
        sigma_j: Std dev of log jump size.
    """
    k = np.exp(mu_j + 0.5 * sigma_j**2) - 1.0
    return float(mu_log + 0.5 * sigma**2 + lambda_j * (k - mu_j))


def rough_bergomi_drift(mu_log: float, xi: float) -> float:
    """Arithmetic drift for rough Bergomi (flat forward variance ``xi``).

    Args:
        mu_log: Annualised mean log return.
        xi: Annualised forward variance.
    """
    return float(mu_log + 0.5 * xi)
