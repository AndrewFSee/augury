"""Heston stochastic volatility model."""

from __future__ import annotations

import numpy as np

__all__ = ["HestonModel"]


class HestonModel:
    """Heston (1993) stochastic volatility model: parameters plus simulation.

    dS = mu*S dt + sqrt(V)*S dW_S
    dV = kappa*(theta - V) dt + sigma_v*sqrt(V) dW_V
    Corr(dW_S, dW_V) = rho

    Args:
        v0: Initial variance.
        kappa: Mean-reversion speed of variance.
        theta: Long-run mean variance.
        sigma_v: Volatility of variance (vol-of-vol).
        rho: Correlation between asset and variance Brownian motions.
        mu: Asset drift (annualised).
        r: Risk-free rate (for option pricing).
        q: Dividend yield.
    """

    def __init__(
        self,
        v0: float = 0.04,
        kappa: float = 2.0,
        theta: float = 0.04,
        sigma_v: float = 0.3,
        rho: float = -0.7,
        mu: float = 0.05,
        r: float = 0.05,
        q: float = 0.0,
    ) -> None:
        self.v0 = v0
        self.kappa = kappa
        self.theta = theta
        self.sigma_v = sigma_v
        self.rho = rho
        self.mu = mu
        self.r = r
        self.q = q

    def simulate(
        self,
        S0: float,
        T: float,
        n_paths: int,
        n_steps: int,
        random_state: int | np.random.Generator | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Simulate Heston price and variance paths.

        Delegates to :class:`~augury.simulation.engines.heston.HestonEngine`
        (Andersen 2008 QE scheme), so there is one Heston simulator in the
        package. QE keeps variance non-negative without the bias that
        truncated Euler adds at large ``sigma_v`` or coarse steps.

        Args:
            S0: Initial asset price.
            T: Time horizon in years.
            n_paths: Number of Monte Carlo paths.
            n_steps: Number of time steps.
            random_state: Random seed or Generator.

        Returns:
            Tuple of (price_paths, var_paths) each with shape (n_paths, n_steps+1).
        """
        from augury.simulation.engines.heston import HestonEngine

        return HestonEngine().simulate_with_variance(
            n_paths=n_paths,
            n_steps=n_steps,
            dt=T / n_steps,
            S0=S0,
            v0=self.v0,
            mu=self.mu,
            kappa=self.kappa,
            theta=self.theta,
            sigma_v=self.sigma_v,
            rho=self.rho,
            random_state=random_state,
        )
