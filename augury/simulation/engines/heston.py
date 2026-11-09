"""Heston stochastic volatility simulation engine (QE scheme)."""

from __future__ import annotations

import logging

import numpy as np

from augury.simulation.engines.base import SimulationEngine

logger = logging.getLogger(__name__)

__all__ = ["HestonEngine"]


class HestonEngine(SimulationEngine):
    """Heston stochastic volatility simulation engine.

    Uses the Quadratic-Exponential (QE) discretisation scheme of
    Andersen (2008) for the variance process, which avoids the
    negative-variance artefact of naive Euler discretisation.
    """

    def simulate(  # type: ignore[override]
        self,
        n_paths: int,
        n_steps: int,
        dt: float,
        S0: float = 100.0,
        v0: float = 0.04,
        mu: float = 0.05,
        kappa: float = 2.0,
        theta: float = 0.04,
        sigma_v: float = 0.3,
        rho: float = -0.7,
        random_state: int | np.random.Generator | None = None,
    ) -> np.ndarray:
        """Simulate Heston price paths via the Andersen (2008) QE scheme.

        Args:
            n_paths: Number of simulation paths.
            n_steps: Number of time steps.
            dt: Time step size in years.
            S0: Initial asset price.
            v0: Initial variance.
            mu: Asset drift (annualised).
            kappa: Mean-reversion speed of variance.
            theta: Long-run mean variance.
            sigma_v: Volatility of variance.
            rho: Correlation between asset and variance Brownians.
            random_state: Random seed or Generator.

        Returns:
            Array of shape (n_paths, n_steps+1) with simulated prices.
        """
        rng = np.random.default_rng(random_state)

        S = np.empty((n_paths, n_steps + 1))
        V = np.empty((n_paths, n_steps + 1))
        S[:, 0] = S0
        V[:, 0] = v0

        # QE constants
        exp_kappa = np.exp(-kappa * dt)
        c2 = sigma_v**2 * exp_kappa * (1 - exp_kappa) / kappa
        # Var[V(t+dt) | V(t)] = V(t) * c2 + c1  (Andersen 2008, eq. 17)
        c1 = theta * sigma_v**2 * (1 - exp_kappa) ** 2 / (2 * kappa)

        # Log-price discretisation constants (Andersen 2008, eq. 33) with
        # central weighting gamma1 = gamma2 = 0.5. The price/variance
        # correlation enters through K0..K2, which tie the log-price step
        # to the *realised* variance move V(t) -> V(t+dt), so rho is honoured
        # without needing a second correlated normal.
        g1 = g2 = 0.5
        if sigma_v > 1e-8:
            k0 = -rho * kappa * theta / sigma_v * dt
            k1 = g1 * dt * (kappa * rho / sigma_v - 0.5) - rho / sigma_v
            k2 = g2 * dt * (kappa * rho / sigma_v - 0.5) + rho / sigma_v
        else:  # deterministic variance: nothing to correlate with
            k0, k1, k2 = 0.0, -0.5 * g1 * dt, -0.5 * g2 * dt
        k3 = g1 * dt * (1.0 - rho**2) if sigma_v > 1e-8 else g1 * dt
        k4 = g2 * dt * (1.0 - rho**2) if sigma_v > 1e-8 else g2 * dt

        for t in range(n_steps):
            # Conditional mean and variance of V(t+dt) | V(t)
            m = theta + (V[:, t] - theta) * exp_kappa
            s2 = V[:, t] * c2 + c1
            m2 = m**2
            psi = np.divide(s2, m2, out=np.zeros_like(s2), where=m2 > 1e-32)

            # QE sampling of V(t+dt)
            V_next = np.empty(n_paths)
            # (Near-)deterministic variance, e.g. sigma_v = 0: take the mean
            mask_det = psi < 1e-12
            V_next[mask_det] = m[mask_det]
            # Low dispersion regime: moment-matched normal
            mask_low = (psi <= 1.5) & ~mask_det
            if np.any(mask_low):
                b2 = (
                    2 / psi[mask_low]
                    - 1
                    + np.sqrt(2 / psi[mask_low]) * np.sqrt(2 / psi[mask_low] - 1)
                )
                a = m[mask_low] / (1 + b2)
                Z_v = rng.standard_normal(np.sum(mask_low))
                V_next[mask_low] = a * (np.sqrt(b2) + Z_v) ** 2

            # High dispersion regime: exponential
            mask_high = ~(mask_low | mask_det)
            if np.any(mask_high):
                p = (psi[mask_high] - 1) / (psi[mask_high] + 1)
                beta = 2 / (m[mask_high] * (1 + psi[mask_high]))
                U = rng.random(np.sum(mask_high))
                V_next[mask_high] = np.where(
                    p >= U,
                    0.0,
                    np.log((1 - p) / (1 - U)) / beta,
                )

            V[:, t + 1] = np.maximum(V_next, 0.0)

            # Asset price step (Andersen QE log-price scheme)
            v_t, v_n = V[:, t], V[:, t + 1]
            Z_s = rng.standard_normal(n_paths)
            log_step = mu * dt + k0 + k1 * v_t + k2 * v_n + np.sqrt(k3 * v_t + k4 * v_n) * Z_s
            S[:, t + 1] = S[:, t] * np.exp(log_step)

        logger.debug(f"HestonEngine QE simulated {n_paths} paths x {n_steps} steps")
        return S
