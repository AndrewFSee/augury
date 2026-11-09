"""Rough Bergomi stochastic volatility model."""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["RoughBergomi"]


class RoughBergomi:
    """Rough Bergomi model (Bayer, Friz, Gatheral 2016).

    The variance process is driven by fractional Brownian motion with
    Hurst exponent H < 0.5, capturing the roughness of empirical vol surfaces.

    V(t) = xi * exp(eta * W^H(t) - 0.5 * eta^2 * t^(2H))

    where W^H is a fractional Brownian motion with Hurst exponent H.

    Args:
        H: Hurst exponent (0 < H < 0.5 for rough volatility).
        xi: Initial forward variance (level parameter).
        eta: Vol-of-vol scaling parameter.
        rho: Correlation between asset and variance Brownian motions.
        mu: Asset drift (annualised).
    """

    def __init__(
        self,
        H: float = 0.1,
        xi: float = 0.04,
        eta: float = 1.9,
        rho: float = -0.9,
        mu: float = 0.0,
    ) -> None:
        if not 0 < H < 0.5:
            raise ValueError(f"Hurst exponent H must be in (0, 0.5), got {H}")
        self.H = H
        self.xi = xi
        self.eta = eta
        self.rho = rho
        self.mu = mu

    def simulate(
        self,
        S0: float,
        T: float,
        n_paths: int,
        n_steps: int,
        random_state: int | np.random.Generator | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Simulate Rough Bergomi price and variance paths.

        Uses Hosking's method for exact fractional Brownian motion
        generation, providing accurate roughness for small H.

        Args:
            S0: Initial asset price.
            T: Time horizon in years.
            n_paths: Number of Monte Carlo paths.
            n_steps: Number of time steps.
            random_state: Random seed or Generator.

        Returns:
            Tuple of (price_paths, var_paths) each with shape (n_paths, n_steps+1).
        """
        rng = np.random.default_rng(random_state)
        dt = T / n_steps

        # --- Exact fBM via Cholesky of covariance matrix ---
        # For moderate n_steps, Cholesky is exact and tractable.
        # cov(W^H_s, W^H_t) = 0.5 * (|s|^{2H} + |t|^{2H} - |t-s|^{2H})
        times = np.arange(1, n_steps + 1) * dt
        cov_matrix = np.empty((n_steps, n_steps))
        for i in range(n_steps):
            for j in range(n_steps):
                si, tj = times[i], times[j]
                cov_matrix[i, j] = 0.5 * (
                    si ** (2 * self.H) + tj ** (2 * self.H) - abs(si - tj) ** (2 * self.H)
                )

        # Add small regularisation for numerical stability
        cov_matrix += np.eye(n_steps) * 1e-12
        L_fbm = np.linalg.cholesky(cov_matrix)

        # Correlated Brownian increments for (W_V, W_S)
        cov_ws = np.array([[1.0, self.rho], [self.rho, 1.0]])
        L_corr = np.linalg.cholesky(cov_ws)
        sqrt_dt = np.sqrt(dt)

        Z = rng.standard_normal((n_paths, n_steps, 2))
        W = Z @ L_corr.T
        Z_V = W[:, :, 0]  # standard normal driving fBm
        dW_S = W[:, :, 1] * sqrt_dt  # Brownian increments for price

        # Generate fBm paths: multiply Cholesky factor by standard normals
        # fBm[:, k] = (L_fbm @ Z_V.T)[k] for each path
        fBm_values = Z_V @ L_fbm.T  # (n_paths, n_steps)

        fBm = np.zeros((n_paths, n_steps + 1))
        fBm[:, 1:] = fBm_values

        # Variance process: V(t) = xi * exp(eta * W^H(t) - 0.5 * eta^2 * t^(2H))
        V = np.empty((n_paths, n_steps + 1))
        for k in range(n_steps + 1):
            t = k * dt
            V[:, k] = self.xi * np.exp(
                self.eta * fBm[:, k] - 0.5 * self.eta**2 * (t ** (2 * self.H))
            )

        # Asset price: log-Euler scheme
        S = np.empty((n_paths, n_steps + 1))
        S[:, 0] = S0
        for k in range(n_steps):
            v_pos = np.maximum(V[:, k], 0.0)
            S[:, k + 1] = S[:, k] * np.exp(
                (self.mu - 0.5 * v_pos) * dt + np.sqrt(v_pos) * dW_S[:, k]
            )

        logger.debug(f"RoughBergomi simulated {n_paths} paths x {n_steps} steps, H={self.H}")
        return S, V
