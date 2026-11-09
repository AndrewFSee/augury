"""Multi-frequency realized volatility measures.

Provides a suite of realized volatility estimators at various frequencies,
robust to microstructure noise and jumps. These serve as inputs for the
HAR-RV model and as conditioning variables for regime detection.

Estimators:
    - Realized Variance (standard sum of squared returns)
    - Bipower Variation (Barndorff-Nielsen & Shephard, robust to jumps)
    - Realized Semi-Variance (upside/downside decomposition)
    - Jump Variation (truncated power variation)
    - Realized Kernel (Barndorff-Nielsen et al., noise-robust)
    - Parkinson Range Estimator
    - Garman-Klass Estimator (uses OHLC data)

References:
    Barndorff-Nielsen, O. E. & Shephard, N. (2004).
    "Power and Bipower Variation with Stochastic Volatility and Jumps."
    Journal of Financial Econometrics, 2(1), 1-37.
"""

from __future__ import annotations

import logging

import numpy as np

__all__ = ["RealizedMeasures"]

logger = logging.getLogger(__name__)


class RealizedMeasures:
    """Compute realized volatility measures at multiple frequencies.

    All methods are static and operate on return or price arrays.
    """

    @staticmethod
    def realized_variance(
        returns: np.ndarray,
        frequency_minutes: int = 5,
        sampling_minutes: int = 1,
    ) -> float:
        """Realized variance from intraday returns.

        Args:
            returns: Array of intraday log returns at base frequency.
            frequency_minutes: Target sampling frequency in minutes.
            sampling_minutes: Base frequency of input returns in minutes.

        Returns:
            Realized variance (sum of squared returns at target frequency).
        """
        r = np.asarray(returns, dtype=float)
        step = max(1, frequency_minutes // sampling_minutes)
        if step > 1:
            n_blocks = len(r) // step
            r = np.sum(r[: n_blocks * step].reshape(n_blocks, step), axis=1)
        return float(np.sum(r**2))

    @staticmethod
    def bipower_variation(returns: np.ndarray) -> float:
        """Bipower variation — robust to jumps.

        BPV = (π/2) × Σ|r_i| × |r_{i-1}|

        Args:
            returns: Array of log returns.

        Returns:
            Bipower variation estimate.
        """
        r = np.asarray(returns, dtype=float)
        if len(r) < 2:
            return float(np.sum(r**2))
        abs_r = np.abs(r)
        return float((np.pi / 2) * np.sum(abs_r[1:] * abs_r[:-1]))

    @staticmethod
    def realized_semivariance(
        returns: np.ndarray,
    ) -> tuple[float, float]:
        """Realized semi-variance (upside/downside decomposition).

        Args:
            returns: Array of log returns.

        Returns:
            Tuple of (downside_variance, upside_variance).
        """
        r = np.asarray(returns, dtype=float)
        neg = r[r < 0]
        pos = r[r >= 0]
        rv_down = float(np.sum(neg**2)) if len(neg) > 0 else 0.0
        rv_up = float(np.sum(pos**2)) if len(pos) > 0 else 0.0
        return rv_down, rv_up

    @staticmethod
    def jump_variation(
        returns: np.ndarray,
        threshold_multiplier: float = 3.0,
    ) -> float:
        """Jump variation via truncated realized variance.

        Returns the portion of realized variance attributable to jumps,
        estimated as RV - BPV (truncated at zero).

        Args:
            returns: Array of log returns.
            threshold_multiplier: Not used in BPV-based method; kept for API
                compatibility with threshold-based alternatives.

        Returns:
            Jump variation estimate (non-negative).
        """
        rv = float(np.sum(np.asarray(returns, dtype=float) ** 2))
        bpv = RealizedMeasures.bipower_variation(returns)
        return max(rv - bpv, 0.0)

    @staticmethod
    def realized_kernel(
        returns: np.ndarray,
        kernel: str = "parzen",
        bandwidth: int | None = None,
    ) -> float:
        """Realized kernel estimator (noise-robust).

        Computes a kernel-weighted autocovariance estimator that is robust
        to market microstructure noise.

        Args:
            returns: Array of log returns.
            kernel: Kernel function — 'parzen' or 'bartlett'.
            bandwidth: Kernel bandwidth. Default: n^(2/3).

        Returns:
            Realized kernel variance estimate.
        """
        r = np.asarray(returns, dtype=float)
        n = len(r)

        if bandwidth is None:
            bandwidth = max(1, int(n ** (2 / 3)))

        # Autocovariance at lag h
        gamma_0 = float(np.sum(r**2))
        rk = gamma_0

        for h in range(1, bandwidth + 1):
            gamma_h = float(np.sum(r[h:] * r[:-h]))
            x = h / (bandwidth + 1)

            if kernel == "parzen":
                k = 1 - 6 * x**2 + 6 * x**3 if x <= 0.5 else 2 * (1 - x) ** 3
            elif kernel == "bartlett":
                k = 1 - x
            else:
                raise ValueError(f"Unknown kernel: {kernel}")

            rk += 2 * k * gamma_h

        return max(rk, 0.0)

    @staticmethod
    def parkinson_variance(high: np.ndarray, low: np.ndarray) -> float:
        """Parkinson range-based variance estimator.

        Args:
            high: Array of log(high) prices.
            low: Array of log(low) prices.

        Returns:
            Parkinson variance estimate.
        """
        hi = np.asarray(high, dtype=float)
        lo = np.asarray(low, dtype=float)
        return float(np.mean((hi - lo) ** 2) / (4 * np.log(2)))

    @staticmethod
    def garman_klass_variance(
        open_: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
    ) -> float:
        """Garman-Klass OHLC variance estimator.

        Args:
            open_: Array of log(open) prices.
            high: Array of log(high) prices.
            low: Array of log(low) prices.
            close: Array of log(close) prices.

        Returns:
            Garman-Klass variance estimate.
        """
        op = np.asarray(open_, dtype=float)
        hi = np.asarray(high, dtype=float)
        lo = np.asarray(low, dtype=float)
        cl = np.asarray(close, dtype=float)

        return float(np.mean(0.5 * (hi - lo) ** 2 - (2 * np.log(2) - 1) * (cl - op) ** 2))

    @staticmethod
    def multi_frequency_rv(
        returns: np.ndarray,
        frequencies_minutes: list[int] | None = None,
        base_frequency_minutes: int = 1,
    ) -> dict[str, float]:
        """Compute realized variance at multiple sampling frequencies.

        Args:
            returns: Array of intraday log returns at base frequency.
            frequencies_minutes: List of target frequencies.
                Default: [5, 15, 60] (5-min, 15-min, 1-hour).
            base_frequency_minutes: Frequency of input data in minutes.

        Returns:
            Dict mapping frequency labels to RV values.
        """
        if frequencies_minutes is None:
            frequencies_minutes = [5, 15, 60]

        result = {}
        for freq in frequencies_minutes:
            label = f"rv_{freq}min"
            result[label] = RealizedMeasures.realized_variance(
                returns, frequency_minutes=freq, sampling_minutes=base_frequency_minutes
            )
        return result

    @staticmethod
    def rolling_realized_measures(
        returns: np.ndarray,
        window: int = 78,
    ) -> dict[str, np.ndarray]:
        """Compute rolling realized measures over a window.

        Useful for building time series of daily RV, BPV, jump variation
        from intraday data (e.g., window=78 for 5-min bars in a 6.5-hour day).

        Args:
            returns: Array of intraday log returns.
            window: Number of intraday observations per period.

        Returns:
            Dict with 'rv', 'bpv', 'jump', 'semivar_down', 'semivar_up' arrays.
        """
        r = np.asarray(returns, dtype=float)
        n = len(r)
        n_periods = n // window

        rv = np.empty(n_periods)
        bpv = np.empty(n_periods)
        jump = np.empty(n_periods)
        sv_down = np.empty(n_periods)
        sv_up = np.empty(n_periods)

        for i in range(n_periods):
            block = r[i * window : (i + 1) * window]
            rv[i] = float(np.sum(block**2))
            bpv[i] = RealizedMeasures.bipower_variation(block)
            jump[i] = max(rv[i] - bpv[i], 0.0)
            d, u = RealizedMeasures.realized_semivariance(block)
            sv_down[i] = d
            sv_up[i] = u

        return {
            "rv": rv,
            "bpv": bpv,
            "jump": jump,
            "semivar_down": sv_down,
            "semivar_up": sv_up,
        }
