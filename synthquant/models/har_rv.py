"""Heterogeneous Autoregressive Realized Volatility (HAR-RV) model.

Implements the HAR-RV model of Corsi (2009) and extensions:

- **HAR-RV**: Base model with daily, weekly, monthly realized variance components.
- **HAR-RV-J**: Adds a jump component via Barndorff-Nielsen-Shephard bipower variation.
- **HAR-RV-CJ**: Separates continuous and jump variation.

The multi-scale structure captures heterogeneous information processing
across different trader horizons (intraday, daily, weekly, monthly)
and maps naturally to Synth's multi-scale CRPS scoring at 5-min,
30-min, 3-hour, and 24-hour intervals.

References:
    Corsi, F. (2009). "A Simple Approximate Long-Memory Model of Realized Volatility."
    Journal of Financial Econometrics, 7(2), 174-196.

    Andersen, T.G., Bollerslev, T., & Diebold, F.X. (2007).
    "Roughing It Up: Including Jump Components in the Measurement,
    Modeling, and Forecasting of Return Volatility."
    Review of Economics and Statistics, 89(4), 701-720.
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np

__all__ = ["HARRV"]

logger = logging.getLogger(__name__)


class HARRV:
    """Heterogeneous Autoregressive Realized Volatility model.

    Forecasts realized variance at multiple horizons by regressing on
    daily, weekly, and monthly realized variance components.

    Model: ``RV_{t+h} = c + β_d·RV_t^(d) + β_w·RV_t^(w) + β_m·RV_t^(m) [+ β_j·J_t] + ε``

    where:
        - ``RV_t^(d)`` = daily realized variance (1-day average)
        - ``RV_t^(w)`` = weekly realized variance (5-day average)
        - ``RV_t^(m)`` = monthly realized variance (22-day average)
        - ``J_t`` = jump component (RV - BPV, truncated at zero)

    Args:
        variant: Model variant — 'standard', 'jump' (HAR-RV-J),
            or 'cj' (HAR-RV-CJ with separate continuous/jump).
        daily_window: Window for daily RV component (default 1).
        weekly_window: Window for weekly RV component (default 5).
        monthly_window: Window for monthly RV component (default 22).
        use_log: If True, fit in log-RV space (more robust to outliers).
    """

    def __init__(
        self,
        variant: Literal["standard", "jump", "cj"] = "standard",
        daily_window: int = 1,
        weekly_window: int = 5,
        monthly_window: int = 22,
        use_log: bool = False,
    ) -> None:
        self.variant = variant
        self.daily_window = daily_window
        self.weekly_window = weekly_window
        self.monthly_window = monthly_window
        self.use_log = use_log

        self._coefficients: np.ndarray | None = None
        self._residual_std: float = 0.0
        self._rv_series: np.ndarray | None = None
        self._bpv_series: np.ndarray | None = None

    @staticmethod
    def realized_variance(returns: np.ndarray, window: int = 1) -> np.ndarray:
        """Compute rolling realized variance from returns.

        Args:
            returns: 1-D array of log returns.
            window: Number of periods to average over.

        Returns:
            Array of rolling realized variance values.
        """
        rv = returns**2
        if window == 1:
            return rv
        n = len(rv)
        out = np.empty(n, dtype=float)
        cs = np.cumsum(rv)
        out[:window] = cs[:window] / np.arange(1, min(window, n) + 1)
        if n > window:
            out[window:] = (cs[window:] - cs[:-window]) / window
        return out

    @staticmethod
    def bipower_variation(returns: np.ndarray, window: int = 1) -> np.ndarray:
        """Compute rolling bipower variation (robust to jumps).

        BPV_t = (π/2) × (1/n) × Σ|r_{t-i}|·|r_{t-i-1}|

        Args:
            returns: 1-D array of log returns.
            window: Rolling window size.

        Returns:
            Array of bipower variation values.
        """
        abs_r = np.abs(returns)
        bpv_raw = abs_r[1:] * abs_r[:-1] * (np.pi / 2)
        bpv_raw = np.concatenate([[returns[0] ** 2], bpv_raw])
        if window == 1:
            return bpv_raw
        n = len(bpv_raw)
        out = np.empty(n, dtype=float)
        cs = np.cumsum(bpv_raw)
        out[:window] = cs[:window] / np.arange(1, min(window, n) + 1)
        if n > window:
            out[window:] = (cs[window:] - cs[:-window]) / window
        return out

    def _build_features(self, rv: np.ndarray, bpv: np.ndarray | None) -> np.ndarray:
        """Build the regressor matrix from RV/BPV series."""
        n = len(rv)
        rv_d = rv  # daily RV component
        rv_w = self._rolling_mean(rv, self.weekly_window)
        rv_m = self._rolling_mean(rv, self.monthly_window)

        if self.use_log:
            rv_d = np.log(np.maximum(rv_d, 1e-20))
            rv_w = np.log(np.maximum(rv_w, 1e-20))
            rv_m = np.log(np.maximum(rv_m, 1e-20))

        if self.variant == "standard":
            X = np.column_stack([np.ones(n), rv_d, rv_w, rv_m])
        elif self.variant == "jump":
            assert bpv is not None
            jump = np.maximum(rv - bpv, 0.0)
            if self.use_log:
                jump = np.log(np.maximum(jump, 1e-20))
            X = np.column_stack([np.ones(n), rv_d, rv_w, rv_m, jump])
        elif self.variant == "cj":
            assert bpv is not None
            continuous = bpv
            jump = np.maximum(rv - bpv, 0.0)
            if self.use_log:
                continuous = np.log(np.maximum(continuous, 1e-20))
                jump = np.log(np.maximum(jump, 1e-20))
            X = np.column_stack([np.ones(n), continuous, rv_w, rv_m, jump])
        else:
            raise ValueError(f"Unknown variant: {self.variant}")
        return X

    @staticmethod
    def _rolling_mean(x: np.ndarray, window: int) -> np.ndarray:
        """Vectorized rolling mean via cumsum."""
        n = len(x)
        cs = np.cumsum(x)
        out = np.empty(n, dtype=float)
        counts = np.arange(1, min(window, n) + 1, dtype=float)
        out[:window] = cs[:window] / counts
        if n > window:
            out[window:] = (cs[window:] - cs[:-window]) / window
        return out

    def fit(
        self,
        returns: np.ndarray,
        forecast_horizon: int = 1,
    ) -> HARRV:
        """Fit the HAR-RV model via OLS.

        Args:
            returns: 1-D array of daily log returns.
            forecast_horizon: Horizon for the dependent variable
                (RV averaged over next h days). Default 1.

        Returns:
            self
        """
        r = np.asarray(returns, dtype=float).ravel()
        rv = r**2  # daily realized variance proxy
        self._rv_series = rv

        bpv = None
        if self.variant in ("jump", "cj"):
            bpv = self.bipower_variation(r)
            self._bpv_series = bpv

        # Build target: h-day ahead average RV
        if forecast_horizon == 1:
            y = rv[1:]
        else:
            cs = np.cumsum(rv)
            y_end = cs[forecast_horizon:]
            y_start = cs[:-forecast_horizon]
            y = (y_end - y_start) / forecast_horizon
            # Trim to align
            y = y[: len(rv) - forecast_horizon]

        # Build features (lagged by 1)
        X = self._build_features(rv, bpv)
        # Align: X[:-forecast_horizon] predicts y
        trim = max(self.monthly_window, forecast_horizon)
        X_train = X[trim : trim + len(y)]
        y_train = y[trim - 1 : trim - 1 + len(y)]

        # Ensure alignment
        min_len = min(len(X_train), len(y_train))
        X_train = X_train[:min_len]
        y_train = y_train[:min_len]

        if self.use_log:
            y_train = np.log(np.maximum(y_train, 1e-20))

        # OLS: β = (X'X)^{-1} X'y
        try:
            self._coefficients = np.linalg.lstsq(X_train, y_train, rcond=None)[0]
        except np.linalg.LinAlgError:
            logger.warning("HAR-RV OLS failed, using fallback coefficients")
            n_features = X_train.shape[1]
            self._coefficients = np.zeros(n_features)
            self._coefficients[0] = np.mean(y_train)

        residuals = y_train - X_train @ self._coefficients
        self._residual_std = float(np.std(residuals))

        return self

    def forecast(
        self,
        horizon: int = 1,
        returns: np.ndarray | None = None,
    ) -> np.ndarray:
        """Forecast realized variance for the next ``horizon`` steps.

        Uses the direct projection method: if the model was fitted with
        ``forecast_horizon=h``, a single forecast gives the average RV
        over the next h days.  For multi-step, we iterate using the
        forecast as the new daily RV input.

        Args:
            horizon: Number of forecast steps.
            returns: Optional new returns to condition on (otherwise uses
                the last fitted values).

        Returns:
            Array of shape ``(horizon,)`` with annualised volatility forecasts.
        """
        if self._coefficients is None:
            raise RuntimeError("Must call fit() before forecast()")

        if returns is not None:
            r = np.asarray(returns, dtype=float).ravel()
            rv = r**2
            bpv = self.bipower_variation(r) if self.variant in ("jump", "cj") else None
        else:
            assert self._rv_series is not None
            rv = self._rv_series
            bpv = self._bpv_series

        forecasts = np.empty(horizon, dtype=float)
        current_rv = rv.copy()

        for step in range(horizon):
            X = self._build_features(current_rv, bpv)
            pred = float(X[-1] @ self._coefficients)

            if self.use_log:
                pred = np.exp(pred)

            pred = max(pred, 1e-10)
            forecasts[step] = pred

            # Append forecast as next day's RV for multi-step
            current_rv = np.append(current_rv, pred)
            if bpv is not None:
                bpv = np.append(bpv, pred * 0.9)  # approximate BPV

        # Convert daily variance to annualised volatility
        return np.sqrt(forecasts * 252)

    @property
    def coefficients(self) -> np.ndarray | None:
        """Fitted OLS coefficients [intercept, β_d, β_w, β_m, ...]."""
        return self._coefficients

    @property
    def residual_std(self) -> float:
        """Standard deviation of OLS residuals."""
        return self._residual_std
