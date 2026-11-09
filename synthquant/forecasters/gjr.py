"""GJR-GARCH forecasters: parametric innovations and filtered historical simulation."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
from scipy.signal import lfilter

from synthquant.forecasters.base import (
    ReturnForecast,
    ReturnForecaster,
    stratified_uniforms,
    validate_horizons,
)

logger = logging.getLogger(__name__)

__all__ = ["GJRGarchForecaster", "FilteredHistoricalSimulation"]

Dist = Literal["normal", "t", "skewt"]


def _backcast(x: np.ndarray) -> float:
    """Initial variance as in ``arch``: exponentially weighted mean of early x^2."""
    tau = min(75, x.size)
    w = 0.94 ** np.arange(tau)
    return float(np.sum(w / w.sum() * x[:tau] ** 2))


class GJRGarchForecaster(ReturnForecaster):
    """GJR-GARCH(1,1) with normal, Student-t or Hansen skew-t innovations.

    The model that passed every VaR backtest in example 28. Parameters are
    estimated by maximum likelihood with ``arch`` at :meth:`fit`; at
    :meth:`predict` the conditional variance is filtered through the latest
    history with those parameters, and multi-step distributions are simulated
    through the GJR recursion so volatility clustering and the leverage effect
    compound over the horizon. The first step uses stratified draws, so the
    one-day quantiles have very little Monte Carlo noise.

    Args:
        dist: Innovation distribution: ``"normal"``, ``"t"`` or ``"skewt"``.
        mean: ``"zero"`` (default) or ``"constant"`` (estimated mean return).
        asymmetric: Include the GJR leverage term (False gives plain GARCH).
        scale: Returns are multiplied by this before fitting, for optimiser
            stability (100 = percent returns).
    """

    def __init__(
        self,
        dist: Dist = "skewt",
        mean: Literal["zero", "constant"] = "zero",
        asymmetric: bool = True,
        scale: float = 100.0,
    ) -> None:
        self.dist = dist
        self.mean = mean
        self.asymmetric = asymmetric
        self.scale = scale
        prefix = "gjr" if asymmetric else "garch"
        self.name = f"{prefix}-{dist}"
        self._params: dict[str, float] | None = None
        self._dist_obj: Any = None
        self._dist_params: np.ndarray | None = None
        self._std_resid: np.ndarray | None = None
        self._ppf_table: tuple[np.ndarray, np.ndarray] | None = None

    # ------------------------------------------------------------------ fitting
    def fit(self, returns: np.ndarray) -> GJRGarchForecaster:
        """Estimate GJR-GARCH parameters by maximum likelihood."""
        from arch import arch_model

        r = np.asarray(returns, dtype=float).ravel()
        r = r[np.isfinite(r)] * self.scale
        am = arch_model(
            r,
            mean="Zero" if self.mean == "zero" else "Constant",
            vol="GARCH",
            p=1,
            o=1 if self.asymmetric else 0,
            q=1,
            dist=self.dist,
            rescale=False,
        )
        res = am.fit(disp="off", show_warning=False)
        p = res.params
        self._params = {
            "mu": float(p.get("mu", 0.0)),
            "omega": float(p["omega"]),
            "alpha": float(p["alpha[1]"]),
            "gamma": float(p.get("gamma[1]", 0.0)),
            "beta": float(p["beta[1]"]),
        }
        self._dist_obj = res.model.distribution
        k = self._dist_obj.num_params
        self._dist_params = np.asarray(p.iloc[len(p) - k :], dtype=float) if k else None
        z = np.asarray(res.std_resid, dtype=float)
        z = z[np.isfinite(z)]
        self._std_resid = np.sort((z - z.mean()) / z.std())
        self._ppf_table = None
        if self.dist != "normal":
            # scipy's t / skew-t ppf is slow (root finding); invert a dense CDF
            # table once per fit instead. Linear interpolation on a 0.005-wide
            # grid out to +-30 std is accurate far beyond Monte Carlo error.
            grid = np.linspace(-30.0, 30.0, 12001)
            cdf = np.asarray(self._dist_obj.cdf(grid, self._dist_params), dtype=float)
            cdf, idx = np.unique(cdf, return_index=True)
            self._ppf_table = (cdf, grid[idx])
        return self

    @property
    def params(self) -> dict[str, float]:
        """Fitted parameters in scaled (``scale`` x returns) units."""
        self._check_fitted()
        assert self._params is not None
        out = dict(self._params)
        if self._dist_params is not None:
            names = ["nu"] if self.dist == "t" else ["eta", "lambda"]
            out.update(dict(zip(names, self._dist_params.tolist(), strict=False)))
        return out

    def _check_fitted(self) -> None:
        if self._params is None:
            raise RuntimeError(f"{type(self).__name__} must be fit() before predict()")

    # -------------------------------------------------------------- conditioning
    def conditional_variance(self, history: np.ndarray) -> float:
        """Next-period variance (in return units squared) given ``history``."""
        return self._next_variance(history) / self.scale**2

    def _next_variance(self, history: np.ndarray) -> float:
        self._check_fitted()
        assert self._params is not None
        p = self._params
        x = np.asarray(history, dtype=float).ravel()
        x = x[np.isfinite(x)] * self.scale - p["mu"]
        if x.size == 0:
            raise ValueError("history must contain at least one finite return")
        h0 = _backcast(x)
        # h_{t+1} = beta * h_t + omega + (alpha + gamma * 1[x_t < 0]) * x_t^2
        u = p["omega"] + (p["alpha"] + p["gamma"] * (x < 0)) * x**2
        h = lfilter([1.0], [1.0, -p["beta"]], u, zi=[p["beta"] * h0])[0]
        return float(h[-1])

    def _innovations(self, u: np.ndarray) -> np.ndarray:
        """Map uniforms to standardized (zero-mean, unit-variance) innovations."""
        if self._ppf_table is not None:
            cdf, z = self._ppf_table
            return np.interp(u, cdf, z)
        return np.asarray(self._dist_obj.ppf(u, self._dist_params), dtype=float)

    # ---------------------------------------------------------------- prediction
    def predict(
        self,
        history: np.ndarray,
        horizons: Sequence[int],
        n_samples: int = 2000,
        random_state: int | np.random.Generator | None = None,
    ) -> ReturnForecast:
        hs = validate_horizons(horizons)
        h_next = self._next_variance(history)
        assert self._params is not None
        p = self._params
        rng = np.random.default_rng(random_state)

        u = rng.random((n_samples, hs[-1]))
        u[:, 0] = stratified_uniforms(n_samples, rng)
        z = self._innovations(np.clip(u, 1e-12, 1 - 1e-12).ravel()).reshape(u.shape)

        var = np.full(n_samples, h_next)
        cum = np.zeros(n_samples)
        out = np.empty((n_samples, len(hs)))
        col = {h: j for j, h in enumerate(hs)}
        for k in range(hs[-1]):
            eps = np.sqrt(var) * z[:, k]
            cum += p["mu"] + eps
            if k + 1 in col:
                out[:, col[k + 1]] = cum
            var = p["omega"] + (p["alpha"] + p["gamma"] * (eps < 0)) * eps**2 + p["beta"] * var
        return ReturnForecast(hs, out / self.scale)


class FilteredHistoricalSimulation(GJRGarchForecaster):
    """Filtered historical simulation (Barone-Adesi et al., 1999).

    GJR-GARCH supplies the volatility dynamics; innovations are resampled
    from the model's own standardized residuals instead of a parametric
    distribution, so skew and tail shape come from the data. Innovations are
    drawn from the empirical quantile function, so they can never exceed the
    most extreme residual in the training window.

    Args:
        fit_dist: Distribution used for the (quasi-)MLE variance fit.
        mean: ``"zero"`` (default) or ``"constant"``.
        scale: Fitting scale factor.
    """

    def __init__(
        self,
        fit_dist: Dist = "normal",
        mean: Literal["zero", "constant"] = "zero",
        scale: float = 100.0,
    ) -> None:
        super().__init__(dist=fit_dist, mean=mean, asymmetric=True, scale=scale)
        self.name = "fhs"

    def _innovations(self, u: np.ndarray) -> np.ndarray:
        z = self._std_resid
        assert z is not None
        probs = (np.arange(z.size) + 0.5) / z.size
        return np.interp(u, probs, z)
