"""Common interface for distributional return forecasters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

__all__ = ["ReturnForecast", "ReturnForecaster", "quantile_grid", "stratified_uniforms"]


@dataclass(frozen=True)
class ReturnForecast:
    """Forecast distribution of cumulative log returns at several horizons.

    Attributes:
        horizons: Forecast horizons in periods (e.g. trading days), ascending.
        samples: Array of shape ``(n_samples, n_horizons)``; column ``j`` holds
            draws of the cumulative log return over the next ``horizons[j]``
            periods. Columns are valid *marginal* distributions; whether rows
            form coherent paths depends on the model (simulation models yes,
            per-horizon parametric models no).
    """

    horizons: tuple[int, ...]
    samples: np.ndarray

    def __post_init__(self) -> None:
        if self.samples.ndim != 2 or self.samples.shape[1] != len(self.horizons):
            raise ValueError(
                f"samples must have shape (n_samples, {len(self.horizons)}), "
                f"got {self.samples.shape}"
            )

    def at(self, horizon: int) -> np.ndarray:
        """Samples of the cumulative log return at ``horizon``."""
        return self.samples[:, self.horizons.index(horizon)]

    def quantiles(self, q: Sequence[float] | np.ndarray) -> np.ndarray:
        """Quantiles of cumulative log return, shape ``(len(q), n_horizons)``."""
        return np.quantile(self.samples, np.asarray(q, dtype=float), axis=0)

    def price_quantiles(self, price: float, q: Sequence[float] | np.ndarray) -> np.ndarray:
        """Price quantiles given today's ``price``, shape ``(len(q), n_horizons)``."""
        return price * np.exp(self.quantiles(q))


class ReturnForecaster(ABC):
    """Base class for models that forecast the distribution of future log returns.

    The contract separates slow parameter estimation from fast conditioning:

    * :meth:`fit` estimates parameters from a training window. It may be
      expensive and is called only at refit dates.
    * :meth:`predict` conditions the fitted model on the most recent history
      (e.g. filters the GARCH variance up to today) and returns forecast
      samples. It must use only the ``history`` it is given.

    Models whose "parameters" are just rolling statistics (historical
    simulation, Laplace, EWMA) can make :meth:`fit` a no-op and estimate
    everything in :meth:`predict`.
    """

    #: Short identifier used in evaluation tables.
    name: str = "base"

    def fit(self, returns: np.ndarray) -> ReturnForecaster:
        """Estimate model parameters from a window of log returns."""
        return self

    @abstractmethod
    def predict(
        self,
        history: np.ndarray,
        horizons: Sequence[int],
        n_samples: int = 2000,
        random_state: int | np.random.Generator | None = None,
    ) -> ReturnForecast:
        """Forecast cumulative log returns over each horizon.

        Args:
            history: Log returns up to and including the forecast origin,
                oldest first.
            horizons: Positive horizons in periods.
            n_samples: Number of draws per horizon.
            random_state: Seed or Generator for any simulation.

        Returns:
            :class:`ReturnForecast` with ``n_samples`` draws per horizon.
        """

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r})"


def validate_horizons(horizons: Sequence[int]) -> tuple[int, ...]:
    """Return horizons as a sorted tuple of unique positive ints."""
    hs = tuple(sorted({int(h) for h in horizons}))
    if not hs or hs[0] < 1:
        raise ValueError(f"horizons must be positive integers, got {horizons}")
    return hs


def quantile_grid(n: int) -> np.ndarray:
    """Midpoint probability grid ``(i + 0.5) / n``.

    Mapping it through a quantile function gives deterministic, evenly
    spaced "samples" whose CRPS and quantiles carry no Monte Carlo noise.
    """
    return (np.arange(n) + 0.5) / n


def stratified_uniforms(n: int, rng: np.random.Generator) -> np.ndarray:
    """One uniform per stratum ``[i/n, (i+1)/n)``, in random order."""
    return rng.permutation((np.arange(n) + rng.random(n)) / n)
