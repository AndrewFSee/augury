"""Adapters that let path-simulating forecasters be evaluated like any other."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from synthquant.forecasters.base import ReturnForecast, ReturnForecaster, validate_horizons

__all__ = ["PathModelForecaster"]


class PathModelForecaster(ReturnForecaster):
    """Wrap a model with ``fit(returns)`` and ``forecast(horizon, current_price)``.

    Works with :class:`~synthquant.forecast.ProbabilisticForecaster` and
    :class:`~synthquant.ensemble_forecast.EnsembleForecaster`, whose
    ``forecast()`` returns price paths. The number of samples is the wrapped
    model's ``n_paths``; ``n_samples`` and ``random_state`` are ignored.

    These models condition only on the window they were fit on, so for a
    fair comparison evaluate them with ``refit_every`` equal to
    ``eval_every`` in the walk-forward harness.

    Args:
        factory: Zero-argument callable returning a fresh, unfitted model.
        name: Identifier used in evaluation tables.

    Example:
        >>> from synthquant.forecast import ProbabilisticForecaster
        >>> fc = PathModelForecaster(
        ...     lambda: ProbabilisticForecaster(n_paths=2000, random_state=0),
        ...     name="regime-hmm",
        ... )
    """

    def __init__(self, factory: Callable[[], Any], name: str = "path-model") -> None:
        self.factory = factory
        self.name = name
        self._model: Any = None

    def fit(self, returns: np.ndarray) -> PathModelForecaster:
        self._model = self.factory()
        self._model.fit(np.asarray(returns, dtype=float))
        return self

    def predict(
        self,
        history: np.ndarray,
        horizons: Sequence[int],
        n_samples: int = 2000,
        random_state: int | np.random.Generator | None = None,
    ) -> ReturnForecast:
        if self._model is None:
            raise RuntimeError("PathModelForecaster must be fit() before predict()")
        hs = validate_horizons(horizons)
        res = self._model.forecast(horizon=hs[-1], current_price=1.0)
        paths = np.asarray(res.paths, dtype=float)
        return ReturnForecast(hs, np.log(paths[:, list(hs)]))
