"""Distributional return forecasters sharing one fit/predict interface.

Every forecaster returns a :class:`ReturnForecast`: samples of the cumulative
log return at each requested horizon. Evaluate them with
:func:`synthquant.evaluation.walk_forward`.
"""

from synthquant.forecasters.adapters import PathModelForecaster
from synthquant.forecasters.base import ReturnForecast, ReturnForecaster
from synthquant.forecasters.baselines import (
    GaussianForecaster,
    HistoricalSimulation,
    LaplaceForecaster,
)
from synthquant.forecasters.gjr import FilteredHistoricalSimulation, GJRGarchForecaster
from synthquant.forecasters.har import HARForecaster

__all__ = [
    "ReturnForecast",
    "ReturnForecaster",
    "HistoricalSimulation",
    "GaussianForecaster",
    "LaplaceForecaster",
    "GJRGarchForecaster",
    "FilteredHistoricalSimulation",
    "HARForecaster",
    "PathModelForecaster",
]
