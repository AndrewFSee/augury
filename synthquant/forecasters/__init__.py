"""Distributional return forecasters sharing one fit/predict interface.

Start with :func:`forecast_prices`, which uses the default model
(GJR-GARCH with skew-t innovations, see :func:`default_forecaster`).

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
from synthquant.forecasters.price import (
    DEFAULT_LOOKBACK,
    PriceForecast,
    default_forecaster,
    forecast_prices,
)

__all__ = [
    "forecast_prices",
    "default_forecaster",
    "PriceForecast",
    "DEFAULT_LOOKBACK",
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
