"""Distributional return forecasters sharing one fit/predict interface.

Start with :func:`forecast_prices`, which uses the default model
(GJR-GARCH with skew-t innovations, see :func:`default_forecaster`).

Every forecaster returns a :class:`ReturnForecast`: samples of the cumulative
log return at each requested horizon. Evaluate them with
:func:`augury.evaluation.walk_forward`.
"""

from augury.forecasters.adapters import PathModelForecaster
from augury.forecasters.base import ReturnForecast, ReturnForecaster
from augury.forecasters.baselines import (
    GaussianForecaster,
    HistoricalSimulation,
    LaplaceForecaster,
)
from augury.forecasters.gjr import FilteredHistoricalSimulation, GJRGarchForecaster
from augury.forecasters.har import HARForecaster
from augury.forecasters.price import (
    DEFAULT_LOOKBACK,
    PriceForecast,
    default_forecaster,
    forecast_prices,
)
from augury.forecasters.views import ImpliedVolForecaster

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
    "ImpliedVolForecaster",
]
