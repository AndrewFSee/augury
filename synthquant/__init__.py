"""SynthQuant: probabilistic forecasting of financial returns.

Quick start::

    from synthquant import forecast_prices
    fc = forecast_prices(close_prices, horizon=21)  # GJR-GARCH skew-t by default
    fc.summary()

See https://github.com/AndrewFSee/synthquant for full documentation.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("synthquant")
except PackageNotFoundError:
    __version__ = "0.0.0"

from synthquant.config import Settings, get_settings
from synthquant.forecasters import (
    GJRGarchForecaster,
    PriceForecast,
    ReturnForecast,
    default_forecaster,
    forecast_prices,
)

__all__ = [
    "__version__",
    "Settings",
    "get_settings",
    "forecast_prices",
    "default_forecaster",
    "PriceForecast",
    "ReturnForecast",
    "GJRGarchForecaster",
]
