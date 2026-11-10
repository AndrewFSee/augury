"""FastAPI application server for Augury."""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd

from augury import __version__
from augury.analytics.risk_metrics import (
    conditional_var,
    max_drawdown_distribution,
    tail_ratio,
    value_at_risk,
)
from augury.api.schemas import (
    ForecastResponse,
    HealthResponse,
    RegimeResponse,
    RiskMetrics,
    SimulationRequest,
)
from augury.forecasters.price import DEFAULT_LOOKBACK
from augury.simulation.engines.gbm import GBMEngine
from augury.simulation.engines.heston import HestonEngine
from augury.simulation.engines.merton_jd import MertonJDEngine

logger = logging.getLogger(__name__)

try:
    from fastapi import Depends, FastAPI, HTTPException, Query
    from fastapi.middleware.cors import CORSMiddleware
except ImportError as e:
    raise ImportError(
        "FastAPI is required for the API server. Install with: pip install augury[api]"
    ) from e

app = FastAPI(
    title="Augury API",
    description="Probabilistic price forecasting, risk and regime detection on market data",
    version=__version__,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_ENGINES = {
    "gbm": GBMEngine(),
    "heston": HestonEngine(),
    "merton_jd": MertonJDEngine(),
}


PriceLoader = Callable[[str, int], pd.Series]


def _fetch_prices(symbol: str, n: int) -> pd.Series:
    """Last ``n`` daily adjusted closes for ``symbol`` via :class:`DataIngestor`."""
    from augury.data.ingest import DataIngestor

    # ~1.45 calendar days per trading day, plus slack for holidays
    start = date.today() - timedelta(days=int(n * 1.45) + 30)
    df = DataIngestor().fetch(symbol, start, date.today() + timedelta(days=1))
    return df["close"].dropna().iloc[-n:]


def get_price_loader() -> PriceLoader:
    """Dependency returning the price loader; override in tests."""
    return _fetch_prices


_PRICE_LOADER = Depends(get_price_loader)


def _load(loader: PriceLoader, symbol: str, n: int) -> pd.Series:
    try:
        prices = loader(symbol, n)
    except Exception as exc:  # noqa: BLE001 - data-source errors vary by backend
        raise HTTPException(
            status_code=502, detail=f"Could not fetch prices for {symbol}: {exc}"
        ) from exc
    if len(prices) < min(n, 127):
        raise HTTPException(
            status_code=422,
            detail=f"Only {len(prices)} prices available for {symbol}; need at least 127",
        )
    return prices


def _as_of(prices: pd.Series) -> str | None:
    idx = prices.index
    return str(idx[-1].date()) if isinstance(idx, pd.DatetimeIndex) else None


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health() -> HealthResponse:
    """Liveness check endpoint."""
    return HealthResponse(status="ok", version=__version__)


@app.post("/simulate", response_model=ForecastResponse, tags=["Simulation"])
async def simulate(request: SimulationRequest) -> ForecastResponse:
    """Run a Monte Carlo simulation and return probabilistic forecast statistics.

    Args:
        request: Simulation parameters.

    Returns:
        ForecastResponse with mean, std, VaR, CVaR, and percentiles.
    """
    engine = _ENGINES.get(request.model)
    if engine is None:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model '{request.model}'. Choose from: {list(_ENGINES.keys())}",
        )

    dt = request.horizon / request.n_steps
    params: dict[str, Any] = {"S0": 100.0, "mu": 0.05, "sigma": 0.20}
    params.update(request.parameters)
    params["random_state"] = request.random_seed

    try:
        paths = engine.simulate(
            n_paths=request.n_paths,
            n_steps=request.n_steps,
            dt=dt,
            **params,
        )
    except Exception as exc:
        logger.error(f"Simulation failed: {exc}")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    S0 = paths[:, 0]
    terminal = paths[:, -1]
    log_returns = np.log(terminal / S0)

    percentile_keys = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    percentiles = {f"p{p}": float(np.percentile(log_returns, p)) for p in percentile_keys}

    return ForecastResponse(
        symbol=request.symbol,
        model=request.model,
        horizon=request.horizon,
        n_paths=request.n_paths,
        mean=float(np.mean(log_returns)),
        std=float(np.std(log_returns)),
        var_95=value_at_risk(paths, alpha=0.05),
        cvar_95=conditional_var(paths, alpha=0.05),
        percentiles=percentiles,
    )


@app.get("/regimes/{symbol}", response_model=RegimeResponse, tags=["Regime Detection"])
async def detect_regimes(
    symbol: str,
    n_regimes: int = Query(2, ge=2, le=4),
    lookback: int = Query(DEFAULT_LOOKBACK, ge=126, le=5000),
    features: bool = True,
    load_prices: PriceLoader = _PRICE_LOADER,
) -> RegimeResponse:
    """Today's market regime for ``symbol`` from an HMM fitted to recent prices.

    Fits on the last ``lookback`` daily returns and reports the forward-filtered
    state probabilities on the latest day. States are ordered by return
    volatility, so 0 is the calmest. With ``features`` (the default) the HMM
    uses returns, signed volatility and momentum, which gave far more stable
    regimes than returns alone in out-of-sample tests.

    The regime is descriptive. Out of sample it added almost nothing to a
    GJR-GARCH volatility forecast (see ``augury.evaluation.regimes``), so use
    ``/risk`` for risk numbers.
    """
    from augury.regime import HMMRegimeDetector, build_hmm_features

    prices = _load(load_prices, symbol, lookback + 1)
    r = np.diff(np.log(prices.to_numpy(dtype=float)))[-lookback:]
    X = build_hmm_features(r) if features else (r - r.mean()) / r.std()
    detector = HMMRegimeDetector(n_components=n_regimes, random_state=42)
    try:
        detector.fit(X)
        proba = detector.predict_proba_filtered(X)
    except Exception as exc:  # noqa: BLE001 - surface fit failures as 500s
        raise HTTPException(status_code=500, detail=f"Regime fit failed: {exc}") from exc

    # Describe each state by the returns it explains, in annualized units.
    w = proba / np.maximum(proba.sum(axis=0), 1e-12)
    mu = w.T @ r
    sigma = np.sqrt(np.maximum(w.T @ r**2 - mu**2, 0.0))
    order = np.argsort(sigma)
    current = proba[-1, order]
    params = {
        str(i): {"mu": float(mu[j] * 252), "sigma": float(sigma[j] * np.sqrt(252))}
        for i, j in enumerate(order)
    }
    return RegimeResponse(
        symbol=symbol,
        as_of=_as_of(prices),
        current_regime=int(np.argmax(current)),
        regime_proba=current.tolist(),
        regime_params=params,
        n_regimes=n_regimes,
    )


@app.post("/forecast", response_model=ForecastResponse, tags=["Simulation"])
async def forecast(request: SimulationRequest) -> ForecastResponse:
    """Alias for /simulate with semantic forecast framing."""
    return await simulate(request)


@app.get("/risk/{symbol}", response_model=RiskMetrics, tags=["Analytics"])
async def risk(
    symbol: str,
    horizon: int = Query(21, ge=1, le=252, description="Horizon in trading days"),
    n_paths: int = Query(10_000, ge=500, le=200_000),
    annual_drift: float = 0.0,
    lookback: int = Query(DEFAULT_LOOKBACK, ge=126, le=5000),
    load_prices: PriceLoader = _PRICE_LOADER,
) -> RiskMetrics:
    """Forward-looking risk for ``symbol`` from the default GJR-GARCH skew-t forecast.

    Fits :func:`augury.forecast_prices` to the last ``lookback`` daily returns
    and simulates ``n_paths`` daily price paths over ``horizon`` trading days.
    The drift is zero unless ``annual_drift`` is given, so ``sharpe_estimate``
    only reflects that assumption.
    """
    from augury.forecasters import forecast_prices

    prices = _load(load_prices, symbol, lookback + 1)
    try:
        fc = forecast_prices(
            prices,
            horizon,
            lookback=lookback,
            n_samples=n_paths,
            annual_drift=annual_drift,
            random_state=42,
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Forecast failed: {exc}") from exc

    paths = np.column_stack([np.full(n_paths, fc.last_price), fc.prices])
    log_ret = np.log(paths[:, -1] / fc.last_price)
    sharpe = float(np.mean(log_ret) / (np.std(log_ret) + 1e-12) * np.sqrt(252 / horizon))
    return RiskMetrics(
        symbol=symbol,
        as_of=_as_of(prices),
        horizon_days=horizon,
        model=fc.model,
        last_price=fc.last_price,
        var_95=fc.value_at_risk(0.05),
        cvar_95=fc.expected_shortfall(0.05),
        max_drawdown_mean=float(np.mean(max_drawdown_distribution(paths))),
        tail_ratio=tail_ratio(paths, alpha=0.05),
        sharpe_estimate=sharpe,
    )
