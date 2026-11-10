"""Pydantic request and response schemas for the Augury API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

__all__ = [
    "SimulationRequest",
    "ForecastResponse",
    "RegimeResponse",
    "RiskMetrics",
    "HealthResponse",
]


class SimulationRequest(BaseModel):
    """Request body for the /simulate endpoint."""

    symbol: str = Field(default="SPY", description="Ticker symbol")
    model: str = Field(
        default="gbm",
        description="Simulation model: 'gbm', 'heston', 'merton_jd', 'regime_switching'",
    )
    n_paths: int = Field(default=10_000, ge=100, le=500_000, description="Number of MC paths")
    horizon: float = Field(default=1.0, gt=0, description="Time horizon in years")
    n_steps: int = Field(default=252, ge=1, description="Number of time steps")
    parameters: dict[str, Any] = Field(
        default_factory=dict,
        description="Model-specific parameters (e.g., mu, sigma, v0, kappa)",
    )
    random_seed: int | None = Field(default=None, description="Random seed for reproducibility")


class ForecastResponse(BaseModel):
    """Probabilistic forecast summary statistics."""

    symbol: str
    model: str
    horizon: float
    n_paths: int
    mean: float = Field(description="Mean terminal log-return")
    std: float = Field(description="Standard deviation of terminal log-returns")
    var_95: float = Field(description="95% Value at Risk (5th percentile log-return)")
    cvar_95: float = Field(description="95% Conditional VaR (Expected Shortfall)")
    percentiles: dict[str, float] = Field(
        description="Key percentiles: p1, p5, p10, p25, p50, p75, p90, p95, p99"
    )


class RegimeResponse(BaseModel):
    """Regime detection response."""

    symbol: str
    as_of: str | None = Field(default=None, description="Date of the last price used")
    current_regime: int = Field(
        description="Most likely current regime; regimes are ordered calm (0) to volatile"
    )
    regime_proba: list[float] = Field(description="Filtered probability of each regime today")
    regime_params: dict[str, dict[str, float]] = Field(
        description="Annualized mean ('mu') and volatility ('sigma') of returns per regime"
    )
    n_regimes: int


class RiskMetrics(BaseModel):
    """Computed risk metrics for a position."""

    symbol: str
    as_of: str | None = Field(default=None, description="Date of the last price used")
    horizon_days: int | None = Field(default=None, description="Horizon in trading days")
    model: str | None = Field(default=None, description="Forecaster that produced the paths")
    last_price: float | None = Field(default=None, description="Price at the forecast origin")
    var_95: float = Field(description="95% VaR: 5th percentile simple return over the horizon")
    cvar_95: float = Field(description="95% expected shortfall (simple return)")
    max_drawdown_mean: float = Field(description="Mean maximum drawdown across paths")
    tail_ratio: float = Field(description="Upper/lower tail ratio")
    sharpe_estimate: float = Field(
        description="Annualized mean/std of simulated returns; reflects the drift assumption"
    )


class HealthResponse(BaseModel):
    """API liveness check response."""

    status: str = "ok"
    version: str
