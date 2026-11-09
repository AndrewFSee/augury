"""Tests for WebSocket endpoints."""

from __future__ import annotations

import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from augury.api.websocket import register_websocket_routes


@pytest.fixture()
def ws_app() -> FastAPI:
    """Create a minimal FastAPI app with WS routes registered."""
    app = FastAPI()
    register_websocket_routes(app)
    return app


@pytest.fixture()
def ws_client(ws_app) -> TestClient:
    return TestClient(ws_app)


# ── register_websocket_routes ─────────────────────────────────────────────────


def test_register_rejects_non_fastapi() -> None:
    with pytest.raises(TypeError, match="FastAPI"):
        register_websocket_routes("not_an_app")  # type: ignore[arg-type]


def test_routes_registered(ws_app) -> None:
    routes = [r.path for r in ws_app.routes]
    assert "/ws/regimes/{symbol}" in routes
    assert "/ws/forecast/{symbol}" in routes


# ── Regime WebSocket stream ───────────────────────────────────────────────────


class TestRegimeStream:
    def test_receives_regime_update(self, ws_client) -> None:
        with ws_client.websocket_connect("/ws/regimes/SPY") as ws:
            raw = ws.receive_text()
            msg = json.loads(raw)
            assert msg["event"] == "regime_update"
            assert msg["symbol"] == "SPY"
            assert "current_regime" in msg
            assert "regime_proba" in msg
            assert len(msg["regime_proba"]) == 2

    def test_regime_proba_sums_to_one(self, ws_client) -> None:
        with ws_client.websocket_connect("/ws/regimes/AAPL") as ws:
            msg = json.loads(ws.receive_text())
            total = sum(msg["regime_proba"])
            assert abs(total - 1.0) < 1e-6


# ── Forecast WebSocket stream ────────────────────────────────────────────────


class TestForecastStream:
    def test_receives_forecast_update(self, ws_client) -> None:
        with ws_client.websocket_connect("/ws/forecast/SPY") as ws:
            raw = ws.receive_text()
            msg = json.loads(raw)
            assert msg["event"] == "forecast_update"
            assert msg["symbol"] == "SPY"
            for key in ("mean", "std", "var_95", "cvar_95", "p50", "p5", "p95"):
                assert key in msg

    def test_forecast_values_are_floats(self, ws_client) -> None:
        with ws_client.websocket_connect("/ws/forecast/QQQ") as ws:
            msg = json.loads(ws.receive_text())
            assert isinstance(msg["mean"], float)
            assert isinstance(msg["std"], float)
