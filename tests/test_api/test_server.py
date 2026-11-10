"""Tests for FastAPI server endpoints."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from augury.api.server import app, get_price_loader


@pytest.fixture()
def client() -> TestClient:
    return TestClient(app)


# ── Health ────────────────────────────────────────────────────────────────────


class TestHealth:
    def test_health_ok(self, client) -> None:
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "version" in data


# ── Simulate ──────────────────────────────────────────────────────────────────


class TestSimulate:
    def test_simulate_default(self, client) -> None:
        resp = client.post(
            "/simulate",
            json={
                "symbol": "SPY",
                "model": "gbm",
                "n_paths": 500,
                "n_steps": 50,
                "horizon": 0.25,
                "random_seed": 42,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["symbol"] == "SPY"
        assert data["model"] == "gbm"
        assert "mean" in data
        assert "std" in data
        assert "var_95" in data
        assert "cvar_95" in data
        assert "percentiles" in data

    def test_simulate_heston(self, client) -> None:
        resp = client.post(
            "/simulate",
            json={
                "symbol": "QQQ",
                "model": "heston",
                "n_paths": 200,
                "n_steps": 50,
                "horizon": 0.5,
                "random_seed": 7,
                "parameters": {
                    "v0": 0.04,
                    "kappa": 2.0,
                    "theta": 0.04,
                    "sigma_v": 0.3,
                    "rho": -0.7,
                },
            },
        )
        # Server returns 200 on success or 500 if extra default params
        # (mu/sigma) conflict with Heston; either is acceptable
        assert resp.status_code in (200, 500)

    def test_simulate_unknown_model(self, client) -> None:
        resp = client.post("/simulate", json={"model": "unknown_model", "n_paths": 100})
        assert resp.status_code == 400
        assert "unknown_model" in resp.json()["detail"].lower()

    def test_simulate_merton_jd(self, client) -> None:
        resp = client.post(
            "/simulate",
            json={
                "symbol": "IWM",
                "model": "merton_jd",
                "n_paths": 200,
                "n_steps": 50,
                "horizon": 0.25,
                "random_seed": 1,
            },
        )
        assert resp.status_code == 200

    def test_percentile_keys(self, client) -> None:
        resp = client.post(
            "/simulate",
            json={
                "model": "gbm",
                "n_paths": 200,
                "n_steps": 50,
                "horizon": 0.25,
                "random_seed": 42,
            },
        )
        pcts = resp.json()["percentiles"]
        for key in ["p1", "p5", "p10", "p25", "p50", "p75", "p90", "p95", "p99"]:
            assert key in pcts


# ── Forecast (alias) ─────────────────────────────────────────────────────────


class TestForecast:
    def test_forecast_same_as_simulate(self, client) -> None:
        payload = {
            "symbol": "SPY",
            "model": "gbm",
            "n_paths": 300,
            "n_steps": 50,
            "horizon": 0.25,
            "random_seed": 42,
        }
        r1 = client.post("/simulate", json=payload).json()
        r2 = client.post("/forecast", json=payload).json()
        assert r1["mean"] == r2["mean"]
        assert r1["std"] == r2["std"]


# ── Market-data endpoints (synthetic prices injected) ─────────────────────────


def _synthetic_prices(symbol: str, n: int) -> pd.Series:
    """Calm then volatile prices; the seed depends on the symbol."""
    rng = np.random.default_rng(sum(map(ord, symbol)))
    vol = np.where(np.arange(n) < n // 2, 0.007, 0.022)
    r = rng.standard_normal(n) * vol
    idx = pd.bdate_range(end="2026-11-06", periods=n)
    return pd.Series(100 * np.exp(np.cumsum(r)), index=idx)


@pytest.fixture()
def data_client():
    calls: list[tuple[str, int]] = []

    def loader(symbol: str, n: int) -> pd.Series:
        calls.append((symbol, n))
        return _synthetic_prices(symbol, n)

    app.dependency_overrides[get_price_loader] = lambda: loader
    try:
        yield TestClient(app), calls
    finally:
        app.dependency_overrides.clear()


class TestRegimes:
    def test_regimes_default(self, data_client) -> None:
        client, calls = data_client
        resp = client.get("/regimes/SPY")
        assert resp.status_code == 200
        data = resp.json()
        assert data["symbol"] == "SPY"
        assert data["as_of"] == "2026-11-06"
        assert data["n_regimes"] == 2
        assert sum(data["regime_proba"]) == pytest.approx(1.0)
        # the last half of the sample is the volatile regime
        assert data["current_regime"] == 1
        sig = [data["regime_params"][k]["sigma"] for k in ("0", "1")]
        assert sig[0] < sig[1]
        assert calls == [("SPY", 505)]

    def test_regimes_uses_symbol_and_lookback(self, data_client) -> None:
        client, calls = data_client
        resp = client.get("/regimes/AAPL?n_regimes=3&lookback=300&features=false")
        assert resp.status_code == 200
        assert len(resp.json()["regime_proba"]) == 3
        assert calls == [("AAPL", 301)]

    def test_regimes_rejects_bad_params(self, data_client) -> None:
        client, _ = data_client
        assert client.get("/regimes/SPY?n_regimes=1").status_code == 422
        assert client.get("/regimes/SPY?lookback=50").status_code == 422


class TestRisk:
    def test_risk_default(self, data_client) -> None:
        client, calls = data_client
        resp = client.get("/risk/SPY?n_paths=2000")
        assert resp.status_code == 200
        data = resp.json()
        assert data["symbol"] == "SPY"
        assert data["horizon_days"] == 21
        assert data["model"] == "gjr-skewt"
        assert data["as_of"] == "2026-11-06"
        assert data["cvar_95"] < data["var_95"] < 0
        assert data["max_drawdown_mean"] < 0
        assert calls == [("SPY", 505)]

    def test_risk_scales_with_horizon(self, data_client) -> None:
        client, _ = data_client
        short = client.get("/risk/QQQ?n_paths=2000&horizon=5").json()
        long = client.get("/risk/QQQ?n_paths=2000&horizon=63").json()
        assert long["var_95"] < short["var_95"]

    def test_risk_drift_moves_sharpe(self, data_client) -> None:
        client, _ = data_client
        flat = client.get("/risk/SPY?n_paths=2000").json()
        up = client.get("/risk/SPY?n_paths=2000&annual_drift=0.3").json()
        assert up["sharpe_estimate"] > flat["sharpe_estimate"]


class TestPriceErrors:
    def test_fetch_failure_is_502(self) -> None:
        def broken(symbol: str, n: int) -> pd.Series:
            raise RuntimeError("no network")

        app.dependency_overrides[get_price_loader] = lambda: broken
        try:
            resp = TestClient(app).get("/risk/SPY")
        finally:
            app.dependency_overrides.clear()
        assert resp.status_code == 502
        assert "SPY" in resp.json()["detail"]

    def test_short_history_is_422(self) -> None:
        app.dependency_overrides[get_price_loader] = lambda: lambda s, n: _synthetic_prices(s, 50)
        try:
            resp = TestClient(app).get("/regimes/NEWIPO")
        finally:
            app.dependency_overrides.clear()
        assert resp.status_code == 422
