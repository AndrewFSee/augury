"""Tests for event days (earnings) and TermForecaster: calendar, sessions, multiplier, models."""

from __future__ import annotations

import sys
import types

import numpy as np
import pandas as pd
import pytest

from augury.data.events import event_sessions, fetch_earnings_dates, trading_days_after
from augury.forecasters import (
    EventForecaster,
    GaussianForecaster,
    GJRGarchForecaster,
    ImpliedVolForecaster,
    TermForecaster,
    event_multiplier,
)
from augury.monitoring import monitor_model, track_models

NY = "America/New_York"


def test_trading_days_skip_weekends_and_nyse_holidays():
    # Thanksgiving 2026-11-26; Good Friday 2026-04-03; Juneteenth 2026-06-19 (Friday)
    assert list(trading_days_after(pd.Timestamp("2026-11-25"), 3).strftime("%m-%d")) == [
        "11-27",
        "11-30",
        "12-01",
    ]
    assert list(trading_days_after(pd.Timestamp("2026-04-02"), 1).strftime("%m-%d")) == ["04-06"]
    assert list(trading_days_after(pd.Timestamp("2026-06-18"), 1).strftime("%m-%d")) == ["06-22"]


def test_event_sessions_follow_release_time():
    times = [
        pd.Timestamp("2026-11-17 16:00", tz=NY),  # after the close -> next session
        pd.Timestamp("2026-10-13 06:00", tz=NY),  # before the open -> that session
        pd.Timestamp("2026-11-07 07:00", tz=NY),  # Saturday -> Monday
        pd.Timestamp("2026-11-25 16:05"),  # naive = New York; next session skips Thanksgiving
    ]
    assert list(event_sessions(times).strftime("%Y-%m-%d")) == [
        "2026-10-13",
        "2026-11-09",
        "2026-11-18",
        "2026-11-27",
    ]


def _returns_with_events(scale: float, n: int = 1500, seed: int = 3):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2019-01-01", periods=n)
    r = pd.Series(rng.standard_normal(n) * 0.01, index=idx)
    events = idx[30::63]
    r[events] *= scale
    return r, events


def test_event_multiplier_recovers_the_variance_ratio():
    r, ev = _returns_with_events(3.0)
    m = event_multiplier(r, ev, prior_events=0.0, halflife=1e9)
    assert m == pytest.approx(9.0, rel=0.3)
    assert event_multiplier(r, pd.DatetimeIndex([])) == 1.0
    # ordinary "events" carry no extra variance; never below 1
    flat, ev2 = _returns_with_events(1.0)
    assert event_multiplier(flat, ev2) == pytest.approx(1.0, abs=0.35)


def test_event_forecaster_adds_variance_only_when_an_event_is_ahead():
    r, ev = _returns_with_events(3.0)
    origin = r.index[-1]
    ahead = trading_days_after(origin, 3)
    base = GaussianForecaster()
    quiet = EventForecaster(ev, GaussianForecaster(), history=r).fit(r)
    b = base.fit(r.to_numpy()).predict(
        quiet._cleaned(r).to_numpy(), (1, 5), n_samples=4000, random_state=1
    )
    q = quiet.predict(r, (1, 5), n_samples=4000, random_state=1)
    np.testing.assert_allclose(q.samples, b.samples)  # no event in the next 5 days

    soon = ev.append(pd.DatetimeIndex([ahead[0]]))
    prop = EventForecaster(soon, GaussianForecaster(), history=r, size="proportional").fit(r)
    f = prop.predict(r, (1, 5), n_samples=40_000, random_state=1)
    assert prop.last_event_days == [1]
    m = prop.multiplier
    assert m > 3
    assert f.at(1).var() / b.at(1).var() == pytest.approx(m, rel=0.1)
    assert f.at(5).var() / b.at(5).var() == pytest.approx(1 + (m - 1) / 5, rel=0.1)

    absolute = EventForecaster(soon, GaussianForecaster(), history=r).fit(r)
    past = (r[ev] ** 2).mean()
    assert absolute.event_variance == pytest.approx(past, rel=0.35)
    g = absolute.predict(r, (1, 5), n_samples=40_000, random_state=1)
    assert g.at(1).var() == pytest.approx(absolute.event_variance, rel=0.1)


def test_absolute_event_size_does_not_follow_current_vol():
    r, ev = _returns_with_events(3.0)
    origin = r.index[-1]
    soon = ev.append(pd.DatetimeIndex([trading_days_after(origin, 1)[0]]))
    calm = r.copy()
    calm.iloc[-60:] *= 0.5  # quiet lately: the event-day variance stays where past events put it
    a = EventForecaster(soon, GaussianForecaster(), history=calm).fit(calm)
    p = EventForecaster(soon, GaussianForecaster(), history=calm, size="proportional").fit(calm)
    va = a.predict(calm, (1,), n_samples=40_000, random_state=2).at(1).var()
    vp = p.predict(calm, (1,), n_samples=40_000, random_state=2).at(1).var()
    assert va > vp
    with pytest.raises(ValueError):
        EventForecaster(ev, size="huge")


def test_event_forecaster_cleans_event_days_before_fitting():
    r, ev = _returns_with_events(4.0)
    raw = GJRGarchForecaster().fit(r.to_numpy())
    wrapped = EventForecaster(ev, GJRGarchForecaster(), history=r).fit(r)
    assert (
        wrapped.base.params["omega"] < raw.params["omega"]
        or wrapped.base.params["alpha"] < raw.params["alpha"]
    )
    assert np.abs(wrapped._cleaned(r)[ev]).mean() < np.abs(r[ev]).mean()


def test_event_forecaster_needs_dated_returns():
    r, ev = _returns_with_events(2.0)
    with pytest.raises(TypeError):
        EventForecaster(ev).fit(r.to_numpy())


def test_term_forecaster_routes_horizons():
    r, _ = _returns_with_events(1.0)
    wide = ImpliedVolForecaster(pd.Series(0.6, index=r.index), weight=1.0, iv_horizon=6)
    narrow = ImpliedVolForecaster(pd.Series(0.1, index=r.index), weight=1.0, iv_horizon=21)
    t = TermForecaster(narrow, {1: wide}).fit(r)
    f = t.predict(r, (1, 5, 21), n_samples=20_000, random_state=0)
    assert f.horizons == (1, 5, 21)
    assert f.at(1).std() * np.sqrt(252) == pytest.approx(0.6, rel=0.08)
    assert f.at(21).std() * np.sqrt(252 / 21) == pytest.approx(0.1, rel=0.08)


def test_monitor_model_adds_short_view_and_earnings():
    r, ev = _returns_with_events(3.0)
    iv = pd.Series(0.2, index=r.index)
    spy = monitor_model("SPY", iv, short_implied_vol=iv * 0.9)
    assert isinstance(spy, TermForecaster) and set(spy.by_horizon) == {1}
    assert isinstance(
        monitor_model("SPY", iv), ImpliedVolForecaster
    )  # no short series -> as before
    assert isinstance(
        monitor_model("QQQ", iv, short_implied_vol=iv), ImpliedVolForecaster
    )  # not validated there
    times = [pd.Timestamp(d).tz_localize(NY) + pd.Timedelta(hours=16) for d in ev]
    nvda = monitor_model("NVDA", None, earnings=times, returns_history=r)
    assert isinstance(nvda, EventForecaster)
    assert set(track_models("NVDA", None, earnings=times)) == {"augury", "gjr"}
    assert set(track_models("NVDA", None, earnings=[])) == {"augury"}


def test_fetch_earnings_dates(monkeypatch):
    idx = pd.DatetimeIndex(["2026-11-17 16:00", "2026-08-26 16:00"]).tz_localize(NY)

    class FakeTicker:
        def __init__(self, symbol):
            self.symbol = symbol

        def get_earnings_dates(self, limit=100):
            if self.symbol == "SPY":
                raise KeyError("no earnings")
            return pd.DataFrame({"EPS Estimate": [1.0, 0.9]}, index=idx)

    monkeypatch.setitem(sys.modules, "yfinance", types.SimpleNamespace(Ticker=FakeTicker))
    got = fetch_earnings_dates("NVDA")
    assert list(got.strftime("%Y-%m-%d")) == ["2026-08-26", "2026-11-17"]
    assert len(fetch_earnings_dates("SPY")) == 0
