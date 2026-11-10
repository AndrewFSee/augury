"""Tests for augury.monitoring (dashboard / hub publisher helpers)."""

from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from augury.forecasters import GJRGarchForecaster, ImpliedVolForecaster
from augury.monitoring import (
    HORIZONS,
    last_final_session,
    live_forecast,
    monitor_model,
    score_track,
    track_models,
    track_record,
)

NY = ZoneInfo("America/New_York")


@pytest.fixture(scope="module")
def prices() -> pd.Series:
    rng = np.random.default_rng(7)
    n = 800
    var, r = np.empty(n), np.empty(n)
    var[0] = 1e-4
    for t in range(n):
        if t:
            var[t] = 2e-6 + 0.06 * r[t - 1] ** 2 + 0.92 * var[t - 1]
        r[t] = np.sqrt(var[t]) * rng.standard_t(6) / np.sqrt(1.5)
    return pd.Series(100 * np.exp(np.cumsum(r)), index=pd.bdate_range("2022-01-03", periods=n))


@pytest.fixture(scope="module")
def iv(prices: pd.Series) -> pd.Series:
    return pd.Series(0.2, index=prices.index)


def test_last_final_session():
    assert str(last_final_session(datetime(2026, 11, 10, 15, 0, tzinfo=NY))) == "2026-11-09"
    assert str(last_final_session(datetime(2026, 11, 10, 16, 30, tzinfo=NY))) == "2026-11-10"
    monday, sunday = datetime(2026, 11, 9, 9, 0, tzinfo=NY), datetime(2026, 11, 8, 20, 0, tzinfo=NY)
    assert str(last_final_session(monday)) == "2026-11-06"
    assert str(last_final_session(sunday)) == "2026-11-06"


def test_monitor_model_uses_views_only_where_validated(iv):
    assert isinstance(monitor_model("SPY", iv), ImpliedVolForecaster)
    assert monitor_model("TLT", iv).calibrate  # MOVE is not TLT's own vol
    assert isinstance(monitor_model("spy", None), GJRGarchForecaster)
    assert isinstance(monitor_model("NVDA", iv), GJRGarchForecaster)
    assert set(track_models("SPY", iv)) == {"augury", "gjr"}
    assert set(track_models("NVDA", iv)) == {"augury"}


def test_live_forecast(prices, iv):
    risk, bands = live_forecast(prices, monitor_model("SPY", iv), n_samples=4000)
    assert list(risk["horizon"]) == list(HORIZONS)
    assert (risk["var_99"] < risk["var_95"]).all() and (risk["var_95"] < 0).all()
    assert (risk["es_99"] <= risk["var_99"]).all()
    assert (risk["lo_90"] < risk["median"]).all() and (risk["median"] < risk["hi_90"]).all()
    assert risk["p_up"].between(0, 1).all()
    assert risk["var_99"].is_monotonic_decreasing  # longer horizon, bigger loss
    assert len(bands) == 21 and bands.index[0] > prices.index[-1]
    assert (bands.diff(axis=1).iloc[:, 1:] > 0).all().all()  # q5 < q25 < ... < q95


def test_track_record_extends_like_a_full_rebuild(prices, iv):
    start = prices.index[-130]
    full = track_record(prices, track_models("SPY", iv), start=start, n_samples=500)
    early = track_record(prices.iloc[:-40], track_models("SPY", iv), start=start, n_samples=500)
    extended = track_record(prices, track_models("SPY", iv), previous=early, redo_days=30,
                            n_samples=500)
    key = ["origin", "model", "horizon"]
    assert len(extended) == len(full)
    a = extended.set_index(key).sort_index()
    b = full.set_index(key).sort_index()
    one = a.index.get_level_values("horizon") == 1
    # same fits on the same dates; only Monte Carlo noise differs
    assert np.allclose(a.loc[one, "std"], b.loc[one, "std"], rtol=0.1)
    # 21-day outcomes that were still open in `early` are filled in
    assert a.xs(21, level="horizon").index.get_level_values("origin").max() > \
        early[early["horizon"] == 21]["origin"].max()


def test_score_track(prices, iv):
    rec = track_record(prices, track_models("SPY", iv), start=prices.index[-300], n_samples=500)
    sc = score_track(rec, "augury", baseline="gjr", recent=100)
    assert sc["n"] == (rec[(rec["model"] == "augury") & (rec["horizon"] == 1)]).shape[0]
    v = sc["var"]["0.01"]
    assert v["recent_n"] == 100 and v["recent_zone"] in {"Green", "Yellow", "Red"}
    assert 0 <= v["rate"] <= 1 and 0 <= v["kupiec_p"] <= 1
    assert set(sc["coverage_90"]) == set(HORIZONS) and set(sc["skill"]) == set(HORIZONS)
    assert score_track(rec, "missing") == {"n": 0}
