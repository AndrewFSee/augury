"""Tests for the out-of-sample regime evaluation tools."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from augury.evaluation import (
    drawdown_detection,
    incremental_information,
    regime_stability,
    regime_tail_test,
    rolling_garch_variance,
    rolling_regime_probabilities,
)
from augury.forecasters import GJRGarchForecaster
from augury.regime import HMMRegimeDetector, build_hmm_features

LOOKBACK, REFIT = 252, 63


@pytest.fixture(scope="module")
def switching_returns() -> tuple[pd.Series, np.ndarray]:
    """Persistent two-state volatility regimes (true regimes exist)."""
    rng = np.random.default_rng(1)
    n = 1200
    state = np.zeros(n, dtype=int)
    for t in range(1, n):
        state[t] = state[t - 1] if rng.random() > 0.01 else 1 - state[t - 1]
    r = rng.standard_normal(n) * np.where(state == 1, 0.025, 0.007)
    return pd.Series(r, index=pd.bdate_range("2015-01-01", periods=n)), state


@pytest.fixture(scope="module")
def switching_eval(switching_returns):
    r, _ = switching_returns
    rr = rolling_regime_probabilities(
        r, HMMRegimeDetector(2, random_state=0), lookback=LOOKBACK, refit_every=REFIT
    )
    g = rolling_garch_variance(r, rr.origins, lookback=LOOKBACK, refit_every=REFIT)
    return rr, g


def test_probabilities_shape_and_order(switching_returns, switching_eval):
    r, state = switching_returns
    rr, _ = switching_eval
    assert rr.probs.shape == (len(r) - LOOKBACK + 1, 2)
    np.testing.assert_allclose(rr.probs.sum(axis=1), 1.0, atol=1e-9)
    # last column is the high-vol state: it should track the true crisis state
    truth = state[rr.origins - 1]
    assert ((rr.crisis.to_numpy() > 0.5) == truth).mean() > 0.75
    # origin s is labelled by the date of its last observed return
    assert rr.probs.index[0] == r.index[LOOKBACK - 1]
    assert set(rr.implied_variance.columns) == {1, 5, 21}
    assert (rr.implied_variance[21] > rr.implied_variance[5]).all()


def test_no_lookahead(switching_returns):
    """Appending future data must not change any earlier probability."""
    r, _ = switching_returns
    kw = {"lookback": LOOKBACK, "refit_every": REFIT}
    short = rolling_regime_probabilities(r.iloc[:800], HMMRegimeDetector(2, random_state=0), **kw)
    full = rolling_regime_probabilities(r, HMMRegimeDetector(2, random_state=0), **kw)
    pd.testing.assert_frame_equal(short.probs, full.probs.loc[short.probs.index])


def test_no_lookahead_with_features(switching_returns):
    r, _ = switching_returns
    kw = {
        "lookback": LOOKBACK,
        "refit_every": REFIT,
        "features": lambda x: build_hmm_features(x, standardize=False),
    }
    short = rolling_regime_probabilities(r.iloc[:700], HMMRegimeDetector(2, random_state=0), **kw)
    full = rolling_regime_probabilities(r, HMMRegimeDetector(2, random_state=0), **kw)
    pd.testing.assert_frame_equal(short.probs, full.probs.loc[short.probs.index])


def test_incremental_information_detects_real_regimes(switching_returns, switching_eval):
    """Positive control: with true regimes, P(crisis) adds to GJR at 5 days."""
    r, _ = switching_returns
    rr, g = switching_eval
    res = incremental_information(r, rr, g, horizons=(5, 21))
    assert list(res.index) == [5, 21]
    assert res.loc[5, "t_p"] > 2.0
    assert res.loc[5, "r2_garch_p"] > res.loc[5, "r2_garch"]


def test_incremental_information_null_on_garch_data():
    """Negative control: on GJR-simulated data P(crisis) adds nothing."""
    rng = np.random.default_rng(3)
    n, omega, alpha, gamma, beta = 1500, 0.02, 0.03, 0.10, 0.90
    h, x = omega / (1 - alpha - gamma / 2 - beta), np.empty(n)
    for t in range(n):
        x[t] = np.sqrt(h) * rng.standard_normal()
        h = omega + (alpha + gamma * (x[t] < 0)) * x[t] ** 2 + beta * h
    r = pd.Series(x / 100, index=pd.bdate_range("2012-01-02", periods=n))
    rr = rolling_regime_probabilities(
        r, HMMRegimeDetector(2, random_state=0), lookback=LOOKBACK, refit_every=REFIT
    )
    g = rolling_garch_variance(r, rr.origins, lookback=LOOKBACK, refit_every=REFIT)
    res = incremental_information(r, rr, g, horizons=(5,))
    assert abs(res.loc[5, "t_p"]) < 3.0


def test_tail_test_and_stability(switching_eval):
    rr, g = switching_eval
    tt = regime_tail_test(rr, g, alphas=(0.05,))
    assert tt.loc[(0.05, "all"), "n"] == tt.loc[(0.05, "calm"), "n"] + tt.loc[(0.05, "crisis"), "n"]
    assert 0 <= tt.loc[(0.05, "all"), "kupiec_p"] <= 1
    st = regime_stability(rr)
    assert 0 < st["pct_crisis"] < 1
    assert st["switches_per_year"] > 0
    assert st["mean_revision"] >= 0


def test_garch_variance_pit_is_uniformish(switching_returns, switching_eval):
    _, g = switching_eval
    pit = g["pit1"].dropna()
    assert pit.between(0, 1).all()
    assert abs(pit.mean() - 0.5) < 0.05
    assert np.isnan(g["pit1"].iloc[-1])  # last origin has no next return


def test_expected_variance_matches_simulation():
    rng = np.random.default_rng(0)
    r = rng.standard_t(5, 1000) * 0.01
    m = GJRGarchForecaster(dist="skewt").fit(r)
    ev = m.expected_variance(r, 21)
    assert ev.shape == (21,)
    assert ev[0] == pytest.approx(m.conditional_variance(r))
    sim = m.predict(r, (21,), n_samples=100_000, random_state=0).samples[:, 0]
    assert ev.sum() == pytest.approx(sim.var(), rel=0.03)


def test_innovation_cdf_inverts_innovations():
    r = np.random.default_rng(0).standard_t(5, 800) * 0.01
    m = GJRGarchForecaster(dist="skewt").fit(r)
    u = np.array([0.01, 0.25, 0.5, 0.75, 0.99])
    np.testing.assert_allclose(m.innovation_cdf(m._innovations(u)), u, atol=1e-3)


def test_build_hmm_features_unstandardized_is_raw():
    r = np.random.default_rng(0).normal(0, 0.01, 300)
    raw = build_hmm_features(r, standardize=False)
    np.testing.assert_allclose(raw[:, 0], r)
    z = build_hmm_features(r)
    np.testing.assert_allclose(z.mean(axis=0), 0, atol=1e-12)


def test_drawdown_detection():
    idx = pd.bdate_range("2020-01-01", periods=60)
    r = pd.Series(0.0, index=idx)
    r.iloc[10:20] = -0.02  # ~18% drawdown from day 9, trough day 19
    r.iloc[20:40] = 0.015  # recovers
    early = pd.Series(0, index=idx)
    early.iloc[12:] = 1
    never = pd.Series(0, index=idx)
    dd = drawdown_detection(r, {"early": early, "never": never}, min_drawdown=0.10)
    assert len(dd) == 1
    row = dd.iloc[0]
    assert row["peak"] == idx[9] and row["trough"] == idx[19]
    assert row["early_lag"] == 3
    assert np.isnan(row["never_lag"])
    # 10% is first reached on the 6th down day (day 15)
    assert row["early_vs_cross"] == 3 - 6
