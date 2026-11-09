"""Regression tests that make HMM forward-backward lookahead explicit.

These tests document *why* forecasting uses filtered (forward-only)
posteriors. On a synthetic series with a sharp regime break, the
smoothed posterior at times preceding the break is pulled toward the
post-break regime because forward-backward smoothing incorporates
future observations. The filtered posterior does not have this
property.
"""

from __future__ import annotations

import numpy as np
import pytest

from augury.regime.hmm import HMMRegimeDetector


@pytest.fixture
def regime_break_returns() -> tuple[np.ndarray, int]:
    """500-day return series with a sharp vol regime break at t=300."""
    rng = np.random.default_rng(0)
    calm = rng.normal(0.0005, 0.005, size=300)
    crisis = rng.normal(-0.001, 0.030, size=200)
    return np.concatenate([calm, crisis]), 300


def test_smoothed_leaks_future_near_break(regime_break_returns):
    """In the window just before a regime break, the smoothed posterior
    is pulled toward the post-break regime (lookahead), while the filtered
    posterior — which cannot see future data — stays in the pre-break
    regime.
    """
    returns, break_point = regime_break_returns
    det = HMMRegimeDetector(n_components=2, random_state=0).fit(returns)

    smoothed = det.predict_proba(returns)
    filtered = det.predict_proba_filtered(returns)
    crisis_state = 1

    # Look at the 10 days immediately preceding the break.
    pre = slice(break_point - 10, break_point)
    smoothed_near = smoothed[pre, crisis_state].mean()
    filtered_near = filtered[pre, crisis_state].mean()

    # Effect is small in absolute terms because emission variances are very
    # different, but the ratio should be large. Require both a small
    # absolute gap and a meaningful ratio.
    assert smoothed_near > filtered_near + 0.02, (
        f"Smoothed pre-break crisis prob {smoothed_near:.4f} is not "
        f"meaningfully higher than filtered {filtered_near:.4f} — "
        "lookahead contamination should pull smoothed higher near the break."
    )
    assert smoothed_near > filtered_near * 5, (
        f"Smoothed/filtered ratio {smoothed_near / max(filtered_near, 1e-12):.1f} "
        "too small — expected lookahead to inflate by at least 5×."
    )


def test_filtered_recovers_post_break_regime(regime_break_returns):
    """Filtered posterior should identify crisis *after* the break."""
    returns, break_point = regime_break_returns
    det = HMMRegimeDetector(n_components=2, random_state=0).fit(returns)

    filtered = det.predict_proba_filtered(returns)
    crisis_state = 1

    post_break = filtered[break_point + 20 :, crisis_state].mean()
    assert post_break > 0.7, (
        f"Filtered posterior fails to identify crisis after break "
        f"(mean P(crisis)={post_break:.4f})."
    )


def test_filtered_at_cutoff_uses_only_past(regime_break_returns):
    """P(state_T | x_1..T) must not change when future data is appended."""
    returns, _ = regime_break_returns
    det = HMMRegimeDetector(n_components=2, random_state=0).fit(returns)

    cutoff = 200
    short = det.predict_proba_filtered(returns[:cutoff])[-1]
    long = det.predict_proba_filtered(returns)[cutoff - 1]

    np.testing.assert_allclose(short, long, atol=1e-10)
