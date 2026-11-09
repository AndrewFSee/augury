"""Shared fixtures: return series with known GJR-GARCH dynamics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

#: True GJR parameters in percent-return units.
GJR_TRUE = {"omega": 0.02, "alpha": 0.03, "gamma": 0.12, "beta": 0.88}


def simulate_gjr(n: int, seed: int, df: float = 5.0) -> tuple[np.ndarray, np.ndarray]:
    """Simulate zero-mean GJR-GARCH(1,1) returns with Student-t innovations.

    Returns:
        ``(returns, next_var)`` in decimal units, where ``next_var[t]`` is the
        true conditional variance of ``returns[t]``.
    """
    rng = np.random.default_rng(seed)
    p = GJR_TRUE
    h = p["omega"] / (1 - p["alpha"] - p["gamma"] / 2 - p["beta"])
    z = rng.standard_t(df, size=n) / np.sqrt(df / (df - 2))
    r = np.empty(n)
    var = np.empty(n)
    for t in range(n):
        var[t] = h
        r[t] = np.sqrt(h) * z[t]
        h = p["omega"] + (p["alpha"] + p["gamma"] * (r[t] < 0)) * r[t] ** 2 + p["beta"] * h
    return r / 100, var / 100**2


@pytest.fixture(scope="session")
def gjr_series() -> pd.Series:
    r, _ = simulate_gjr(1600, seed=11)
    return pd.Series(r, index=pd.bdate_range("2018-01-01", periods=r.size))
