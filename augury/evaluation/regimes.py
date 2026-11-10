"""Out-of-sample evaluation of regime detectors.

A regime has no ground truth, so a detector cannot be scored for accuracy.
What can be measured is whether its signal is *useful* and whether it carries
information beyond a plain volatility forecast. This module provides:

- :func:`rolling_regime_probabilities` -- refits a detector on a rolling window
  and records forward-filtered state probabilities, so each day's regime uses
  only data available that day (parameters *and* feature standardization).
- :func:`rolling_garch_variance` -- the GJR-GARCH benchmark on the same origins.
- :func:`incremental_information` -- does P(crisis) or the regime-implied
  variance predict future realized variance once GJR variance is known?
- :func:`regime_tail_test` -- do GJR VaR breaches cluster in one regime?
- :func:`regime_stability` and :func:`drawdown_detection` -- flip rate, spell
  lengths, revisions on refit and how quickly drawdowns are flagged.

Convention (shared with :func:`augury.evaluation.walk_forward`): origin ``s``
has observed ``returns[:s]`` and is labelled by the date of ``returns[s - 1]``.
Targets start at ``returns[s]``.
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from augury.evaluation.backtests import diebold_mariano, kupiec_test

logger = logging.getLogger(__name__)

__all__ = [
    "RollingRegimes",
    "rolling_regime_probabilities",
    "rolling_garch_variance",
    "incremental_information",
    "regime_tail_test",
    "regime_stability",
    "drawdown_detection",
]

FeatureFn = Callable[[np.ndarray], np.ndarray]


def _as_array(returns: pd.Series | np.ndarray) -> tuple[np.ndarray, pd.Index]:
    if isinstance(returns, pd.Series):
        s = returns.dropna()
        return s.to_numpy(dtype=float), s.index
    r = np.asarray(returns, dtype=float).ravel()
    if not np.all(np.isfinite(r)):
        raise ValueError("returns contain non-finite values")
    return r, pd.RangeIndex(r.size)


@dataclass
class RollingRegimes:
    """Out-of-sample regime probabilities from :func:`rolling_regime_probabilities`.

    Attributes:
        probs: Filtered state probabilities per origin. Columns are ordered by
            ascending state volatility, so the last column is the crisis state.
        implied_variance: Regime-implied variance of the cumulative return
            over each horizon (columns are horizons), from the filtered
            probabilities propagated through the transition matrix.
        origins: Integer origin positions (``returns[:s]`` observed).
        revisions: Per refit, the fraction of the previous block's published
            labels that the new fit would label differently.
        failures: Number of refits that raised and reused the previous fit.
    """

    probs: pd.DataFrame
    implied_variance: pd.DataFrame
    origins: np.ndarray
    revisions: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    failures: int = 0

    @property
    def crisis(self) -> pd.Series:
        """Probability of the highest-volatility state."""
        return self.probs.iloc[:, -1].rename("p_crisis")

    def labels(self, threshold: float | None = None) -> pd.Series:
        """Most likely state, or ``crisis > threshold`` as 0/1 if given."""
        if threshold is not None:
            return (self.crisis > threshold).astype(int).rename("crisis")
        return pd.Series(self.probs.to_numpy().argmax(axis=1), index=self.probs.index)


def _state_moments(detector: Any, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (sigma of first feature per state, transition matrix)."""
    params = detector.get_regime_params()
    sigma = np.array([float(params[i]["sigma"]) for i in range(k)])
    model = getattr(detector, "_model", None)
    trans = getattr(model, "transmat_", None)
    trans = np.eye(k) if trans is None else np.asarray(trans, dtype=float)
    return sigma, trans


def rolling_regime_probabilities(
    returns: pd.Series | np.ndarray,
    detector: Any,
    *,
    features: FeatureFn | None = None,
    lookback: int = 504,
    refit_every: int = 21,
    horizons: Sequence[int] = (1, 5, 21),
    start: str | pd.Timestamp | None = None,
) -> RollingRegimes:
    """Refit a regime detector on a rolling window and filter forward.

    Every ``refit_every`` origins the detector is refit on the last
    ``lookback`` observations, standardized with that window's mean and std.
    Between refits the fitted model's forward filter is run on new data only,
    so P(state at origin s) depends on nothing after ``returns[s - 1]``.

    Args:
        returns: Daily log returns, oldest first.
        detector: Object with ``n_components``, ``fit(X)``,
            ``predict_proba_filtered(X)`` and ``get_regime_params()`` (e.g.
            :class:`~augury.regime.HMMRegimeDetector`). It is refit in place.
        features: Optional map from the full return array to a causal
            ``(T, d)`` raw feature matrix whose first column is returns, e.g.
            ``lambda r: build_hmm_features(r, standardize=False)``.
            ``None`` uses returns alone.
        lookback: Training window length.
        refit_every: Origins between refits.
        horizons: Horizons for the regime-implied variance.
        start: First origin date to evaluate (Series input only).

    Returns:
        :class:`RollingRegimes`.
    """
    r, idx = _as_array(returns)
    n = r.size
    if n <= lookback:
        raise ValueError(f"Need more than lookback={lookback} returns")
    if refit_every < 1:
        raise ValueError("refit_every must be >= 1")
    X_raw = r[:, None] if features is None else np.asarray(features(r), dtype=float)
    if X_raw.ndim == 1:
        X_raw = X_raw[:, None]
    if X_raw.shape[0] != n:
        raise ValueError("features must return one row per return")
    k = int(detector.n_components)
    hs = sorted({int(h) for h in horizons})

    origins = np.arange(lookback, n + 1)
    if start is not None:
        if not isinstance(idx, pd.DatetimeIndex):
            raise ValueError("start requires returns with a DatetimeIndex")
        origins = origins[idx[origins - 1] >= pd.Timestamp(start)]
    m = origins.size
    probs = np.full((m, k), np.nan)
    ivar = np.full((m, len(hs)), np.nan)
    revisions: dict[Any, float] = {}
    failures = 0
    fitted: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None
    prev: tuple[np.ndarray, np.ndarray] | None = None  # (origins, labels) of last block

    for b in range(0, m, refit_every):
        block = origins[b : b + refit_every]
        t = int(block[0])
        train = X_raw[t - lookback : t]
        mu, sd = train.mean(axis=0), train.std(axis=0)
        sd = np.where(sd > 1e-12, sd, 1.0)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                detector.fit((train - mu) / sd)
            sigma, trans = _state_moments(detector, k)
            order = np.argsort(sigma)
            fitted = (mu, sd, order, sigma[order], trans[np.ix_(order, order)])
        except Exception as exc:  # noqa: BLE001 - reuse the previous fit
            failures += 1
            logger.warning(f"regime fit failed at origin {t} ({exc})")
            if fitted is None:
                continue
        assert fitted is not None
        f_mu, f_sd, order, sig, P = fitted

        # Forward filter from the window start to the last obs this block needs.
        lo = t - lookback
        seg = (X_raw[lo : int(block[-1])] - f_mu) / f_sd
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            alpha = np.asarray(detector.predict_proba_filtered(seg), dtype=float)[:, order]

        if prev is not None:
            p_orig, p_lab = prev
            new_lab = alpha[p_orig - 1 - lo].argmax(axis=1)
            revisions[idx[t - 1]] = float(np.mean(new_lab != p_lab))

        rows = alpha[block - 1 - lo]
        probs[b : b + block.size] = rows
        prev = (block, rows.argmax(axis=1))

        # Regime-implied cumulative variance: sum_k (p P^k) . sigma^2
        s2 = (sig * f_sd[0]) ** 2
        pi = rows.copy()
        acc = np.zeros(rows.shape[0])
        for step in range(1, hs[-1] + 1):
            pi = pi @ P
            acc += pi @ s2
            if step in hs:
                ivar[b : b + block.size, hs.index(step)] = acc

    labels = idx[origins - 1]
    cols = [f"state{i}" for i in range(k)]
    keep = ~np.isnan(probs).any(axis=1)
    return RollingRegimes(
        probs=pd.DataFrame(probs[keep], index=labels[keep], columns=cols),
        implied_variance=pd.DataFrame(ivar[keep], index=labels[keep], columns=hs),
        origins=origins[keep],
        revisions=pd.Series(revisions, dtype=float),
        failures=failures,
    )


def rolling_garch_variance(
    returns: pd.Series | np.ndarray,
    origins: np.ndarray,
    horizons: Sequence[int] = (1, 5, 21),
    *,
    forecaster: Any = None,
    lookback: int = 504,
    refit_every: int = 21,
) -> pd.DataFrame:
    """GJR-GARCH cumulative-variance forecasts on given origins.

    Args:
        returns: Daily log returns, oldest first.
        origins: Integer origin positions, e.g. ``RollingRegimes.origins``.
        horizons: Horizons in periods.
        forecaster: A fitted-on-demand
            :class:`~augury.forecasters.GJRGarchForecaster` (default skew-t,
            zero mean).
        lookback: Rolling window length.
        refit_every: Origins between refits.

    Returns:
        DataFrame indexed like the origins with one column per horizon (the
        variance of the h-period cumulative return) and ``pit1``, the PIT of
        the next day's return under the one-day forecast (NaN at the last
        origin if no next return exists).
    """
    from augury.forecasters import GJRGarchForecaster

    model = forecaster if forecaster is not None else GJRGarchForecaster(dist="skewt")
    r, idx = _as_array(returns)
    n = r.size
    hs = sorted({int(h) for h in horizons})
    out = np.full((len(origins), len(hs) + 1), np.nan)
    last: int | None = None
    for i, s in enumerate(np.asarray(origins, dtype=int)):
        window = r[s - lookback : s]
        if last is None or s - last >= refit_every:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model.fit(window)
                last = int(s)
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"GJR fit failed at origin {s} ({exc})")
                if last is None:
                    continue
        daily = model.expected_variance(window, hs[-1])
        cum = np.cumsum(daily)
        out[i, : len(hs)] = cum[[h - 1 for h in hs]]
        if s < n:
            out[i, -1] = float(model.innovation_cdf(r[s] / np.sqrt(daily[0])))
    return pd.DataFrame(out, index=idx[np.asarray(origins) - 1], columns=[*hs, "pit1"])


def _realized_variance(r: np.ndarray, origins: np.ndarray, h: int) -> np.ndarray:
    c = np.concatenate([[0.0], np.cumsum(r**2)])
    rv = np.full(origins.size, np.nan)
    ok = origins + h <= r.size
    rv[ok] = c[origins[ok] + h] - c[origins[ok]]
    return rv


def _ols(y: np.ndarray, X: np.ndarray, lags: int) -> Any:
    import statsmodels.api as sm

    return sm.OLS(y, sm.add_constant(X)).fit(cov_type="HAC", cov_kwds={"maxlags": lags})


def incremental_information(
    returns: pd.Series | np.ndarray,
    regimes: RollingRegimes,
    garch: pd.DataFrame,
    horizons: Sequence[int] = (5, 21),
) -> pd.DataFrame:
    """Test whether the regime signal predicts realized variance beyond GJR.

    For each horizon ``h`` the target is ``log RV``, the log of the sum of
    squared returns over the next ``h`` days. Regressions use Newey-West
    standard errors with ``h`` lags (forecasts overlap):

    - ``r2_garch``: ``log RV ~ log GJR variance``.
    - ``coef_p`` / ``t_p``: adding P(crisis). A significant coefficient means
      the regime carries variance information GJR does not.
    - ``r2_regime``: ``log RV ~ log regime-implied variance`` alone.
    - ``coef_enc`` / ``t_enc``: the regime-implied variance's coefficient when
      both forecasts are in the regression (forecast encompassing).

    It also compares the two variance forecasts directly, out of sample, with
    QLIKE loss (``RV / f + log f``; robust to a noisy RV proxy) and a
    Diebold-Mariano test. A negative ``dm_stat`` favours the regime forecast.

    Args:
        returns: The returns passed to :func:`rolling_regime_probabilities`.
        regimes: Its result.
        garch: :func:`rolling_garch_variance` on ``regimes.origins``.
        horizons: Horizons to test (each must be in both inputs).

    Returns:
        DataFrame indexed by horizon.
    """
    r, _ = _as_array(returns)
    p = regimes.crisis.to_numpy()
    rows = {}
    for h in horizons:
        rv = _realized_variance(r, regimes.origins, h)
        g = garch[h].to_numpy()
        iv = regimes.implied_variance[h].to_numpy()
        ok = np.isfinite(rv) & (rv > 0) & np.isfinite(g) & np.isfinite(iv) & np.isfinite(p)
        y, lg, li, pp = np.log(rv[ok]), np.log(g[ok]), np.log(iv[ok]), p[ok]
        m0 = _ols(y, lg[:, None], h)
        m1 = _ols(y, np.column_stack([lg, pp]), h)
        m2 = _ols(y, li[:, None], h)
        m3 = _ols(y, np.column_stack([lg, li]), h)
        q_g = rv[ok] / g[ok] + lg
        q_i = rv[ok] / iv[ok] + li
        dm, dm_p = diebold_mariano(q_i, q_g, lags=h - 1)
        rows[h] = {
            "n": int(ok.sum()),
            "r2_garch": m0.rsquared,
            "r2_garch_p": m1.rsquared,
            "coef_p": m1.params[2],
            "t_p": m1.tvalues[2],
            "pval_p": m1.pvalues[2],
            "r2_regime": m2.rsquared,
            "coef_enc": m3.params[2],
            "t_enc": m3.tvalues[2],
            "qlike_garch": q_g.mean(),
            "qlike_regime": q_i.mean(),
            "dm_stat": dm,
            "dm_pval": dm_p,
        }
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis("horizon")


def regime_tail_test(
    regimes: RollingRegimes,
    garch: pd.DataFrame,
    alphas: Sequence[float] = (0.01, 0.05),
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Split GJR one-day VaR breaches by regime.

    If GJR is calibrated in both regimes, breach rates match ``alpha`` in each
    and the regime adds nothing for tail risk. A low Kupiec p-value in one
    regime, or a low Fisher p-value (equal breach rates across regimes),
    means the regime tells you something about tails that GJR misses.

    Returns:
        DataFrame indexed by (alpha, regime) with ``n``, ``breaches``,
        ``expected``, ``rate``, ``kupiec_p`` and, on each row of an alpha,
        ``fisher_p``.
    """
    pit = garch["pit1"].to_numpy()
    crisis = regimes.crisis.to_numpy() > threshold
    ok = np.isfinite(pit)
    rows = []
    for a in alphas:
        hits = pit[ok] < a
        cr = crisis[ok]
        table = [
            [int(hits[~cr].sum()), int((~hits[~cr]).sum())],
            [int(hits[cr].sum()), int((~hits[cr]).sum())],
        ]
        fisher_p = float(stats.fisher_exact(table)[1]) if cr.any() and (~cr).any() else np.nan
        for name, mask in (("calm", ~cr), ("crisis", cr), ("all", np.ones_like(cr))):
            hm = hits[mask]
            rows.append(
                {
                    "alpha": a,
                    "regime": name,
                    "n": int(mask.sum()),
                    "breaches": int(hm.sum()),
                    "expected": a * mask.sum(),
                    "rate": hm.mean() if hm.size else np.nan,
                    "kupiec_p": kupiec_test(hm, a)[1] if hm.size else np.nan,
                    "fisher_p": fisher_p,
                }
            )
    return pd.DataFrame(rows).set_index(["alpha", "regime"])


def _spells(flag: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Run lengths of consecutive equal values and the value of each run."""
    if flag.size == 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    change = np.flatnonzero(np.diff(flag)) + 1
    bounds = np.concatenate([[0], change, [flag.size]])
    return np.diff(bounds), flag[bounds[:-1]]


def regime_stability(
    regimes: RollingRegimes,
    threshold: float = 0.5,
    periods_per_year: int = 252,
) -> pd.Series:
    """Summary of how often and how erratically the crisis flag changes.

    Spells cut off by the sample start or end are excluded from the mean
    spell lengths.
    """
    flag = (regimes.crisis.to_numpy() > threshold).astype(int)
    lengths, values = _spells(flag)
    inner = slice(1, -1) if lengths.size > 2 else slice(0, 0)
    li, vi = lengths[inner], values[inner]
    years = flag.size / periods_per_year
    return pd.Series(
        {
            "days": flag.size,
            "pct_crisis": flag.mean(),
            "switches_per_year": (lengths.size - 1) / years if years > 0 else np.nan,
            "mean_calm_spell": li[vi == 0].mean() if (vi == 0).any() else np.nan,
            "mean_crisis_spell": li[vi == 1].mean() if (vi == 1).any() else np.nan,
            "short_crisis_spells": float(np.mean(li[vi == 1] <= 5)) if (vi == 1).any() else np.nan,
            "mean_revision": regimes.revisions.mean(),
            "refit_failures": regimes.failures,
        }
    )


def drawdown_detection(
    returns: pd.Series,
    flags: Mapping[str, pd.Series],
    min_drawdown: float = 0.10,
) -> pd.DataFrame:
    """How quickly each 0/1 flag series fires during large drawdowns.

    An episode runs from a price peak until the price recovers to it (or the
    data end) and qualifies if its depth reaches ``min_drawdown``. For each
    flag, ``<name>_lag`` is the number of trading days from the peak to the
    first day the flag is on (NaN if never during the episode) and
    ``<name>_vs_cross`` the same relative to the day the drawdown first hit
    ``min_drawdown`` (negative means it fired earlier).

    Args:
        returns: Daily log returns with a DatetimeIndex.
        flags: Name to 0/1 Series indexed by date (the flag on a date uses
            data through that date).
        min_drawdown: Minimum peak-to-trough fall, as a fraction.

    Returns:
        DataFrame with one row per episode.
    """
    common = None
    for f in flags.values():
        common = f.index if common is None else common.intersection(f.index)
    if common is None or len(common) == 0:
        raise ValueError("flags share no dates")
    r = returns.reindex(common).fillna(0.0)
    logp = r.cumsum().to_numpy()
    dates = common
    peak_val, peak_i = -np.inf, 0
    episodes = []
    i, n = 0, len(logp)
    while i < n:
        if logp[i] >= peak_val:
            peak_val, peak_i = logp[i], i
            i += 1
            continue
        # in a drawdown from peak_i: run to recovery
        j = i
        while j < n and logp[j] < peak_val:
            j += 1
        seg = logp[peak_i:j]
        trough_rel = int(np.argmin(seg))
        depth = 1 - np.exp(seg[trough_rel] - peak_val)
        if depth >= min_drawdown:
            cross_rel = int(np.argmax(1 - np.exp(seg - peak_val) >= min_drawdown))
            row: dict[str, Any] = {
                "peak": dates[peak_i],
                "cross": dates[peak_i + cross_rel],
                "trough": dates[peak_i + trough_rel],
                "recovered": dates[j] if j < n else pd.NaT,
                "depth": depth,
                "days_to_trough": trough_rel,
            }
            for name, f in flags.items():
                on = np.flatnonzero(f.reindex(dates[peak_i:j]).fillna(0).to_numpy() > 0)
                lag = float(on[0]) if on.size else np.nan
                row[f"{name}_lag"] = lag
                row[f"{name}_vs_cross"] = lag - cross_rel
            episodes.append(row)
        i = j
    return pd.DataFrame(episodes)
