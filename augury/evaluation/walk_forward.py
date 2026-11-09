"""Walk-forward (out-of-sample) evaluation of distributional forecasters."""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from augury.evaluation.backtests import (
    basel_zone,
    christoffersen_test,
    conditional_coverage_test,
    diebold_mariano,
    kupiec_test,
)
from augury.evaluation.metrics import crps_ensemble
from augury.forecasters.base import ReturnForecaster, validate_horizons

logger = logging.getLogger(__name__)

__all__ = ["walk_forward", "WalkForwardResult"]


def _pct(x: float) -> str:
    return f"{100 * x:g}"


@dataclass
class WalkForwardResult:
    """Out-of-sample forecast records and the summaries computed from them.

    Attributes:
        records: One row per (origin, model, horizon) with the realised
            cumulative log return ``y``, ``crps``, ``pit``, forecast ``mean``
            and ``std``, VaR quantiles ``var_<a>`` and violation flags
            ``hit_<a>``, and interval coverage ``cov_<l>`` / ``width_<l>``.
        horizons: Evaluated horizons.
        var_levels: VaR tail probabilities.
        interval_levels: Central interval probabilities.
        eval_every: Spacing between forecast origins, in periods.
        baseline: Default model the others are compared against.
    """

    records: pd.DataFrame
    horizons: tuple[int, ...]
    var_levels: tuple[float, ...]
    interval_levels: tuple[float, ...]
    eval_every: int
    baseline: str | None = None
    fit_failures: dict[str, int] = field(default_factory=dict)

    @property
    def models(self) -> list[str]:
        """Model names in evaluation order."""
        return list(dict.fromkeys(self.records["model"]))

    def _overlap_lags(self, horizon: int) -> int:
        return max(math.ceil(horizon / self.eval_every) - 1, 0)

    def summary(self, baseline: str | None = None) -> pd.DataFrame:
        """Score table per model and horizon.

        Columns:
            ``n``: number of scored forecasts.
            ``crps``: mean CRPS (lower is better), in log-return units.
            ``skill``: ``1 - crps / crps_baseline`` on common origins (>0 beats it).
            ``dm_p``: two-sided Diebold-Mariano p-value of the CRPS difference
                versus the baseline, with HAC lags for overlapping horizons.
            ``pit_ks_p``: KS test of PIT uniformity; only reported when
                forecasts don't overlap (overlap invalidates the test).
            ``cov_<l>``: empirical coverage of the central l% interval.
            ``width_<l>``: mean width of that interval.
            ``hit_<a>``: VaR violation rate at tail probability a.

        Args:
            baseline: Model to compare against (defaults to ``self.baseline``).
        """
        base = baseline if baseline is not None else self.baseline
        df = self.records
        rows = []
        for key, g in df.groupby(["model", "horizon"], sort=False):
            model, h = str(key[0]), int(key[1])  # type: ignore[index,call-overload]
            row: dict[str, object] = {"model": model, "horizon": h, "n": len(g)}
            row["crps"] = g["crps"].mean()
            if base is not None and model != base:
                b = df[(df["model"] == base) & (df["horizon"] == h)]
                m = g.merge(b[["origin", "crps"]], on="origin", suffixes=("", "_b"))
                if len(m) >= 3:
                    row["skill"] = 1.0 - m["crps"].mean() / m["crps_b"].mean()
                    row["dm_p"] = diebold_mariano(
                        m["crps"].to_numpy(), m["crps_b"].to_numpy(), self._overlap_lags(h)
                    )[1]
            elif model == base:
                row["skill"] = 0.0
            if self._overlap_lags(h) == 0 and len(g) >= 10:
                row["pit_ks_p"] = stats.kstest(g["pit"].to_numpy(), "uniform").pvalue
            for lvl in self.interval_levels:
                row[f"cov_{_pct(lvl)}"] = g[f"cov_{_pct(lvl)}"].mean()
                row[f"width_{_pct(lvl)}"] = g[f"width_{_pct(lvl)}"].mean()
            for a in self.var_levels:
                row[f"hit_{_pct(a)}"] = g[f"hit_{_pct(a)}"].mean()
            rows.append(row)
        out = pd.DataFrame(rows)
        cols = ["model", "horizon", "n", "crps", "skill", "dm_p", "pit_ks_p"]
        return out.reindex(columns=cols + [c for c in out.columns if c not in cols])

    def var_backtest(self, horizon: int = 1) -> pd.DataFrame:
        """Kupiec, Christoffersen and Basel backtests of the VaR forecasts.

        Most meaningful for ``horizon=1`` with ``eval_every=1`` (one VaR
        forecast per day, as in regulatory backtesting).
        """
        df = self.records[self.records["horizon"] == horizon]
        rows = []
        for model, g in df.groupby("model", sort=False):
            for a in self.var_levels:
                hits = g[f"hit_{_pct(a)}"].to_numpy(dtype=bool)
                rows.append(
                    {
                        "model": model,
                        "alpha": a,
                        "n": hits.size,
                        "violations": int(hits.sum()),
                        "rate": hits.mean(),
                        "kupiec_p": kupiec_test(hits, a)[1],
                        "christoffersen_p": christoffersen_test(hits)[1],
                        "cc_p": conditional_coverage_test(hits, a)[1],
                        "basel": basel_zone(int(hits.sum()), hits.size, a),
                    }
                )
        return pd.DataFrame(rows)

    def pit(self, model: str, horizon: int) -> np.ndarray:
        """PIT values for one model and horizon, in time order."""
        df = self.records
        return df[(df["model"] == model) & (df["horizon"] == horizon)]["pit"].to_numpy()


def walk_forward(
    returns: pd.Series | np.ndarray,
    forecasters: Sequence[ReturnForecaster] | Mapping[str, ReturnForecaster],
    *,
    horizons: Sequence[int] = (1, 5, 21),
    lookback: int = 504,
    refit_every: int = 21,
    eval_every: int = 1,
    n_samples: int = 2000,
    var_levels: Sequence[float] = (0.01, 0.05),
    interval_levels: Sequence[float] = (0.5, 0.9),
    start: str | pd.Timestamp | None = None,
    baseline: str | None = None,
    random_state: int | None = 0,
) -> WalkForwardResult:
    """Evaluate forecasters out of sample with a rolling window.

    At each forecast origin ``t`` (every ``eval_every`` periods after the
    first ``lookback``), each model sees only ``returns[t - lookback : t]``.
    Parameters are re-estimated with :meth:`ReturnForecaster.fit` every
    ``refit_every`` periods on that same window; in between, :meth:`predict`
    conditions the last fit on the latest window. The forecast for horizon
    ``h`` is scored against ``sum(returns[t : t + h])``. All models share the
    same origins, so their scores are directly comparable.

    Args:
        returns: Log returns, oldest first. A Series with a DatetimeIndex
            labels each origin with the date of its last observed return.
        forecasters: Models to evaluate, as a sequence (keyed by ``.name``)
            or a mapping of name to model.
        horizons: Forecast horizons in periods.
        lookback: Rolling window length for fitting and conditioning.
        refit_every: Periods between parameter re-estimations.
        eval_every: Periods between forecast origins (1 = every day).
        n_samples: Draws per forecast.
        var_levels: Lower-tail probabilities for VaR quantiles.
        interval_levels: Central prediction-interval probabilities.
        start: Only use origins whose date is on or after this (Series input).
        baseline: Reference model for skill scores (default: the first model).
        random_state: Seed; each model gets an independent stream.

    Returns:
        :class:`WalkForwardResult`.
    """
    if isinstance(forecasters, Mapping):
        models = dict(forecasters)
    else:
        models = {}
        for fc in forecasters:
            if fc.name in models:
                raise ValueError(f"Duplicate forecaster name {fc.name!r}; pass a mapping")
            models[fc.name] = fc
    if not models:
        raise ValueError("No forecasters given")
    hs = validate_horizons(horizons)
    var_levels = tuple(float(a) for a in var_levels)
    interval_levels = tuple(float(lvl) for lvl in interval_levels)
    if refit_every < 1 or eval_every < 1:
        raise ValueError("refit_every and eval_every must be >= 1")

    if isinstance(returns, pd.Series):
        s = returns.dropna()
        r = s.to_numpy(dtype=float)
        labels: pd.Index = s.index
    else:
        r = np.asarray(returns, dtype=float).ravel()
        if not np.all(np.isfinite(r)):
            raise ValueError("returns contain non-finite values")
        labels = pd.RangeIndex(r.size)
    n = r.size
    if n < lookback + hs[0]:
        raise ValueError(f"Need at least lookback + {hs[0]} = {lookback + hs[0]} returns")

    origins = np.arange(lookback, n - hs[0] + 1, eval_every)
    if start is not None:
        if not isinstance(labels, pd.DatetimeIndex):
            raise ValueError("start requires returns with a DatetimeIndex")
        origins = origins[labels[origins - 1] >= pd.Timestamp(start)]

    cum = np.concatenate([[0.0], np.cumsum(r)])
    tails = sorted(set(var_levels) | {(1 - lvl) / 2 for lvl in interval_levels})
    tails += [1 - (1 - lvl) / 2 for lvl in interval_levels]
    q_index = {round(q, 12): i for i, q in enumerate(tails)}
    probs = np.asarray(tails)

    seeds = np.random.SeedSequence(random_state).spawn(len(models))
    rngs = {name: np.random.default_rng(seed) for name, seed in zip(models, seeds, strict=True)}
    last_fit: dict[str, int | None] = dict.fromkeys(models)
    failures: dict[str, int] = dict.fromkeys(models, 0)
    blocks: list[pd.DataFrame] = []

    for name, model in models.items():
        cols: dict[str, list[np.ndarray]] = {}
        for t in origins:
            window = r[t - lookback : t]
            lf = last_fit[name]
            if lf is None or t - lf >= refit_every:
                try:
                    model.fit(window)
                    last_fit[name] = int(t)
                except Exception as exc:  # noqa: BLE001 - keep evaluating other origins
                    failures[name] += 1
                    logger.warning(f"{name}: fit failed at origin {t} ({exc})")
                    if lf is None:
                        continue
            try:
                forecast = model.predict(window, hs, n_samples=n_samples, random_state=rngs[name])
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"{name}: predict failed at origin {t} ({exc})")
                continue

            avail = [j for j, h in enumerate(hs) if t + h <= n]
            h_av = np.asarray([hs[j] for j in avail])
            x = forecast.samples[:, avail].T  # (n_h, n_samples)
            y = cum[t + h_av] - cum[t]
            qs = np.quantile(x, probs, axis=1)  # (n_q, n_h)

            block = {
                "origin_idx": np.full(len(avail), t),
                "horizon": h_av,
                "y": y,
                "crps": crps_ensemble(x, y),
                "pit": np.mean(x <= y[:, None], axis=1),
                "mean": x.mean(axis=1),
                "std": x.std(axis=1),
            }
            for a in var_levels:
                qa = qs[q_index[round(a, 12)]]
                block[f"var_{_pct(a)}"] = qa
                block[f"hit_{_pct(a)}"] = y < qa
            for lvl in interval_levels:
                lo = qs[q_index[round((1 - lvl) / 2, 12)]]
                hi = qs[q_index[round(1 - (1 - lvl) / 2, 12)]]
                block[f"cov_{_pct(lvl)}"] = (y >= lo) & (y <= hi)
                block[f"width_{_pct(lvl)}"] = hi - lo
            for k, v in block.items():
                cols.setdefault(k, []).append(np.asarray(v))

        if not cols:
            logger.warning(f"{name}: produced no forecasts")
            continue
        df = pd.DataFrame({k: np.concatenate(v) for k, v in cols.items()})
        df.insert(0, "model", name)
        blocks.append(df)

    if not blocks:
        raise RuntimeError("No forecaster produced any forecasts")
    records = pd.concat(blocks, ignore_index=True)
    records.insert(0, "origin", labels[records["origin_idx"].to_numpy() - 1])
    return WalkForwardResult(
        records=records,
        horizons=hs,
        var_levels=var_levels,
        interval_levels=interval_levels,
        eval_every=eval_every,
        baseline=baseline if baseline is not None else next(iter(models)),
        fit_failures=failures,
    )
