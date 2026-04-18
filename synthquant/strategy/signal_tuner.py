"""Signal weight tuning via walk-forward optimisation.

Pre-computes walk-forward forecasts, then uses differential evolution
to find the signal weights that maximise a chosen quality metric on
the training set.  Evaluation on a held-out test set guards against
over-fitting.
"""

from __future__ import annotations

import contextlib
import io
import logging
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import differential_evolution
from scipy.stats import spearmanr

from synthquant.ensemble_forecast import EnsembleForecaster
from synthquant.forecast import ForecastResult
from synthquant.strategy.forecast_signals import ForecastSignal

logger = logging.getLogger(__name__)

__all__ = ["ForecastSnapshot", "SignalTuner"]


@dataclass
class ForecastSnapshot:
    """A forecast paired with the actual forward outcome.

    Attributes:
        forecast: The ForecastResult generated at this point in time.
        recent_returns: Recent daily log-returns up to forecast time
            (for the momentum component).
        actual_forward_return: Realised cumulative log-return over the
            forecast horizon.
        index: Index in the return array where the forecast was made.
    """

    forecast: ForecastResult
    recent_returns: np.ndarray
    actual_forward_return: float
    index: int


class SignalTuner:
    """Optimises ForecastSignal parameters on historical data.

    Workflow:
        1. :meth:`prepare` — walk-forward forecasting (slow, run once).
        2. :meth:`tune` — optimise signal weights on pre-computed data (fast).
        3. :meth:`evaluate` — score any parameter set on any snapshot subset.

    Args:
        n_paths: Monte-Carlo paths per forecast.
        horizon: Forecast horizon in trading days.
        lookback: Days of history used to fit each forecast.
        step: Trading days between consecutive forecasts.
        random_state: Seed for the forecaster and optimiser.
    """

    WEIGHT_NAMES = ["regime", "skew", "prob_profit", "tail_risk", "vol", "momentum"]

    def __init__(
        self,
        n_paths: int = 5000,
        horizon: int = 21,
        lookback: int = 504,
        step: int = 21,
        random_state: int = 42,
    ) -> None:
        self.n_paths = n_paths
        self.horizon = horizon
        self.lookback = lookback
        self.step = step
        self.random_state = random_state

    # ------------------------------------------------------------------
    # Walk-forward data preparation
    # ------------------------------------------------------------------

    def prepare(
        self,
        returns: np.ndarray,
        prices: np.ndarray | None = None,
        progress_fn: Any | None = None,
    ) -> list[ForecastSnapshot]:
        """Walk-forward: fit EnsembleForecaster, record outcomes.

        Args:
            returns: Daily log-returns array.
            prices: Corresponding price array (length = len(returns) + 1).
                Reconstructed from returns if *None*.
            progress_fn: Optional ``callable(current, total, info_str)``
                called after each window for progress reporting.

        Returns:
            List of :class:`ForecastSnapshot`, one per walk-forward window.
        """
        returns = np.asarray(returns, dtype=float)

        if prices is None:
            prices = 100.0 * np.exp(np.concatenate([[0.0], np.cumsum(returns)]))

        n = len(returns)
        if n < self.lookback + self.horizon:
            raise ValueError(f"Need at least {self.lookback + self.horizon} returns, got {n}")

        snapshots: list[ForecastSnapshot] = []
        positions = list(range(self.lookback, n - self.horizon, self.step))
        total = len(positions)

        for i, t in enumerate(positions):
            train_rets = returns[t - self.lookback : t]
            current_price = float(prices[t])
            actual_fwd = float(np.sum(returns[t : t + self.horizon]))
            recent = returns[max(0, t - 63) : t].copy()

            if progress_fn is not None:
                progress_fn(i + 1, total, f"t={t} price={current_price:.2f}")

            try:
                with warnings.catch_warnings(), contextlib.redirect_stderr(io.StringIO()):
                    warnings.simplefilter("ignore")
                    fc = EnsembleForecaster(
                        n_paths=self.n_paths,
                        random_state=self.random_state,
                    )
                    fc.fit(train_rets)
                    forecast = fc.forecast(
                        horizon=self.horizon,
                        current_price=current_price,
                    )

                snapshots.append(
                    ForecastSnapshot(
                        forecast=forecast,
                        recent_returns=recent,
                        actual_forward_return=actual_fwd,
                        index=t,
                    )
                )
            except Exception:
                logger.warning(
                    "prepare: failed at t=%d, skipping",
                    t,
                    exc_info=True,
                )

        return snapshots

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        snapshots: list[ForecastSnapshot],
        params: dict[str, float] | None = None,
    ) -> dict[str, float]:
        """Score signal quality on pre-computed snapshots.

        Args:
            snapshots: ForecastSnapshot list (from :meth:`prepare` or manual).
            params: Parameter dict with keys ``w_<component>``,
                ``conviction_threshold``, ``momentum_short``,
                ``momentum_long``.  Uses defaults when *None*.

        Returns:
            Dict with keys: sharpe, hit_rate, info_ratio, n_long, n_short,
            n_flat, n_trades, total_pnl, avg_ret_long, avg_ret_flat.
        """
        empty: dict[str, float] = {
            "sharpe": 0.0,
            "hit_rate": 0.5,
            "info_ratio": 0.0,
            "n_long": 0,
            "n_short": 0,
            "n_flat": 0,
            "n_trades": 0,
            "total_pnl": 0.0,
            "avg_ret_long": 0.0,
            "avg_ret_flat": 0.0,
        }
        if not snapshots:
            return empty

        params = params if params is not None else self._default_params()

        weights = {name: params[f"w_{name}"] for name in self.WEIGHT_NAMES}
        mom_short = max(2, int(round(params["momentum_short"])))
        mom_long = max(mom_short, int(round(params["momentum_long"])))

        fs = ForecastSignal(
            weights=weights,
            conviction_threshold=params["conviction_threshold"],
            momentum_short=mom_short,
            momentum_long=mom_long,
        )

        signals = np.empty(len(snapshots))
        convictions = np.empty(len(snapshots))
        actual = np.empty(len(snapshots))

        for i, snap in enumerate(snapshots):
            result = fs.generate(
                snap.forecast,
                recent_returns=snap.recent_returns,
            )
            signals[i] = result.signal
            convictions[i] = result.conviction
            actual[i] = snap.actual_forward_return

        # PnL: signal * conviction * actual return
        pnl = signals * convictions * actual

        # Sharpe (annualised)
        periods_per_year = 252 / self.horizon
        if len(pnl) > 1 and np.std(pnl) > 1e-12:
            sharpe = float(np.mean(pnl) / np.std(pnl) * np.sqrt(periods_per_year))
        else:
            sharpe = 0.0

        # Hit rate (active signals only)
        active = signals != 0
        if active.sum() > 0:
            correct = ((signals[active] > 0) & (actual[active] > 0)) | (
                (signals[active] < 0) & (actual[active] < 0)
            )
            hit_rate = float(correct.mean())
        else:
            hit_rate = 0.5

        # Information ratio (Spearman rank correlation)
        scored = signals * convictions
        if len(scored) > 2 and np.std(scored) > 1e-12 and np.std(actual) > 1e-12:
            info_ratio = float(spearmanr(scored, actual).statistic)
        else:
            info_ratio = 0.0

        n_long = int((signals > 0).sum())
        n_short = int((signals < 0).sum())
        n_flat = int((signals == 0).sum())
        avg_ret_long = float(np.mean(actual[signals > 0])) if n_long > 0 else 0.0
        avg_ret_flat = float(np.mean(actual[signals == 0])) if n_flat > 0 else 0.0

        return {
            "sharpe": sharpe,
            "hit_rate": hit_rate,
            "info_ratio": info_ratio,
            "n_long": n_long,
            "n_short": n_short,
            "n_flat": n_flat,
            "n_trades": n_long + n_short,
            "total_pnl": float(pnl.sum()),
            "avg_ret_long": avg_ret_long,
            "avg_ret_flat": avg_ret_flat,
        }

    # ------------------------------------------------------------------
    # Optimisation
    # ------------------------------------------------------------------

    def tune(
        self,
        snapshots: list[ForecastSnapshot],
        metric: str = "sharpe",
        max_iter: int = 80,
        seed: int | None = None,
        progress_fn: Any | None = None,
        regularization: float = 0.0,
        tune_weights_only: bool = False,
    ) -> dict[str, Any]:
        """Optimise signal parameters via differential evolution.

        Args:
            snapshots: Training snapshots.
            metric: Metric to maximise (``"sharpe"``, ``"hit_rate"``,
                or ``"info_ratio"``).
            max_iter: Maximum DE generations.
            seed: Random seed for reproducibility.
            progress_fn: Optional ``callable(generation, best_value)``
                called after each generation for progress reporting.
            regularization: L2 penalty strength on deviation from default
                weights.  Higher values keep weights closer to defaults,
                reducing overfitting at the cost of optimality.
            tune_weights_only: If *True*, only optimise the 6 component
                weights; conviction threshold and momentum windows are
                fixed at their defaults.

        Returns:
            Dict with ``params`` (best parameter dict), ``metrics``
            (best training metrics), ``default_metrics`` (default params
            on the same data), and ``optimizer_result``.
        """
        if seed is None:
            seed = self.random_state

        default = self._default_params()

        if tune_weights_only:
            bounds = [(0.01, 1.0)] * 6

            def _to_params(x: np.ndarray) -> dict[str, float]:
                raw = np.asarray(x[:6], dtype=float)
                total = raw.sum()
                if total <= 0:
                    total = 1.0
                normed = raw / total
                p = {f"w_{name}": float(normed[i]) for i, name in enumerate(self.WEIGHT_NAMES)}
                p["conviction_threshold"] = default["conviction_threshold"]
                p["momentum_short"] = default["momentum_short"]
                p["momentum_long"] = default["momentum_long"]
                return p
        else:
            bounds = [
                (0.01, 1.0),  # w_regime
                (0.01, 1.0),  # w_skew
                (0.01, 1.0),  # w_prob_profit
                (0.01, 1.0),  # w_tail_risk
                (0.01, 1.0),  # w_vol
                (0.01, 1.0),  # w_momentum
                (0.05, 0.40),  # conviction_threshold
                (5, 42),  # momentum_short
                (21, 126),  # momentum_long
            ]
            _to_params = self._vector_to_params

        # Cache raw metric values keyed by parameter vector bytes
        _eval_cache: dict[bytes, float] = {}

        def objective(x: np.ndarray) -> float:
            p = _to_params(x)
            m = self.evaluate(snapshots, p)
            val = m[metric]
            _eval_cache[x.tobytes()] = val  # cache raw metric
            if regularization > 0:
                pen = sum(
                    (p[f"w_{name}"] - default[f"w_{name}"]) ** 2 for name in self.WEIGHT_NAMES
                )
                if not tune_weights_only:
                    pen += (
                        (p["conviction_threshold"] - default["conviction_threshold"]) / 0.35
                    ) ** 2
                    pen += ((p["momentum_short"] - default["momentum_short"]) / 37.0) ** 2
                    pen += ((p["momentum_long"] - default["momentum_long"]) / 105.0) ** 2
                val -= regularization * pen
            return -val

        gen_count = [0]

        def _callback(xk: np.ndarray, convergence: float = 0.0) -> None:
            gen_count[0] += 1
            if progress_fn is not None:
                # Look up cached raw metric (already evaluated during generation)
                best_val = _eval_cache.get(xk.tobytes())
                if best_val is None:
                    # Fallback: shouldn't happen but evaluate if missing
                    best_val = self.evaluate(snapshots, _to_params(xk))[metric]
                progress_fn(gen_count[0], best_val)
            # Trim cache periodically to avoid memory bloat
            if len(_eval_cache) > 500:
                _eval_cache.clear()

        result = differential_evolution(
            objective,
            bounds=bounds,
            maxiter=max_iter,
            seed=seed,
            popsize=10,
            tol=1e-3,
            polish=False,
            disp=False,
            callback=_callback,
        )

        best_params = _to_params(result.x)
        best_metrics = self.evaluate(snapshots, best_params)
        default_metrics = self.evaluate(snapshots)

        return {
            "params": best_params,
            "metrics": best_metrics,
            "default_metrics": default_metrics,
            "optimizer_result": result,
        }

    # ------------------------------------------------------------------
    # Cross-validation
    # ------------------------------------------------------------------

    def cross_validate(
        self,
        snapshots: list[ForecastSnapshot],
        n_folds: int = 5,
        metric: str = "sharpe",
        max_iter: int = 80,
        regularization: float = 0.0,
        tune_weights_only: bool = False,
        seed: int | None = None,
        progress_fn: Any | None = None,
    ) -> dict[str, Any]:
        """Temporal expanding-window cross-validation.

        Divides *snapshots* into *n_folds* consecutive blocks.  For each
        fold *k* (0 … n_folds − 2), the first *k + 1* blocks form the
        training set and the next block is used for out-of-sample testing.

        Args:
            snapshots: Full snapshot list in chronological order.
            n_folds: Number of temporal blocks.
            metric: Metric to maximise in each fold.
            max_iter: DE generations per fold.
            regularization: L2 penalty strength (passed to :meth:`tune`).
            tune_weights_only: Only tune the 6 weights (passed to
                :meth:`tune`).
            seed: Random seed.
            progress_fn: Optional ``callable(fold, total_folds,
                fold_result)`` called after each fold completes.

        Returns:
            Dict with ``fold_results`` (list of per-fold dicts),
            ``mean_oos_tuned`` and ``mean_oos_default`` (averaged OOS
            metrics across folds).
        """
        n = len(snapshots)
        fold_size = n // n_folds
        if fold_size < 3:
            raise ValueError(
                f"Too few snapshots ({n}) for {n_folds} folds (need at least {n_folds * 3})"
            )

        fold_results: list[dict[str, Any]] = []
        total_folds = n_folds - 1

        for fold in range(total_folds):
            train_end = (fold + 1) * fold_size
            test_end = min((fold + 2) * fold_size, n)
            train = snapshots[:train_end]
            test = snapshots[train_end:test_end]

            result = self.tune(
                train,
                metric=metric,
                max_iter=max_iter,
                seed=seed,
                regularization=regularization,
                tune_weights_only=tune_weights_only,
            )

            oos_default = self.evaluate(test)
            oos_tuned = self.evaluate(test, result["params"])

            fold_result = {
                "fold": fold,
                "n_train": len(train),
                "n_test": len(test),
                "tuned_params": result["params"],
                "train_metrics": result["metrics"],
                "default_train_metrics": result["default_metrics"],
                "oos_default": oos_default,
                "oos_tuned": oos_tuned,
            }
            fold_results.append(fold_result)

            if progress_fn is not None:
                progress_fn(fold + 1, total_folds, fold_result)

        # Average OOS metrics across folds
        mean_oos_tuned: dict[str, float] = {}
        mean_oos_default: dict[str, float] = {}
        for key in fold_results[0]["oos_tuned"]:
            mean_oos_tuned[key] = float(np.mean([fr["oos_tuned"][key] for fr in fold_results]))
            mean_oos_default[key] = float(np.mean([fr["oos_default"][key] for fr in fold_results]))

        return {
            "fold_results": fold_results,
            "mean_oos_tuned": mean_oos_tuned,
            "mean_oos_default": mean_oos_default,
        }

    # ------------------------------------------------------------------
    # Multi-ticker preparation
    # ------------------------------------------------------------------

    def prepare_multi(
        self,
        datasets: dict[str, tuple[np.ndarray, np.ndarray | None]],
        progress_fn: Any | None = None,
    ) -> list[ForecastSnapshot]:
        """Walk-forward over multiple tickers and pool snapshots.

        This increases the training set size, which helps the optimiser
        find robust weights across diverse market conditions.

        Args:
            datasets: Mapping of ticker name to ``(returns, prices)``
                tuple.  *prices* may be *None* (reconstructed from
                returns).
            progress_fn: Optional ``callable(ticker, current, total)``
                called when starting each ticker.

        Returns:
            Combined list of :class:`ForecastSnapshot` from all tickers.
        """
        all_snaps: list[ForecastSnapshot] = []
        tickers = list(datasets.keys())

        for i, ticker in enumerate(tickers):
            if progress_fn is not None:
                progress_fn(ticker, i + 1, len(tickers))
            returns, prices = datasets[ticker]
            snaps = self.prepare(np.asarray(returns, dtype=float), prices)
            all_snaps.extend(snaps)

        return all_snaps

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _vector_to_params(self, x: np.ndarray) -> dict[str, float]:
        """Convert optimiser vector to parameter dict (weights normalised)."""
        raw = np.asarray(x[:6], dtype=float)
        total = raw.sum()
        if total <= 0:
            total = 1.0
        normed = raw / total

        mom_short = max(2, int(round(x[7])))
        mom_long = max(mom_short, int(round(x[8])))

        return {
            "w_regime": float(normed[0]),
            "w_skew": float(normed[1]),
            "w_prob_profit": float(normed[2]),
            "w_tail_risk": float(normed[3]),
            "w_vol": float(normed[4]),
            "w_momentum": float(normed[5]),
            "conviction_threshold": float(x[6]),
            "momentum_short": float(mom_short),
            "momentum_long": float(mom_long),
        }

    @staticmethod
    def _default_params() -> dict[str, float]:
        """Default parameters matching ForecastSignal defaults."""
        return {
            "w_regime": 0.25,
            "w_skew": 0.10,
            "w_prob_profit": 0.20,
            "w_tail_risk": 0.15,
            "w_vol": 0.10,
            "w_momentum": 0.20,
            "conviction_threshold": 0.15,
            "momentum_short": 21.0,
            "momentum_long": 63.0,
        }
