"""Tests for signal weight tuning."""

from __future__ import annotations

import numpy as np
import pytest

from synthquant.analytics.distributions import EmpiricalDistribution
from synthquant.forecast import ForecastResult
from synthquant.strategy.signal_tuner import ForecastSnapshot, SignalTuner

# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_forecast(
    *,
    terminal_returns: np.ndarray | None = None,
    n_paths: int = 5000,
    horizon: int = 21,
    current_regime: int = 0,
    regime_probabilities: np.ndarray | None = None,
    regime_params: dict[int, dict[str, float]] | None = None,
    current_price: float = 100.0,
    rng_seed: int = 42,
) -> ForecastResult:
    """Build a ForecastResult with controlled properties."""
    rng = np.random.default_rng(rng_seed)

    if terminal_returns is None:
        daily_mu = 0.08 / 252
        daily_sigma = 0.18 / np.sqrt(252)
        terminal_returns = rng.normal(
            daily_mu * horizon,
            daily_sigma * np.sqrt(horizon),
            n_paths,
        )

    n_paths = len(terminal_returns)
    terminal_prices = current_price * np.exp(terminal_returns)
    paths = np.column_stack(
        [
            np.full(n_paths, current_price),
            terminal_prices,
        ]
    )

    if regime_probabilities is None:
        regime_probabilities = np.array([0.95, 0.05])
    if regime_params is None:
        regime_params = {
            0: {"mu": 0.08 / 252, "sigma": 0.18 / np.sqrt(252)},
            1: {"mu": -0.30 / 252, "sigma": 0.40 / np.sqrt(252)},
        }

    dist = EmpiricalDistribution()
    dist.fit(paths)

    var_95 = float(np.quantile(terminal_returns, 0.05))
    cvar_95 = float(np.mean(terminal_returns[terminal_returns <= var_95]))
    var_99 = float(np.quantile(terminal_returns, 0.01))
    cvar_99 = float(np.mean(terminal_returns[terminal_returns <= var_99]))

    max_drawdowns = rng.uniform(0.0, 0.05, n_paths)
    bands = {"5%": paths[:, -1] * 0.9, "95%": paths[:, -1] * 1.1}

    return ForecastResult(
        paths=paths,
        terminal_prices=terminal_prices,
        terminal_returns=terminal_returns,
        distribution=dist,
        var_95=var_95,
        cvar_95=cvar_95,
        var_99=var_99,
        cvar_99=cvar_99,
        max_drawdowns=max_drawdowns,
        confidence_bands=bands,
        current_regime=current_regime,
        regime_probabilities=regime_probabilities,
        regime_params=regime_params,
        horizon=horizon,
        current_price=current_price,
    )


def _make_snapshot(
    *,
    bullish: bool = True,
    actual_return: float | None = None,
    recent_trend: float = 0.001,
    rng_seed: int = 42,
) -> ForecastSnapshot:
    """Create a ForecastSnapshot with controlled properties."""
    rng = np.random.default_rng(rng_seed)

    if bullish:
        daily_mu = 0.12 / 252
        daily_sigma = 0.15 / np.sqrt(252)
        regime = 0
        regime_probs = np.array([0.95, 0.05])
        regime_params = {
            0: {"mu": daily_mu, "sigma": daily_sigma},
            1: {"mu": -0.40 / 252, "sigma": 0.40 / np.sqrt(252)},
        }
    else:
        daily_mu = -0.30 / 252
        daily_sigma = 0.40 / np.sqrt(252)
        regime = 1
        regime_probs = np.array([0.10, 0.90])
        regime_params = {
            0: {"mu": 0.10 / 252, "sigma": 0.18 / np.sqrt(252)},
            1: {"mu": daily_mu, "sigma": daily_sigma},
        }

    terminal_returns = rng.normal(
        daily_mu * 21,
        daily_sigma * np.sqrt(21),
        5000,
    )

    forecast = _make_forecast(
        terminal_returns=terminal_returns,
        current_regime=regime,
        regime_probabilities=regime_probs,
        regime_params=regime_params,
        rng_seed=rng_seed,
    )

    recent_returns = np.full(63, recent_trend) + rng.normal(0, 0.002, 63)

    if actual_return is None:
        actual_return = 0.03 if bullish else -0.05

    return ForecastSnapshot(
        forecast=forecast,
        recent_returns=recent_returns,
        actual_forward_return=actual_return,
        index=0,
    )


# ── ForecastSnapshot ─────────────────────────────────────────────────────────


class TestForecastSnapshot:
    def test_attributes(self) -> None:
        snap = _make_snapshot()
        assert isinstance(snap.forecast, ForecastResult)
        assert isinstance(snap.recent_returns, np.ndarray)
        assert isinstance(snap.actual_forward_return, float)
        assert isinstance(snap.index, int)

    def test_bullish_and_bearish(self) -> None:
        bull = _make_snapshot(bullish=True)
        bear = _make_snapshot(bullish=False)
        assert bull.forecast.current_regime == 0
        assert bear.forecast.current_regime == 1


# ── Parameter conversion ─────────────────────────────────────────────────────


class TestParamConversion:
    def test_vector_normalises_weights(self) -> None:
        tuner = SignalTuner()
        x = np.array([0.5, 0.1, 0.2, 0.1, 0.05, 0.05, 0.15, 21, 63])
        params = tuner._vector_to_params(x)
        total = sum(params[f"w_{n}"] for n in tuner.WEIGHT_NAMES)
        assert abs(total - 1.0) < 1e-10

    def test_momentum_long_ge_short(self) -> None:
        tuner = SignalTuner()
        # Intentionally set long < short in the vector
        x = np.array([0.2, 0.2, 0.2, 0.2, 0.1, 0.1, 0.15, 30, 10])
        params = tuner._vector_to_params(x)
        assert params["momentum_long"] >= params["momentum_short"]

    def test_default_params_sum_to_one(self) -> None:
        params = SignalTuner._default_params()
        total = sum(params[f"w_{n}"] for n in SignalTuner.WEIGHT_NAMES)
        assert abs(total - 1.0) < 1e-10

    def test_default_params_threshold(self) -> None:
        params = SignalTuner._default_params()
        assert params["conviction_threshold"] == 0.15
        assert params["momentum_short"] == 21.0
        assert params["momentum_long"] == 63.0


# ── Evaluate ──────────────────────────────────────────────────────────────────


class TestEvaluate:
    def test_empty_snapshots(self) -> None:
        tuner = SignalTuner()
        m = tuner.evaluate([])
        assert m["sharpe"] == 0.0
        assert m["hit_rate"] == 0.5
        assert m["n_trades"] == 0

    def test_all_bullish_positive_pnl(self) -> None:
        """All bullish forecasts + positive actual returns → positive PnL."""
        snaps = [
            _make_snapshot(
                bullish=True,
                actual_return=0.03,
                recent_trend=0.001,
                rng_seed=i,
            )
            for i in range(10)
        ]
        tuner = SignalTuner()
        m = tuner.evaluate(snaps)
        assert m["n_long"] > 0
        assert m["total_pnl"] > 0

    def test_metrics_keys_complete(self) -> None:
        snaps = [
            _make_snapshot(bullish=True, actual_return=0.02, rng_seed=1),
            _make_snapshot(bullish=False, actual_return=-0.04, rng_seed=2),
            _make_snapshot(bullish=True, actual_return=-0.01, rng_seed=3),
        ]
        tuner = SignalTuner()
        m = tuner.evaluate(snaps)
        expected_keys = {
            "sharpe",
            "hit_rate",
            "info_ratio",
            "n_long",
            "n_short",
            "n_flat",
            "n_trades",
            "total_pnl",
            "avg_ret_long",
            "avg_ret_flat",
        }
        assert set(m.keys()) == expected_keys

    def test_custom_params_change_result(self) -> None:
        """Different params produce different signal counts."""
        snaps = [_make_snapshot(bullish=True, actual_return=0.02, rng_seed=i) for i in range(8)]
        tuner = SignalTuner()
        m_default = tuner.evaluate(snaps)
        # All weight on momentum, very low threshold
        m_custom = tuner.evaluate(
            snaps,
            params={
                "w_regime": 0.01,
                "w_skew": 0.01,
                "w_prob_profit": 0.01,
                "w_tail_risk": 0.01,
                "w_vol": 0.01,
                "w_momentum": 0.95,
                "conviction_threshold": 0.05,
                "momentum_short": 10.0,
                "momentum_long": 30.0,
            },
        )
        assert m_default != m_custom

    def test_signal_count_sums_to_total(self) -> None:
        snaps = [_make_snapshot(bullish=True, rng_seed=i) for i in range(6)]
        tuner = SignalTuner()
        m = tuner.evaluate(snaps)
        assert m["n_long"] + m["n_short"] + m["n_flat"] == len(snaps)

    def test_high_threshold_all_flat(self) -> None:
        """Very high conviction threshold should produce all flat signals."""
        snaps = [_make_snapshot(bullish=True, rng_seed=i) for i in range(5)]
        tuner = SignalTuner()
        params = SignalTuner._default_params()
        params["conviction_threshold"] = 0.99
        m = tuner.evaluate(snaps, params)
        assert m["n_long"] == 0
        assert m["n_flat"] == 5


# ── Tune ──────────────────────────────────────────────────────────────────────


class TestTune:
    def test_tune_returns_expected_keys(self) -> None:
        snaps = [_make_snapshot(bullish=True, actual_return=0.03, rng_seed=i) for i in range(15)]
        tuner = SignalTuner()
        result = tuner.tune(snaps, metric="sharpe", max_iter=5)
        assert "params" in result
        assert "metrics" in result
        assert "default_metrics" in result
        assert all(f"w_{n}" in result["params"] for n in SignalTuner.WEIGHT_NAMES)

    def test_tuned_weights_sum_to_one(self) -> None:
        snaps = [_make_snapshot(bullish=True, actual_return=0.03, rng_seed=i) for i in range(15)]
        tuner = SignalTuner()
        result = tuner.tune(snaps, metric="sharpe", max_iter=5)
        total = sum(result["params"][f"w_{n}"] for n in SignalTuner.WEIGHT_NAMES)
        assert abs(total - 1.0) < 1e-8

    def test_tune_momentum_data(self) -> None:
        """When momentum is the useful predictor, its weight should be non-trivial."""
        snaps = []
        for i in range(20):
            # Even indices: positive momentum + positive actual
            # Odd indices: negative momentum + negative actual
            if i % 2 == 0:
                s = _make_snapshot(
                    bullish=True,
                    actual_return=0.04,
                    recent_trend=0.003,
                    rng_seed=i,
                )
            else:
                s = _make_snapshot(
                    bullish=True,
                    actual_return=-0.03,
                    recent_trend=-0.003,
                    rng_seed=i,
                )
            snaps.append(s)

        tuner = SignalTuner()
        result = tuner.tune(snaps, metric="sharpe", max_iter=15)
        # Momentum should get a meaningful weight
        assert result["params"]["w_momentum"] > 0.05

    def test_tune_all_metrics(self) -> None:
        """All three metrics can be optimised without error."""
        snaps = [_make_snapshot(bullish=True, actual_return=0.02, rng_seed=i) for i in range(10)]
        tuner = SignalTuner()
        for metric in ["sharpe", "hit_rate", "info_ratio"]:
            result = tuner.tune(snaps, metric=metric, max_iter=3)
            assert metric in result["metrics"]


# ── Train/test split ─────────────────────────────────────────────────────────


class TestTrainTestSplit:
    def test_split_produces_separate_metrics(self) -> None:
        train = [_make_snapshot(bullish=True, actual_return=0.03, rng_seed=i) for i in range(10)]
        test = [
            _make_snapshot(bullish=True, actual_return=0.02, rng_seed=i + 100) for i in range(5)
        ]
        tuner = SignalTuner()
        m_train = tuner.evaluate(train)
        m_test = tuner.evaluate(test)
        # Both should have valid structure
        assert m_train["n_long"] + m_train["n_flat"] == 10
        assert m_test["n_long"] + m_test["n_flat"] == 5

    def test_tuned_params_on_test_set(self) -> None:
        """Tuned params from train can be evaluated on test without error."""
        train = [_make_snapshot(bullish=True, actual_return=0.03, rng_seed=i) for i in range(15)]
        test = [
            _make_snapshot(bullish=False, actual_return=-0.04, rng_seed=i + 50) for i in range(5)
        ]
        tuner = SignalTuner()
        result = tuner.tune(train, max_iter=5)
        m_test = tuner.evaluate(test, result["params"])
        assert "sharpe" in m_test
        assert m_test["n_long"] + m_test["n_short"] + m_test["n_flat"] == 5


# ── Regularisation ────────────────────────────────────────────────────────────


class TestRegularisation:
    def test_regularisation_constrains_weights(self) -> None:
        """Higher regularisation keeps weights closer to defaults."""
        snaps = [_make_snapshot(bullish=True, actual_return=0.03, rng_seed=i) for i in range(15)]
        tuner = SignalTuner()
        default = SignalTuner._default_params()

        r_none = tuner.tune(snaps, max_iter=10, regularization=0.0)
        r_high = tuner.tune(snaps, max_iter=10, regularization=2.0)

        # Compute L2 distance from default weights
        def _dist(params):
            return sum(
                (params[f"w_{n}"] - default[f"w_{n}"]) ** 2 for n in SignalTuner.WEIGHT_NAMES
            )

        assert _dist(r_high["params"]) <= _dist(r_none["params"]) + 0.01

    def test_regularisation_returns_valid_params(self) -> None:
        snaps = [_make_snapshot(bullish=True, actual_return=0.02, rng_seed=i) for i in range(10)]
        tuner = SignalTuner()
        result = tuner.tune(snaps, max_iter=5, regularization=1.0)
        total = sum(result["params"][f"w_{n}"] for n in SignalTuner.WEIGHT_NAMES)
        assert abs(total - 1.0) < 1e-8


# ── Weights-only mode ─────────────────────────────────────────────────────────


class TestWeightsOnly:
    def test_fixed_params_unchanged(self) -> None:
        """Non-weight parameters stay at defaults."""
        snaps = [_make_snapshot(bullish=True, actual_return=0.03, rng_seed=i) for i in range(15)]
        tuner = SignalTuner()
        default = SignalTuner._default_params()

        result = tuner.tune(snaps, max_iter=10, tune_weights_only=True)
        p = result["params"]

        assert p["conviction_threshold"] == default["conviction_threshold"]
        assert p["momentum_short"] == default["momentum_short"]
        assert p["momentum_long"] == default["momentum_long"]

    def test_weights_sum_to_one(self) -> None:
        snaps = [_make_snapshot(bullish=True, actual_return=0.02, rng_seed=i) for i in range(10)]
        tuner = SignalTuner()
        result = tuner.tune(snaps, max_iter=5, tune_weights_only=True)
        total = sum(result["params"][f"w_{n}"] for n in SignalTuner.WEIGHT_NAMES)
        assert abs(total - 1.0) < 1e-8

    def test_weights_only_with_regularisation(self) -> None:
        """Both options can be combined."""
        snaps = [_make_snapshot(bullish=True, actual_return=0.03, rng_seed=i) for i in range(12)]
        tuner = SignalTuner()
        result = tuner.tune(
            snaps,
            max_iter=5,
            tune_weights_only=True,
            regularization=0.5,
        )
        assert "params" in result
        assert "metrics" in result


# ── Cross-validation ──────────────────────────────────────────────────────────


class TestCrossValidate:
    def test_returns_expected_structure(self) -> None:
        snaps = [
            _make_snapshot(
                bullish=(i % 2 == 0),
                actual_return=0.02 if i % 2 == 0 else -0.03,
                rng_seed=i,
            )
            for i in range(20)
        ]
        tuner = SignalTuner()
        result = tuner.cross_validate(
            snaps,
            n_folds=4,
            max_iter=3,
        )
        assert "fold_results" in result
        assert "mean_oos_tuned" in result
        assert "mean_oos_default" in result
        assert len(result["fold_results"]) == 3  # n_folds - 1

    def test_fold_sizes_expanding(self) -> None:
        """Training sets should grow with each fold."""
        snaps = [_make_snapshot(bullish=True, actual_return=0.02, rng_seed=i) for i in range(20)]
        tuner = SignalTuner()
        result = tuner.cross_validate(snaps, n_folds=4, max_iter=3)
        train_sizes = [fr["n_train"] for fr in result["fold_results"]]
        assert train_sizes == sorted(train_sizes)
        assert train_sizes[0] < train_sizes[-1]

    def test_too_few_snapshots_raises(self) -> None:
        snaps = [_make_snapshot(bullish=True, rng_seed=i) for i in range(4)]
        tuner = SignalTuner()
        with pytest.raises(ValueError, match="Too few snapshots"):
            tuner.cross_validate(snaps, n_folds=5, max_iter=3)

    def test_mean_oos_metrics_are_averages(self) -> None:
        snaps = [
            _make_snapshot(
                bullish=(i % 3 != 0),
                actual_return=0.02 if i % 3 != 0 else -0.04,
                rng_seed=i,
            )
            for i in range(15)
        ]
        tuner = SignalTuner()
        result = tuner.cross_validate(snaps, n_folds=3, max_iter=3)
        # Check that mean is actually the average of per-fold values
        per_fold_sharpes = [fr["oos_tuned"]["sharpe"] for fr in result["fold_results"]]
        expected_mean = float(np.mean(per_fold_sharpes))
        assert abs(result["mean_oos_tuned"]["sharpe"] - expected_mean) < 1e-10

    def test_progress_callback_called(self) -> None:
        snaps = [_make_snapshot(bullish=True, actual_return=0.02, rng_seed=i) for i in range(15)]
        calls = []
        tuner = SignalTuner()
        tuner.cross_validate(
            snaps,
            n_folds=3,
            max_iter=3,
            progress_fn=lambda f, t, r: calls.append((f, t)),
        )
        assert len(calls) == 2  # n_folds - 1


# ── Multi-ticker preparation ─────────────────────────────────────────────────


class TestPrepareMulti:
    def test_combines_snapshots(self) -> None:
        """Pooling multiple datasets produces more snapshots than one."""
        rng = np.random.default_rng(42)
        n = 600
        r1 = rng.normal(0.0003, 0.01, n)
        r2 = rng.normal(-0.0001, 0.015, n)

        tuner = SignalTuner(n_paths=500, lookback=504, horizon=21, step=21)
        single = tuner.prepare(r1)
        multi = tuner.prepare_multi({"A": (r1, None), "B": (r2, None)})
        assert len(multi) > len(single)

    def test_progress_callback(self) -> None:
        rng = np.random.default_rng(99)
        n = 550
        r1 = rng.normal(0.0003, 0.01, n)
        r2 = rng.normal(0.0001, 0.012, n)

        calls = []
        tuner = SignalTuner(n_paths=500, lookback=504, horizon=21, step=21)
        tuner.prepare_multi(
            {"X": (r1, None), "Y": (r2, None)},
            progress_fn=lambda t, c, tot: calls.append(t),
        )
        assert calls == ["X", "Y"]


# ── Prepare (integration) ────────────────────────────────────────────────────


class TestPrepareValidation:
    def test_too_few_returns_raises(self) -> None:
        tuner = SignalTuner(lookback=504, horizon=21)
        with pytest.raises(ValueError, match="Need at least"):
            tuner.prepare(np.zeros(100))
