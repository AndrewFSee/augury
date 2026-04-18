"""Tests for forecast-driven trading signals."""

from __future__ import annotations

import numpy as np
import pytest

from synthquant.analytics.distributions import EmpiricalDistribution
from synthquant.forecast import ForecastResult
from synthquant.strategy.forecast_signals import ForecastSignal, SignalResult

# ── Helper to build ForecastResult with controlled properties ─────────────────


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
    """Build a ForecastResult with sensible defaults, optionally overriding."""
    rng = np.random.default_rng(rng_seed)

    if terminal_returns is None:
        # Bullish default: 8%/yr drift, 18% vol, 21 day horizon
        daily_mu = 0.08 / 252
        daily_sigma = 0.18 / np.sqrt(252)
        terminal_returns = rng.normal(daily_mu * horizon, daily_sigma * np.sqrt(horizon), n_paths)

    n_paths = len(terminal_returns)
    terminal_prices = current_price * np.exp(terminal_returns)

    # Minimal paths array (just start and end)
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
    bands = {
        "5%": paths[:, -1] * 0.9,
        "95%": paths[:, -1] * 1.1,
    }

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


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture()
def bullish_forecast() -> ForecastResult:
    """Calm regime, positive drift, low vol."""
    rng = np.random.default_rng(42)
    # Clear bullish: 12%/yr mu, 15% vol, slight positive skew
    daily_mu = 0.12 / 252
    daily_sigma = 0.15 / np.sqrt(252)
    base = rng.normal(daily_mu * 21, daily_sigma * np.sqrt(21), 5000)
    # Add slight positive skew via a few extra positive outliers
    base[:50] += rng.exponential(0.02, 50)
    return _make_forecast(
        terminal_returns=base,
        regime_params={
            0: {"mu": daily_mu, "sigma": daily_sigma},
            1: {"mu": -0.40 / 252, "sigma": 0.40 / np.sqrt(252)},
        },
        regime_probabilities=np.array([0.97, 0.03]),
    )


@pytest.fixture()
def bearish_forecast() -> ForecastResult:
    """Crisis regime, negative drift, high vol, negative skew."""
    rng = np.random.default_rng(42)
    daily_mu = -0.30 / 252
    daily_sigma = 0.40 / np.sqrt(252)
    base = rng.normal(daily_mu * 21, daily_sigma * np.sqrt(21), 5000)
    # Add negative skew
    base[:100] -= rng.exponential(0.04, 100)
    return _make_forecast(
        terminal_returns=base,
        current_regime=1,
        regime_params={
            0: {"mu": 0.10 / 252, "sigma": 0.18 / np.sqrt(252)},
            1: {"mu": daily_mu, "sigma": daily_sigma},
        },
        regime_probabilities=np.array([0.10, 0.90]),
    )


@pytest.fixture()
def ambiguous_forecast() -> ForecastResult:
    """Calm regime but near-zero drift, moderate vol — should be flat."""
    rng = np.random.default_rng(42)
    daily_mu = 0.01 / 252  # ~1%/yr drift — very weak
    daily_sigma = 0.22 / np.sqrt(252)
    base = rng.normal(daily_mu * 21, daily_sigma * np.sqrt(21), 5000)
    return _make_forecast(
        terminal_returns=base,
        regime_params={
            0: {"mu": daily_mu, "sigma": daily_sigma},
            1: {"mu": -0.10 / 252, "sigma": 0.35 / np.sqrt(252)},
        },
        regime_probabilities=np.array([0.65, 0.35]),
    )


# ── SignalResult properties ───────────────────────────────────────────────────


class TestSignalResult:
    def test_is_long(self) -> None:
        sr = SignalResult(signal=1, conviction=0.5, regime=0, regime_confidence=0.95, components={})
        assert sr.is_long
        assert not sr.is_short
        assert not sr.is_flat

    def test_is_short(self) -> None:
        sr = SignalResult(
            signal=-1, conviction=0.5, regime=1, regime_confidence=0.90, components={}
        )
        assert sr.is_short
        assert not sr.is_long

    def test_is_flat(self) -> None:
        sr = SignalResult(
            signal=0, conviction=0.08, regime=0, regime_confidence=0.60, components={}
        )
        assert sr.is_flat

    def test_summary_has_all_keys(self) -> None:
        sr = SignalResult(
            signal=1,
            conviction=0.6,
            regime=0,
            regime_confidence=0.95,
            components={
                "regime": 0.8,
                "skew": 0.1,
                "prob_profit": 0.3,
                "tail_risk": 0.2,
                "vol": 0.4,
                "momentum": 0.5,
            },
        )
        s = sr.summary()
        assert s["signal"] == 1
        assert "conviction" in s
        assert "regime" in s and "skew" in s and "momentum" in s


# ── ForecastSignal construction ───────────────────────────────────────────────


class TestForecastSignalInit:
    def test_default_weights_sum_to_one(self) -> None:
        fs = ForecastSignal()
        assert abs(sum(fs.weights.values()) - 1.0) < 1e-10

    def test_custom_weights_normalised(self) -> None:
        fs = ForecastSignal(weights={"regime": 0.5, "skew": 0.5})
        total = sum(fs.weights.values())
        assert abs(total - 1.0) < 1e-10

    def test_unknown_weight_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown weight"):
            ForecastSignal(weights={"regime": 0.5, "magic": 0.5})

    def test_zero_total_weight_raises(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            ForecastSignal(
                weights={
                    "regime": 0.0,
                    "skew": 0.0,
                    "prob_profit": 0.0,
                    "tail_risk": 0.0,
                    "vol": 0.0,
                    "momentum": 0.0,
                }
            )

    def test_allow_short_default_false(self) -> None:
        fs = ForecastSignal()
        assert not fs.allow_short


# ── Bullish scenario ──────────────────────────────────────────────────────────


class TestBullishSignal:
    def test_signal_is_long(self, bullish_forecast: ForecastResult) -> None:
        fs = ForecastSignal()
        result = fs.generate(bullish_forecast)
        assert result.signal == 1, f"Expected long, got {result.summary()}"

    def test_conviction_positive(self, bullish_forecast: ForecastResult) -> None:
        fs = ForecastSignal()
        result = fs.generate(bullish_forecast)
        assert result.conviction > 0.15

    def test_regime_component_positive(self, bullish_forecast: ForecastResult) -> None:
        fs = ForecastSignal()
        result = fs.generate(bullish_forecast)
        assert result.components["regime"] > 0

    def test_prob_profit_component_positive(self, bullish_forecast: ForecastResult) -> None:
        fs = ForecastSignal()
        result = fs.generate(bullish_forecast)
        assert result.components["prob_profit"] > 0

    def test_regime_confidence_high(self, bullish_forecast: ForecastResult) -> None:
        fs = ForecastSignal()
        result = fs.generate(bullish_forecast)
        assert result.regime_confidence > 0.90


# ── Bearish scenario ─────────────────────────────────────────────────────────


class TestBearishSignal:
    def test_signal_is_flat_by_default(self, bearish_forecast: ForecastResult) -> None:
        """Without allow_short, bearish = flat, not short."""
        fs = ForecastSignal()
        result = fs.generate(bearish_forecast)
        assert result.signal <= 0, f"Should not be long: {result.summary()}"

    def test_signal_is_short_when_allowed(self, bearish_forecast: ForecastResult) -> None:
        fs = ForecastSignal(allow_short=True)
        result = fs.generate(bearish_forecast)
        assert result.signal == -1, f"Expected short, got {result.summary()}"

    def test_regime_component_negative(self, bearish_forecast: ForecastResult) -> None:
        fs = ForecastSignal()
        result = fs.generate(bearish_forecast)
        assert result.components["regime"] < 0

    def test_prob_profit_component_negative(self, bearish_forecast: ForecastResult) -> None:
        fs = ForecastSignal()
        result = fs.generate(bearish_forecast)
        assert result.components["prob_profit"] < 0


# ── Ambiguous scenario ────────────────────────────────────────────────────────


class TestAmbiguousSignal:
    def test_signal_is_flat(self, ambiguous_forecast: ForecastResult) -> None:
        fs = ForecastSignal()
        result = fs.generate(ambiguous_forecast)
        assert result.signal == 0, f"Expected flat, got {result.summary()}"

    def test_conviction_below_threshold(self, ambiguous_forecast: ForecastResult) -> None:
        fs = ForecastSignal()
        result = fs.generate(ambiguous_forecast)
        assert result.conviction < 0.30


# ── Component scoring edge cases ─────────────────────────────────────────────


class TestComponentScoring:
    def test_all_components_bounded(self, bullish_forecast: ForecastResult) -> None:
        fs = ForecastSignal()
        result = fs.generate(bullish_forecast)
        for name, val in result.components.items():
            assert -1.0 <= val <= 1.0, f"{name}={val} out of [-1,1]"

    def test_all_components_bounded_bearish(self, bearish_forecast: ForecastResult) -> None:
        fs = ForecastSignal()
        result = fs.generate(bearish_forecast)
        for name, val in result.components.items():
            assert -1.0 <= val <= 1.0, f"{name}={val} out of [-1,1]"

    def test_high_threshold_produces_flat(self, bullish_forecast: ForecastResult) -> None:
        fs = ForecastSignal(conviction_threshold=0.99)
        result = fs.generate(bullish_forecast)
        assert result.signal == 0

    def test_zero_threshold_always_signals(self, bullish_forecast: ForecastResult) -> None:
        fs = ForecastSignal(conviction_threshold=0.0)
        result = fs.generate(bullish_forecast)
        assert result.signal != 0

    def test_regime_only_weight(self, bullish_forecast: ForecastResult) -> None:
        """With only regime weight, signal depends purely on regime mu."""
        fs = ForecastSignal(
            weights={
                "regime": 1.0,
                "skew": 0.0,
                "prob_profit": 0.0,
                "tail_risk": 0.0,
                "vol": 0.0,
                "momentum": 0.0,
            },
            conviction_threshold=0.01,
        )
        result = fs.generate(bullish_forecast)
        assert result.signal == 1


# ── Integration: signal from both forecaster types ────────────────────────────


class TestIntegration:
    def test_signal_from_default_forecast(self) -> None:
        """ForecastSignal works with minimal ForecastResult."""
        fc = _make_forecast()
        fs = ForecastSignal()
        result = fs.generate(fc)
        assert isinstance(result, SignalResult)
        assert result.signal in (-1, 0, 1)
        assert 0 <= result.conviction <= 1.0

    def test_signal_changes_with_regime(self) -> None:
        """Same returns but different regime → different signal."""
        rng = np.random.default_rng(99)
        rets = rng.normal(0.0, 0.01, 3000)  # flat returns

        fc_calm = _make_forecast(
            terminal_returns=rets,
            current_regime=0,
            regime_params={
                0: {"mu": 0.10 / 252, "sigma": 0.18 / np.sqrt(252)},
                1: {"mu": -0.40 / 252, "sigma": 0.40 / np.sqrt(252)},
            },
            regime_probabilities=np.array([0.95, 0.05]),
        )
        fc_crisis = _make_forecast(
            terminal_returns=rets,
            current_regime=1,
            regime_params={
                0: {"mu": 0.10 / 252, "sigma": 0.18 / np.sqrt(252)},
                1: {"mu": -0.40 / 252, "sigma": 0.40 / np.sqrt(252)},
            },
            regime_probabilities=np.array([0.05, 0.95]),
        )

        fs = ForecastSignal()
        sig_calm = fs.generate(fc_calm)
        sig_crisis = fs.generate(fc_crisis)

        # Regime component should flip sign
        assert sig_calm.components["regime"] > 0
        assert sig_crisis.components["regime"] < 0


# ── Momentum component ───────────────────────────────────────────────────────────────────


class TestDecomposedSignal:
    """Direction-from-momentum, size-from-uncertainty separation."""

    def test_direction_flat_produces_zero_position(self, bullish_forecast: ForecastResult) -> None:
        fs = ForecastSignal()
        result = fs.generate_decomposed(bullish_forecast, direction=0)
        assert result.direction == 0
        assert result.position == 0.0

    def test_size_shrinks_in_crisis(
        self, bullish_forecast: ForecastResult, bearish_forecast: ForecastResult
    ) -> None:
        """Higher vol and tail drag → smaller position size."""
        fs = ForecastSignal()
        bull = fs.generate_decomposed(bullish_forecast, direction=1)
        bear = fs.generate_decomposed(bearish_forecast, direction=1)
        assert bull.size > bear.size

    def test_size_in_unit_interval(self, bullish_forecast: ForecastResult) -> None:
        fs = ForecastSignal()
        result = fs.generate_decomposed(bullish_forecast, direction=1)
        assert 0.0 <= result.size <= 1.0

    def test_invalid_direction_raises(self, bullish_forecast: ForecastResult) -> None:
        fs = ForecastSignal()
        with pytest.raises(ValueError):
            fs.generate_decomposed(bullish_forecast, direction=2)

    def test_position_is_direction_times_size(self, bullish_forecast: ForecastResult) -> None:
        fs = ForecastSignal()
        result = fs.generate_decomposed(bullish_forecast, direction=-1)
        assert abs(result.position - (-1 * result.size)) < 1e-12


class TestMomentumComponent:
    def test_no_recent_returns_neutral(self) -> None:
        """Without recent_returns, momentum scores 0."""
        fc = _make_forecast()
        fs = ForecastSignal()
        result = fs.generate(fc)  # no recent_returns
        assert result.components["momentum"] == 0.0

    def test_too_short_returns_neutral(self) -> None:
        """Returns shorter than momentum_short → neutral."""
        fc = _make_forecast()
        fs = ForecastSignal(momentum_short=21)
        short_rets = np.zeros(10)  # only 10 days
        result = fs.generate(fc, recent_returns=short_rets)
        assert result.components["momentum"] == 0.0

    def test_positive_trend_positive_score(self) -> None:
        """Steady upward trend → positive momentum score."""
        fc = _make_forecast()
        fs = ForecastSignal()
        # Steady +0.1%/day for 63 days → ~25%/yr
        rets = np.full(63, 0.001)
        result = fs.generate(fc, recent_returns=rets)
        assert result.components["momentum"] > 0.3

    def test_negative_trend_negative_score(self) -> None:
        """Steady downward trend → negative momentum score."""
        fc = _make_forecast()
        fs = ForecastSignal()
        # Steady -0.1%/day for 63 days → ~-25%/yr
        rets = np.full(63, -0.001)
        result = fs.generate(fc, recent_returns=rets)
        assert result.components["momentum"] < -0.3

    def test_strong_trend_saturates(self) -> None:
        """Very strong trend should be clipped to [-1, 1]."""
        fc = _make_forecast()
        fs = ForecastSignal()
        rets = np.full(63, 0.01)  # +1%/day → huge
        result = fs.generate(fc, recent_returns=rets)
        assert -1.0 <= result.components["momentum"] <= 1.0

    def test_conflicting_timeframes_dampened(self) -> None:
        """Short positive, long negative → dampened by confirmation."""
        fc = _make_forecast()
        fs = ForecastSignal(momentum_short=21, momentum_long=63)
        # Long: 42 days down, then 21 days up
        rets = np.concatenate([np.full(42, -0.002), np.full(21, 0.002)])
        result_mixed = fs.generate(fc, recent_returns=rets)

        # All up for comparison
        rets_aligned = np.full(63, 0.002)
        result_aligned = fs.generate(fc, recent_returns=rets_aligned)

        # Aligned should have stronger momentum than conflicting
        assert abs(result_aligned.components["momentum"]) > abs(result_mixed.components["momentum"])

    def test_spy_scenario_slow_drift_down(self) -> None:
        """Reproduce the SPY scenario: calm regime but slowly drifting lower.

        HMM says calm (low vol), but momentum should catch the drift.
        """
        rng = np.random.default_rng(42)
        # Calm regime, positive mu — HMM thinks bullish
        daily_mu = 0.10 / 252
        daily_sigma = 0.15 / np.sqrt(252)
        base = rng.normal(daily_mu * 21, daily_sigma * np.sqrt(21), 5000)
        fc = _make_forecast(
            terminal_returns=base,
            regime_params={
                0: {"mu": daily_mu, "sigma": daily_sigma},
                1: {"mu": -0.40 / 252, "sigma": 0.40 / np.sqrt(252)},
            },
            regime_probabilities=np.array([0.98, 0.02]),
        )

        # Recent returns: slow grind down at low vol (~-0.15%/day for 30 days)
        recent = np.full(63, -0.0015)

        fs_no_mom = ForecastSignal(
            weights={
                "regime": 0.30,
                "skew": 0.15,
                "prob_profit": 0.25,
                "tail_risk": 0.20,
                "vol": 0.10,
                "momentum": 0.0,
            },
        )
        fs_with_mom = ForecastSignal()  # default includes momentum

        result_no_mom = fs_no_mom.generate(fc)
        result_with_mom = fs_with_mom.generate(fc, recent_returns=recent)

        # Without momentum, regime says bullish → likely LONG
        # With momentum, negative drift should lower conviction or flip to FLAT
        assert (
            result_with_mom.conviction < result_no_mom.conviction
            or result_with_mom.signal < result_no_mom.signal
        ), (
            f"Momentum should reduce conviction: "
            f"no_mom={result_no_mom.summary()}, with_mom={result_with_mom.summary()}"
        )

    def test_custom_lookback_windows(self) -> None:
        """Custom momentum windows are respected."""
        fc = _make_forecast()
        fs = ForecastSignal(momentum_short=10, momentum_long=30)
        rets = np.full(30, 0.002)
        result = fs.generate(fc, recent_returns=rets)
        assert result.components["momentum"] > 0.0

    def test_momentum_bounded(self) -> None:
        """Momentum component always in [-1, 1]."""
        fc = _make_forecast()
        fs = ForecastSignal()
        for val in [0.01, -0.01, 0.0, 0.005, -0.005]:
            rets = np.full(63, val)
            result = fs.generate(fc, recent_returns=rets)
            assert -1.0 <= result.components["momentum"] <= 1.0
