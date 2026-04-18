"""Tests for directional (momentum) signals."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from synthquant.strategy.directional import (
    CrossSectionalMomentum,
    DirectionalSignal,
    TimeSeriesMomentum,
)


class TestTimeSeriesMomentum:
    def test_insufficient_history_returns_flat(self) -> None:
        mom = TimeSeriesMomentum(lookback_long=252, skip=21)
        out = mom.generate(np.zeros(100))
        assert out.direction == 0
        assert out.strength == 0.0

    def test_steady_uptrend_long(self) -> None:
        mom = TimeSeriesMomentum(lookback_long=252, skip=21, allow_short=True)
        rets = np.full(300, 0.001)  # steady up
        out = mom.generate(rets)
        assert out.direction == 1

    def test_steady_downtrend_flat_when_long_only(self) -> None:
        mom = TimeSeriesMomentum(lookback_long=252, skip=21, allow_short=False)
        rets = np.full(300, -0.001)
        out = mom.generate(rets)
        assert out.direction == 0

    def test_steady_downtrend_short_when_allowed(self) -> None:
        mom = TimeSeriesMomentum(lookback_long=252, skip=21, allow_short=True)
        rets = np.full(300, -0.001)
        out = mom.generate(rets)
        assert out.direction == -1

    def test_skip_avoids_recent_reversal(self) -> None:
        """If the most recent month is down but the 12-1 window is up,
        momentum should still be long (that's the point of the skip)."""
        mom = TimeSeriesMomentum(lookback_long=252, skip=21, allow_short=True)
        rets = np.concatenate([np.full(231, 0.001), np.full(21, -0.01)])
        # total = 252, last 21 are sharp down, first 231 are up
        out = mom.generate(rets)
        assert out.direction == 1

    def test_skip_validation(self) -> None:
        with pytest.raises(ValueError):
            TimeSeriesMomentum(lookback_long=21, skip=21)


class TestCrossSectionalMomentum:
    def test_flat_when_too_short(self) -> None:
        cs = CrossSectionalMomentum(lookback=252, skip=21, top_n=1)
        out = cs.generate(np.zeros((100, 3)))
        assert (out == 0).all()

    def test_top_assets_long(self) -> None:
        cs = CrossSectionalMomentum(lookback=252, skip=21, top_n=1)
        rng = np.random.default_rng(0)
        rets = rng.normal(0.0, 0.01, size=(300, 3))
        rets[:, 0] += 0.002  # asset 0 is the winner
        out = cs.generate(rets)
        assert out[0] == 1
        assert out[1] == 0
        assert out[2] == 0

    def test_bottom_short(self) -> None:
        cs = CrossSectionalMomentum(lookback=252, skip=21, top_n=1, bottom_n=1)
        rng = np.random.default_rng(0)
        rets = rng.normal(0.0, 0.01, size=(300, 3))
        rets[:, 0] += 0.002
        rets[:, 2] -= 0.002
        out = cs.generate(rets)
        assert out[0] == 1
        assert out[2] == -1

    def test_invalid_shape_raises(self) -> None:
        cs = CrossSectionalMomentum()
        with pytest.raises(ValueError):
            cs.generate(np.zeros(100))


class TestDirectionalSignalDataclass:
    def test_immutable(self) -> None:
        ds = DirectionalSignal(direction=1, strength=0.5)
        with pytest.raises((TypeError, AttributeError, dataclasses.FrozenInstanceError)):
            ds.direction = 0  # type: ignore[misc]
