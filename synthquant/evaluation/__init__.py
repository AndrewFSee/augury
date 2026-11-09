"""Out-of-sample evaluation: proper scores, VaR backtests and walk-forward runs."""

from synthquant.evaluation.backtests import (
    basel_zone,
    christoffersen_test,
    conditional_coverage_test,
    diebold_mariano,
    kupiec_test,
)
from synthquant.evaluation.metrics import crps_ensemble, interval_coverage, pit_values
from synthquant.evaluation.walk_forward import WalkForwardResult, walk_forward

__all__ = [
    "crps_ensemble",
    "pit_values",
    "interval_coverage",
    "kupiec_test",
    "christoffersen_test",
    "conditional_coverage_test",
    "basel_zone",
    "diebold_mariano",
    "walk_forward",
    "WalkForwardResult",
]
