"""Out-of-sample evaluation: proper scores, VaR backtests, walk-forward runs and regimes."""

from augury.evaluation.backtests import (
    basel_zone,
    christoffersen_test,
    conditional_coverage_test,
    diebold_mariano,
    kupiec_test,
)
from augury.evaluation.metrics import crps_ensemble, interval_coverage, pit_values
from augury.evaluation.regimes import (
    RollingRegimes,
    drawdown_detection,
    incremental_information,
    regime_stability,
    regime_tail_test,
    rolling_garch_variance,
    rolling_regime_probabilities,
)
from augury.evaluation.walk_forward import WalkForwardResult, walk_forward

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
    "RollingRegimes",
    "rolling_regime_probabilities",
    "rolling_garch_variance",
    "incremental_information",
    "regime_tail_test",
    "regime_stability",
    "drawdown_detection",
]
