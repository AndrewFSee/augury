"""Trading strategy components."""

from synthquant.strategy.allocation import MeanCVaROptimizer, RiskParityAllocator
from synthquant.strategy.backtest import WalkForwardBacktest
from synthquant.strategy.directional import (
    CrossSectionalMomentum,
    DirectionalSignal,
    TimeSeriesMomentum,
)
from synthquant.strategy.hedging import DeltaHedger, GammaHedger
from synthquant.strategy.forecast_signals import ForecastSignal, SignalResult
from synthquant.strategy.predictive_allocator import PredictiveAllocator, PredictiveAllocationResult
from synthquant.strategy.predictive_signals import PredictiveSignals, SignalOutput, compute_predictive_signals
from synthquant.strategy.regime_allocator import AllocationResult, RegimeAllocator
from synthquant.strategy.signal_tuner import ForecastSnapshot, SignalTuner
from synthquant.strategy.signals import RegimeSignalGenerator
from synthquant.strategy.sizing import CVaROptimalSizer, KellyCriterion, RiskParitySizer

__all__ = [
    "AllocationResult",
    "CrossSectionalMomentum",
    "DirectionalSignal",
    "TimeSeriesMomentum",
    "ForecastSignal",
    "ForecastSnapshot",
    "PredictiveAllocator",
    "PredictiveAllocationResult",
    "PredictiveSignals",
    "RegimeAllocator",
    "SignalOutput",
    "SignalResult",
    "SignalTuner",
    "compute_predictive_signals",
    "KellyCriterion",
    "RiskParitySizer",
    "CVaROptimalSizer",
    "RegimeSignalGenerator",
    "DeltaHedger",
    "GammaHedger",
    "MeanCVaROptimizer",
    "RiskParityAllocator",
    "WalkForwardBacktest",
]
