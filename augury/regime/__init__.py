"""Regime detection algorithms."""

from augury.regime.clustering import ClusteringRegimeDetector
from augury.regime.ensemble import EnsembleRegimeDetector
from augury.regime.features import (
    ALL_FEATURE_NAMES,
    DEFAULT_FEATURE_NAMES,
    FEATURE_REGISTRY,
    build_hmm_features,
)
from augury.regime.hmm import HMMRegimeDetector
from augury.regime.ms_garch import MarkovSwitchingGARCH

__all__ = [
    "HMMRegimeDetector",
    "MarkovSwitchingGARCH",
    "ClusteringRegimeDetector",
    "EnsembleRegimeDetector",
    "build_hmm_features",
    "FEATURE_REGISTRY",
    "ALL_FEATURE_NAMES",
    "DEFAULT_FEATURE_NAMES",
]
