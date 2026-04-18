"""Probabilistic forecaster combining regime detection, volatility modeling, and Monte Carlo simulation."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from synthquant.analytics.distributions import EmpiricalDistribution
from synthquant.analytics.risk_metrics import (
    conditional_var,
    max_drawdown_distribution,
    value_at_risk,
)
from synthquant.analytics.scoring import ForecastScorer
from synthquant.models.garch import GARCHModel
from synthquant.regime.ensemble import EnsembleRegimeDetector
from synthquant.regime.hmm import HMMRegimeDetector
from synthquant.regime.ms_garch import MarkovSwitchingGARCH
from synthquant.simulation.engines.regime_switching import RegimeSwitchingEngine

logger = logging.getLogger(__name__)

__all__ = ["ProbabilisticForecaster", "ForecastResult"]

DetectorType = Literal["hmm", "ms_garch", "ensemble"]


@dataclass
class ForecastResult:
    """Container for probabilistic forecast outputs.

    Attributes:
        paths: Simulated price paths, shape (n_paths, horizon+1).
        terminal_prices: Terminal prices, shape (n_paths,).
        terminal_returns: Log returns to horizon, shape (n_paths,).
        distribution: Fitted empirical distribution of terminal returns.
        var_95: 95% Value-at-Risk (log return, negative = loss).
        cvar_95: 95% Conditional VaR.
        var_99: 99% Value-at-Risk.
        cvar_99: 99% Conditional VaR.
        max_drawdowns: Max drawdown per path, shape (n_paths,).
        confidence_bands: Dict mapping quantile labels to time-series arrays.
        current_regime: Most likely current regime index.
        regime_probabilities: Current regime membership probabilities.
        regime_params: Fitted parameters per regime.
        horizon: Forecast horizon in steps.
        current_price: Starting price.
    """

    paths: np.ndarray
    terminal_prices: np.ndarray
    terminal_returns: np.ndarray
    distribution: EmpiricalDistribution
    var_95: float
    cvar_95: float
    var_99: float
    cvar_99: float
    max_drawdowns: np.ndarray
    confidence_bands: dict[str, np.ndarray]
    current_regime: int
    regime_probabilities: np.ndarray
    regime_params: dict[int, dict[str, Any]]
    horizon: int
    current_price: float
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def expected_return(self) -> float:
        """Mean terminal log return across all paths."""
        return float(np.mean(self.terminal_returns))

    @property
    def median_return(self) -> float:
        """Median terminal log return."""
        return float(np.median(self.terminal_returns))

    @property
    def return_std(self) -> float:
        """Standard deviation of terminal log returns."""
        return float(np.std(self.terminal_returns))

    @property
    def prob_positive(self) -> float:
        """Probability of a positive return over the horizon."""
        return float(np.mean(self.terminal_returns > 0))

    @property
    def prob_loss_gt_10pct(self) -> float:
        """Probability of losing more than 10%."""
        return float(np.mean(self.terminal_returns < np.log(0.90)))

    @property
    def prob_loss_gt_20pct(self) -> float:
        """Probability of losing more than 20%."""
        return float(np.mean(self.terminal_returns < np.log(0.80)))

    def quantile(self, q: float) -> float:
        """Return quantile of terminal log returns."""
        return float(np.quantile(self.terminal_returns, q))

    @property
    def realized_vol_samples(self) -> np.ndarray:
        """Per-path annualised realised volatility over the forecast horizon.

        For each simulated path, computes the sample std of daily log
        returns and annualises by √252. This is the distribution used
        to score against actual realised volatility (matching the
        synthdata.co volatility target), and avoids the incorrect
        linear √T scaling of terminal-return std.
        """
        if self.paths.shape[1] < 2:
            return np.zeros(self.paths.shape[0])
        log_rets = np.log(self.paths[:, 1:] / self.paths[:, :-1])
        return np.asarray(np.std(log_rets, axis=1) * np.sqrt(252), dtype=float)

    @property
    def forecast_realized_vol(self) -> float:
        """Point forecast (mean) of path-wise annualised realised vol."""
        return float(np.mean(self.realized_vol_samples))

    def price_quantile(self, q: float) -> float:
        """Return quantile of terminal prices."""
        return float(np.quantile(self.terminal_prices, q))

    def summary(self) -> dict[str, Any]:
        """Return a summary dict of key forecast metrics."""
        return {
            "horizon_steps": self.horizon,
            "current_price": self.current_price,
            "current_regime": self.current_regime,
            "regime_probabilities": self.regime_probabilities.tolist(),
            "n_paths": len(self.terminal_returns),
            "expected_return": self.expected_return,
            "median_return": self.median_return,
            "return_std": self.return_std,
            "prob_positive": self.prob_positive,
            "var_95": self.var_95,
            "cvar_95": self.cvar_95,
            "var_99": self.var_99,
            "cvar_99": self.cvar_99,
            "prob_loss_gt_10pct": self.prob_loss_gt_10pct,
            "prob_loss_gt_20pct": self.prob_loss_gt_20pct,
            "median_max_drawdown": float(np.median(self.max_drawdowns)),
            "price_p5": self.price_quantile(0.05),
            "price_p25": self.price_quantile(0.25),
            "price_p50": self.price_quantile(0.50),
            "price_p75": self.price_quantile(0.75),
            "price_p95": self.price_quantile(0.95),
        }


class ProbabilisticForecaster:
    """Regime-aware probabilistic forecaster.

    Combines regime detection (HMM, MS-GARCH, or ensemble), GARCH
    volatility conditioning, and regime-switching Monte Carlo simulation
    to produce full forecast distributions.

    Args:
        detector: Regime detection method.
        n_regimes: Number of regimes.
        n_paths: Number of Monte Carlo paths to simulate.
        vol_model: GARCH variant for volatility conditioning.
        garch_dist: Innovation distribution for GARCH.
        random_state: Seed for reproducibility.
    """

    def __init__(
        self,
        detector: DetectorType = "hmm",
        n_regimes: int = 2,
        n_paths: int = 10_000,
        vol_model: Literal["GARCH", "EGARCH", "GJRGARCH"] = "GARCH",
        garch_dist: str = "normal",
        random_state: int | None = None,
        use_filtered: bool = True,
    ) -> None:
        self.detector_type = detector
        self.n_regimes = n_regimes
        self.n_paths = n_paths
        self.vol_model = vol_model
        self.garch_dist = garch_dist
        self.random_state = random_state
        self.use_filtered = use_filtered

        self._detector: Any = None
        self._garch: GARCHModel | None = None
        self._regime_params: dict[int, dict[str, Any]] = {}
        self._transition_matrix: np.ndarray | None = None
        self._fitted = False
        self._returns: np.ndarray | None = None

    def fit(self, returns: np.ndarray) -> ProbabilisticForecaster:
        """Fit regime detector and volatility model to historical returns.

        Args:
            returns: 1-D array of daily log returns.

        Returns:
            self (fitted forecaster).
        """
        returns = np.asarray(returns, dtype=float)
        self._returns = returns

        # 1. Build and fit regime detector
        self._detector = self._build_detector()
        self._detector.fit(returns)

        # 2. Extract regime parameters
        raw_params = self._detector.get_regime_params()
        self._regime_params = self._normalize_regime_params(raw_params)

        # 3. Extract or estimate transition matrix
        self._transition_matrix = self._estimate_transition_matrix(returns)

        # 4. Fit GARCH for conditional volatility scaling
        self._garch = GARCHModel(
            vol_model=self.vol_model,
            dist=self.garch_dist,
        )
        self._garch.fit(returns)

        self._fitted = True
        logger.info(
            f"ProbabilisticForecaster fitted: detector={self.detector_type}, "
            f"regimes={self.n_regimes}, n_obs={len(returns)}"
        )
        return self

    def forecast(
        self,
        horizon: int = 252,
        current_price: float = 100.0,
        dt: float = 1 / 252,
    ) -> ForecastResult:
        """Generate a probabilistic forecast.

        Args:
            horizon: Number of steps (trading days) to forecast.
            current_price: Starting price level.
            dt: Time step in years (1/252 for daily).

        Returns:
            ForecastResult with full distribution of outcomes.
        """
        self._check_fitted()

        # Current regime state — prefer forward-only filtered posterior to
        # avoid forward-backward smoothing lookahead inside the training window.
        if self.use_filtered and hasattr(self._detector, "predict_proba_filtered"):
            proba = self._detector.predict_proba_filtered(self._returns)
        else:
            proba = self._detector.predict_proba(self._returns)
        current_proba = proba[-1]
        current_regime = int(np.argmax(current_proba))

        # Scale regime volatilities using GARCH's latest conditional vol
        regime_params = self._condition_on_garch(horizon)

        # Annualise regime params for the simulation engine
        ann_params = []
        for i in range(self.n_regimes):
            rp = regime_params[i]
            ann_params.append({
                "mu": rp["mu"] * 252,       # daily → annual drift
                "sigma": rp["sigma"] * np.sqrt(252),  # daily → annual vol
            })

        # Build and run simulation engine
        assert self._transition_matrix is not None, "fit() must be called before forecast()"
        engine = RegimeSwitchingEngine(
            regime_params=ann_params,
            transition_matrix=self._transition_matrix,
            initial_regime=current_regime,
        )
        paths = engine.simulate(
            n_paths=self.n_paths,
            n_steps=horizon,
            dt=dt,
            S0=current_price,
            random_state=self.random_state,
        )

        # Extract terminal values
        terminal_prices = paths[:, -1]
        terminal_returns = np.log(terminal_prices / current_price)

        # Risk metrics
        var95 = value_at_risk(paths, alpha=0.05)
        cvar95 = conditional_var(paths, alpha=0.05)
        var99 = value_at_risk(paths, alpha=0.01)
        cvar99 = conditional_var(paths, alpha=0.01)
        mdd = max_drawdown_distribution(paths)

        # Confidence bands (quantiles over time)
        quantiles = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
        bands = {}
        for q in quantiles:
            bands[f"p{int(q*100):02d}"] = np.quantile(paths, q, axis=0)

        # Fit empirical distribution
        dist = EmpiricalDistribution()
        dist.fit(paths)

        result = ForecastResult(
            paths=paths,
            terminal_prices=terminal_prices,
            terminal_returns=terminal_returns,
            distribution=dist,
            var_95=var95,
            cvar_95=cvar95,
            var_99=var99,
            cvar_99=cvar99,
            max_drawdowns=mdd,
            confidence_bands=bands,
            current_regime=current_regime,
            regime_probabilities=current_proba,
            regime_params=regime_params,
            horizon=horizon,
            current_price=current_price,
            metadata={
                "detector": self.detector_type,
                "n_regimes": self.n_regimes,
                "n_paths": self.n_paths,
                "vol_model": self.vol_model,
                "dt": dt,
                "annualised_params": ann_params,
            },
        )

        logger.info(
            f"Forecast generated: horizon={horizon}, E[r]={result.expected_return:.4f}, "
            f"VaR95={var95:.4f}, regime={current_regime}"
        )
        return result

    def score(
        self,
        forecast_result: ForecastResult,
        actual_returns: np.ndarray,
    ) -> dict[str, Any]:
        """Score a forecast against realised returns.

        Args:
            forecast_result: A previous ForecastResult.
            actual_returns: Realised daily log returns over the forecast horizon.

        Returns:
            Dict with CRPS, coverage, KS test, and calibration metrics.
        """
        actual = np.asarray(actual_returns, dtype=float)
        n_actual = len(actual)

        # Build rolling forecast ensembles for CRPS / coverage
        # Compare cumulative returns at each step
        cum_actual = np.cumsum(actual)
        sim_returns = np.log(
            forecast_result.paths[:, 1:] / forecast_result.paths[:, :-1]
        )

        # Terminal return comparison
        actual_terminal = float(cum_actual[-1]) if n_actual > 0 else 0.0
        sim_terminal = forecast_result.terminal_returns

        # CRPS on terminal return
        forecasts_2d = sim_terminal.reshape(1, -1)
        observations_1d = np.array([actual_terminal])
        crps = ForecastScorer.crps(forecasts_2d, observations_1d)

        # Coverage test
        coverage = ForecastScorer.coverage_test(forecasts_2d, observations_1d)

        # KS test: simulated daily returns vs actual daily returns
        sim_daily = sim_returns.flatten()
        ks = ForecastScorer.ks_test(sim_daily, actual)

        # Realised vs forecasted volatility. Use the path-wise RV
        # distribution rather than linear √T scaling of terminal-return std
        # (the latter is wrong under stochastic / rough vol).
        realised_vol = float(np.std(actual) * np.sqrt(252))
        rv_samples = forecast_result.realized_vol_samples
        forecast_vol = float(np.mean(rv_samples))
        rv_crps = ForecastScorer.crps(
            rv_samples.reshape(1, -1), np.array([realised_vol])
        )

        return {
            "crps": crps,
            "rv_crps": rv_crps,
            "coverage": coverage,
            "ks_test": ks,
            "actual_terminal_return": actual_terminal,
            "forecast_mean_return": forecast_result.expected_return,
            "forecast_median_return": forecast_result.median_return,
            "realised_annual_vol": realised_vol,
            "forecast_annual_vol": forecast_vol,
            "actual_in_90pct_interval": bool(
                forecast_result.quantile(0.05)
                <= actual_terminal
                <= forecast_result.quantile(0.95)
            ),
            "actual_in_50pct_interval": bool(
                forecast_result.quantile(0.25)
                <= actual_terminal
                <= forecast_result.quantile(0.75)
            ),
        }

    # ------------------------------------------------------------------ #
    # Private helpers
    # ------------------------------------------------------------------ #

    def _build_detector(self) -> Any:
        """Instantiate the configured regime detector."""
        if self.detector_type == "hmm":
            return HMMRegimeDetector(
                n_components=self.n_regimes,
                random_state=self.random_state,
            )
        elif self.detector_type == "ms_garch":
            return MarkovSwitchingGARCH(k_regimes=self.n_regimes)
        elif self.detector_type == "ensemble":
            return EnsembleRegimeDetector(
                detectors=[
                    HMMRegimeDetector(
                        n_components=self.n_regimes,
                        random_state=self.random_state,
                    ),
                    MarkovSwitchingGARCH(k_regimes=self.n_regimes),
                ],
                method="proba_average",
                n_regimes=self.n_regimes,
            )
        else:
            raise ValueError(f"Unknown detector type: {self.detector_type}")

    def _normalize_regime_params(
        self, raw: dict[int, dict[str, Any]]
    ) -> dict[int, dict[str, Any]]:
        """Normalise regime params to a consistent {mu, sigma} format."""
        params: dict[int, dict[str, Any]] = {}
        for i, rp in raw.items():
            if "mu" in rp and "sigma" in rp:
                params[i] = {"mu": float(rp["mu"]), "sigma": float(rp["sigma"])}
            elif "means" in rp:
                # GMM / clustering format
                mu = float(rp["means"][0]) if isinstance(rp["means"], list) else float(rp["means"])
                cov = rp.get("covariances", rp.get("sigma", 0.01))
                if isinstance(cov, list):
                    sigma = float(np.sqrt(abs(cov[0]))) if isinstance(cov[0], list) else float(np.sqrt(abs(cov[0])))
                else:
                    sigma = float(np.sqrt(abs(cov)))
                params[i] = {"mu": mu, "sigma": sigma}
            else:
                # Fallback: use whatever keys are available
                mu = float(rp.get("mu", rp.get("mean", 0.0)))
                sigma = float(rp.get("sigma", rp.get("std", rp.get("volatility", 0.01))))
                params[i] = {"mu": mu, "sigma": sigma}
        return params

    def _estimate_transition_matrix(self, returns: np.ndarray) -> np.ndarray:
        """Extract or estimate the regime transition matrix."""
        # HMM has a built-in transition matrix
        if hasattr(self._detector, "_model") and hasattr(
            getattr(self._detector, "_model", None), "transmat_"
        ):
            return np.array(self._detector._model.transmat_)

        # For ensemble / MS-GARCH, estimate from predicted labels
        labels = self._detector.predict(returns)
        n = self.n_regimes
        trans = np.zeros((n, n))
        for t in range(len(labels) - 1):
            trans[labels[t], labels[t + 1]] += 1

        # Add Laplace smoothing and normalise
        trans += 1e-6
        row_sums = trans.sum(axis=1, keepdims=True)
        return trans / row_sums

    def _condition_on_garch(self, horizon: int) -> dict[int, dict[str, Any]]:
        """Scale regime volatilities using GARCH's conditional forecast.

        Uses the ratio of GARCH's 1-step forecast to the unconditional
        volatility to scale regime-specific sigmas, capturing the current
        volatility state.
        """
        if self._garch is None or self._garch._result is None:
            return self._regime_params

        # GARCH 1-step conditional vol
        garch_vol = self._garch.forecast(horizon=1)
        cond_vol_daily = garch_vol[0] / np.sqrt(252)  # annualised → daily

        # Unconditional daily vol from training data
        uncond_vol = float(np.std(self._returns))

        # Scaling ratio: if GARCH says vol is elevated, scale all regimes up
        if uncond_vol > 1e-10:
            scale = cond_vol_daily / uncond_vol
        else:
            scale = 1.0

        # Clamp scaling to avoid extremes
        scale = float(np.clip(scale, 0.5, 2.0))

        conditioned = {}
        for i, rp in self._regime_params.items():
            conditioned[i] = {
                "mu": rp["mu"],
                "sigma": rp["sigma"] * scale,
            }

        logger.debug(
            f"GARCH conditioning: cond_vol={cond_vol_daily:.6f}, "
            f"uncond_vol={uncond_vol:.6f}, scale={scale:.3f}"
        )
        return conditioned

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError(
                "Forecaster has not been fitted. Call fit() first."
            )
