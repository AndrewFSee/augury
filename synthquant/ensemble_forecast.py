"""Ensemble forecaster: regime-conditional model calibration + weighted multi-engine simulation.

Wires together all four layers of the SynthQuant architecture:

    Regime Layer  →  detects current market state (HMM / MS-GARCH)
    Model Layer   →  calibrates Heston vol + Merton jump params per regime
    Simulation    →  runs Heston, Merton JD, and Regime-Switching engines
    Analytics     →  blends paths into a single forecast distribution

The ensemble uses **regime-conditional weights**: in calm markets,
the Heston engine (stochastic vol) dominates; in crisis markets, the
Merton JD engine (jumps) gets higher weight.  The Regime-Switching GBM
engine provides the backbone.
"""

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
from synthquant.forecast import ForecastResult
from synthquant.models.garch import GARCHModel
from synthquant.regime.ensemble import EnsembleRegimeDetector
from synthquant.regime.hmm import HMMRegimeDetector
from synthquant.regime.ms_garch import MarkovSwitchingGARCH
from synthquant.simulation.engines.heston import HestonEngine
from synthquant.simulation.engines.merton_jd import MertonJDEngine
from synthquant.simulation.engines.regime_switching import RegimeSwitchingEngine
from synthquant.simulation.engines.rough_bergomi import RoughBergomiEngine

logger = logging.getLogger(__name__)

__all__ = ["EnsembleForecaster", "CalibratedRegimeParams", "CalibrationBounds"]

DetectorType = Literal["hmm", "ms_garch", "ensemble"]


@dataclass
class CalibrationBounds:
    """Configurable bounds for model calibration parameters."""

    heston_rho: tuple[float, float] = (-0.95, -0.2)
    heston_kappa: tuple[float, float] = (0.5, 15.0)
    heston_sigma_v: tuple[float, float] = (0.25, 1.5)
    garch_scale_clamp: tuple[float, float] = (0.5, 2.0)
    jump_lambda: tuple[float, float] = (0.5, 50.0)
    jump_mu: tuple[float, float] = (-0.10, 0.02)
    jump_sigma_floor_mult: float = 1.5
    jump_sigma_ceil: float = 0.20
    min_regime_obs: int = 30
    feller_safety_factor: float = 1.8


@dataclass
class CalibratedRegimeParams:
    """Per-regime calibrated parameters for all model layers.

    Attributes:
        regime_id: Regime index.
        mu: Daily drift.
        sigma: Daily volatility (from regime detector).
        heston: Calibrated Heston params {kappa, theta, sigma_v, rho, v0}.
        jumps: Calibrated jump params {lambda_j, mu_j, sigma_j}.
        weight_in_regime: Fraction of training data in this regime.
    """

    regime_id: int
    mu: float
    sigma: float
    heston: dict[str, float] = field(default_factory=dict)
    jumps: dict[str, float] = field(default_factory=dict)
    rough_bergomi: dict[str, float] = field(default_factory=dict)
    weight_in_regime: float = 0.5


class EnsembleForecaster:
    """Regime-conditional weighted ensemble forecaster.

    Runs three simulation engines with regime-calibrated parameters
    and blends their output distributions using regime-conditional
    weights.

    Engines:
        1. **Heston** — captures stochastic volatility, mean-reversion,
           and vol-of-vol dynamics.  Dominates in calm/trending regimes.
        2. **Merton Jump-Diffusion** — captures crash risk and sudden
           jumps.  Dominates in crisis/volatile regimes.
        3. **Regime-Switching GBM** — captures macro regime transitions.
           Provides the structural backbone.

    Weights are determined by the current regime probabilities:
        - ``w_heston``  = base_heston  × (1 + calm_regime_prob)
        - ``w_merton``  = base_merton  × (1 + crisis_regime_prob)
        - ``w_regime``  = base_regime  (constant backbone weight)
        - Normalised to sum to 1.

    Args:
        detector: Regime detection method.
        n_regimes: Number of regimes.
        n_paths: Total paths across all engines (split proportionally).
        vol_model: GARCH variant for conditional vol scaling.
        random_state: Seed for reproducibility.
        engine_base_weights: Base (mu, sigma, regime) weights before
            regime conditioning.  Default (0.35, 0.30, 0.35).
        use_filtered: If True, use forward-only filtered regime
            probabilities (no look-ahead) instead of the default
            forward-backward smoother.  Recommended for live trading.
        use_features: If True, build multivariate interaction features
            for HMM regime detection so the model can distinguish
            crashes from recoveries.  Only affects the ``'hmm'``
            detector; ignored for others.
        feature_names: Which features to build when *use_features* is
            True.  Defaults to ``['returns', 'signed_vol', 'momentum']``.
            See :data:`synthquant.regime.features.ALL_FEATURE_NAMES`
            for the full list of available features.
    """

    def __init__(
        self,
        detector: DetectorType = "hmm",
        n_regimes: int = 2,
        n_paths: int = 10_000,
        vol_model: Literal["GARCH", "EGARCH", "GJRGARCH"] = "GARCH",
        random_state: int | None = None,
        engine_base_weights: tuple[float, float, float] = (0.35, 0.30, 0.35),
        use_filtered: bool = True,
        use_features: bool = False,
        feature_names: list[str] | None = None,
        calibration_bounds: CalibrationBounds | None = None,
        use_rough_bergomi: bool = False,
    ) -> None:
        self.detector_type = detector
        self.n_regimes = n_regimes
        self.n_paths = n_paths
        self.vol_model = vol_model
        self.random_state = random_state
        self.engine_base_weights = engine_base_weights
        self.use_filtered = use_filtered
        self.use_features = use_features
        self.feature_names = feature_names
        self.bounds = calibration_bounds or CalibrationBounds()
        self.use_rough_bergomi = use_rough_bergomi

        self._detector: Any = None
        self._garch: GARCHModel | None = None
        self._calibrated: dict[int, CalibratedRegimeParams] = {}
        self._transition_matrix: np.ndarray | None = None
        self._fitted = False
        self._returns: np.ndarray | None = None
        self._obs: np.ndarray | None = None  # features or returns for detector

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def fit(self, returns: np.ndarray) -> EnsembleForecaster:
        """Fit all layers: regime detection, model calibration, GARCH.

        Args:
            returns: 1-D array of daily log returns.

        Returns:
            self (fitted forecaster).
        """
        returns = np.asarray(returns, dtype=float)
        self._returns = returns

        # 1. Regime detection — optionally with multivariate features
        self._detector = self._build_detector()
        obs = self._build_obs(returns)
        self._obs = obs
        self._detector.fit(obs)
        labels = self._detector.predict(obs)

        # 2. Extract transition matrix
        self._transition_matrix = self._estimate_transition_matrix(returns, labels)

        # 3. Per-regime model calibration
        regime_raw = self._get_regime_params()
        for i in range(self.n_regimes):
            mask = labels == i
            regime_returns = returns[mask]
            if len(regime_returns) < 5:
                # Too few points in this regime — use overall stats
                regime_mu = float(regime_raw[i].get("mu", np.mean(returns)))
                regime_sigma = float(regime_raw[i].get("sigma", np.std(returns)))
            else:
                regime_mu = float(regime_raw[i].get("mu", np.mean(regime_returns)))
                regime_sigma = float(regime_raw[i].get("sigma", np.std(regime_returns)))

            # Calibrate Heston params from regime data
            heston_params = self._calibrate_heston(regime_returns, regime_sigma)

            # Calibrate jump params from regime data
            jump_params = self._calibrate_jumps(regime_returns, regime_sigma)

            # Calibrate Rough Bergomi params (if enabled)
            rb_params = (
                self._calibrate_rough_bergomi(regime_returns, regime_sigma)
                if self.use_rough_bergomi
                else {}
            )

            self._calibrated[i] = CalibratedRegimeParams(
                regime_id=i,
                mu=regime_mu,
                sigma=regime_sigma,
                heston=heston_params,
                jumps=jump_params,
                rough_bergomi=rb_params,
                weight_in_regime=float(np.mean(mask)),
            )

        # 4. GARCH for conditional vol scaling
        self._garch = GARCHModel(vol_model=self.vol_model)
        self._garch.fit(returns)

        self._fitted = True
        logger.info(
            f"EnsembleForecaster fitted: detector={self.detector_type}, regimes={self.n_regimes}"
        )
        return self

    def forecast(
        self,
        horizon: int = 252,
        current_price: float = 100.0,
        dt: float = 1 / 252,
        market_data: Any = None,
        market_adjust_strength: float = 0.5,
    ) -> ForecastResult:
        """Generate a probabilistic forecast from the weighted ensemble.

        Args:
            horizon: Number of trading days to forecast.
            current_price: Starting price level.
            dt: Time step in years.
            market_data: A :class:`~synthquant.data.market_data.MarketSnapshot`
                providing VIX, VIX3M, SKEW, MOVE, HYG, IEF data aligned to
                the observation window.  When provided, regime probabilities
                are adjusted via Bayesian update before engine weight
                computation and regime parameter selection.
            market_adjust_strength: How much to trust market-implied data
                vs the HMM alone (0.0 = no adjustment, 1.0 = full update).
                Only used when *market_data* is not None.

        Returns:
            ForecastResult with blended path distribution.
        """
        self._check_fitted()

        # Current regime state (use features if available)
        obs = self._obs if self._obs is not None else self._returns
        if self.use_filtered and hasattr(self._detector, "predict_proba_filtered"):
            proba = self._detector.predict_proba_filtered(obs)
        else:
            proba = self._detector.predict_proba(obs)
        current_proba = proba[-1]

        # Bayesian adjustment using market-implied data
        hmm_proba = current_proba.copy()
        if market_data is not None:
            from synthquant.regime.market_adjuster import MarketRegimeAdjuster

            adjuster = MarketRegimeAdjuster(strength=market_adjust_strength)
            current_proba = adjuster.adjust(current_proba, market_data)

        current_regime = int(np.argmax(current_proba))

        # GARCH conditioning: differentiated per engine
        garch_scale = self._garch_scale()
        garch_v0 = self._garch_current_annual_variance()

        # Compute regime-conditional engine weights
        weights = self._compute_engine_weights(current_proba, current_regime)

        if self.use_rough_bergomi:
            # Allocate ~12% to Rough Bergomi, scale others down proportionally
            rb_frac = 0.12
            scale = 1.0 - rb_frac
            n_heston = max(int(weights["heston"] * scale * self.n_paths), 50)
            n_merton = max(int(weights["merton_jd"] * scale * self.n_paths), 50)
            n_rb = max(int(rb_frac * self.n_paths), 50)
            n_regime = self.n_paths - n_heston - n_merton - n_rb
        else:
            n_heston = max(int(weights["heston"] * self.n_paths), 100)
            n_merton = max(int(weights["merton_jd"] * self.n_paths), 100)
            n_rb = 0
            n_regime = self.n_paths - n_heston - n_merton

        rng = np.random.default_rng(self.random_state)
        seeds = rng.integers(0, 2**31, size=4)

        # --- Engine 1: Heston ---
        # v0 from GARCH current forecast (best estimate of current vol state)
        # theta from regime calibration (long-run target, NOT scaled)
        # This preserves mean-reversion structure rather than scaling uniformly
        rp = self._calibrated[current_regime]
        h = rp.heston
        heston_v0 = garch_v0 if garch_v0 is not None else h["v0"]
        heston_paths = HestonEngine().simulate(
            n_paths=n_heston,
            n_steps=horizon,
            dt=dt,
            S0=current_price,
            v0=heston_v0,
            mu=rp.mu * 252,
            kappa=h["kappa"],
            theta=h["theta"],
            sigma_v=h["sigma_v"],
            rho=h["rho"],
            random_state=int(seeds[0]),
        )

        # --- Engine 2: Merton Jump-Diffusion ---
        # The regime sigma includes ALL return variation (diffusion + jumps).
        # The Merton JD engine adds jump variance (lambda_j*(mu_j^2+sigma_j^2))
        # on top of sigma^2, so we subtract the expected jump variance to get
        # the diffusion-only sigma and avoid double-counting.
        j = rp.jumps
        total_ann_var = (rp.sigma * np.sqrt(252)) ** 2
        jump_var = j["lambda_j"] * (j["mu_j"] ** 2 + j["sigma_j"] ** 2)
        diffusion_var = max(total_ann_var - jump_var, total_ann_var * 0.25)
        merton_sigma = np.sqrt(diffusion_var)
        merton_paths = MertonJDEngine().simulate(
            n_paths=n_merton,
            n_steps=horizon,
            dt=dt,
            S0=current_price,
            mu=rp.mu * 252,
            sigma=merton_sigma,
            lambda_j=j["lambda_j"],
            mu_j=j["mu_j"],
            sigma_j=j["sigma_j"],
            random_state=int(seeds[1]),
        )

        # --- Engine 3: Regime-Switching GBM ---
        # No garch_scale: regime-switching dynamics handle vol adaptation,
        # and constant-sigma engines can't mean-revert like Heston.
        ann_params = []
        for i in range(self.n_regimes):
            cp = self._calibrated[i]
            ann_params.append(
                {
                    "mu": cp.mu * 252,
                    "sigma": cp.sigma * np.sqrt(252),
                }
            )
        assert self._transition_matrix is not None, "fit() must be called before forecast()"
        regime_paths = RegimeSwitchingEngine(
            regime_params=ann_params,
            transition_matrix=self._transition_matrix,
            initial_regime=current_regime,
        ).simulate(
            n_paths=n_regime,
            n_steps=horizon,
            dt=dt,
            S0=current_price,
            random_state=int(seeds[2]),
        )

        # --- Engine 4: Rough Bergomi (optional) ---
        if self.use_rough_bergomi and n_rb > 0:
            rb = rp.rough_bergomi
            rb_paths = RoughBergomiEngine().simulate(
                n_paths=n_rb,
                n_steps=horizon,
                dt=dt,
                S0=current_price,
                H=rb.get("H", 0.1),
                xi=rb.get("xi", 0.04),
                eta=rb.get("eta", 1.9),
                rho=rb.get("rho", -0.7),
                mu=rp.mu * 252,
                random_state=int(seeds[3]),
            )
        else:
            rb_paths = np.empty((0, horizon + 1))

        # --- Blend paths ---
        all_engine_paths = [heston_paths, merton_paths, regime_paths]
        if rb_paths.shape[0] > 0:
            all_engine_paths.append(rb_paths)
        paths = np.concatenate(all_engine_paths, axis=0)

        # Shuffle so downstream analytics don't see systematic ordering
        rng2 = np.random.default_rng(self.random_state)
        perm = rng2.permutation(len(paths))
        paths = paths[perm]

        # Standard analytics
        terminal_prices = paths[:, -1]
        terminal_returns = np.log(terminal_prices / current_price)

        var95 = value_at_risk(paths, alpha=0.05)
        cvar95 = conditional_var(paths, alpha=0.05)
        var99 = value_at_risk(paths, alpha=0.01)
        cvar99 = conditional_var(paths, alpha=0.01)
        mdd = max_drawdown_distribution(paths)

        quantiles = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
        bands = {}
        for q in quantiles:
            bands[f"p{int(q * 100):02d}"] = np.quantile(paths, q, axis=0)

        dist = EmpiricalDistribution()
        dist.fit(paths)

        # Build regime params summary for the result
        regime_params_summary = {}
        for i, cp in self._calibrated.items():
            summary: dict[str, Any] = {
                "mu": cp.mu,
                "sigma": cp.sigma,
                "heston": cp.heston,
                "jumps": cp.jumps,
                "weight": cp.weight_in_regime,
            }
            if cp.rough_bergomi:
                summary["rough_bergomi"] = cp.rough_bergomi
            regime_params_summary[i] = summary

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
            regime_params=regime_params_summary,
            horizon=horizon,
            current_price=current_price,
            metadata={
                "forecaster": "EnsembleForecaster",
                "detector": self.detector_type,
                "engine_weights": weights,
                "n_heston": n_heston,
                "n_merton": n_merton,
                "n_regime": n_regime,
                "n_rough_bergomi": n_rb,
                "use_rough_bergomi": self.use_rough_bergomi,
                "garch_scale": garch_scale,
                "dt": dt,
                "hmm_proba": hmm_proba if market_data is not None else None,
                "market_adjusted": market_data is not None,
            },
        )

        logger.info(
            f"EnsembleForecaster: horizon={horizon}, E[r]={result.expected_return:.4f}, "
            f"VaR95={var95:.4f}, weights={weights}"
        )
        return result

    def forecast_intraday(
        self,
        horizon_hours: float = 24.0,
        interval_seconds: int = 300,
        n_paths: int = 100,
        current_price: float = 100.0,
        asset_type: Literal["equity", "crypto"] = "equity",
        market_data: Any = None,
        market_adjust_strength: float = 0.5,
    ) -> ForecastResult:
        """Generate intraday probabilistic forecast matching Synth's format.

        Produces ``n_paths`` price paths at ``interval_seconds`` resolution
        over ``horizon_hours``, compatible with Synth subnet's 100-path ×
        289-point format for CRPS scoring.

        Args:
            horizon_hours: Forecast horizon in hours (default 24).
            interval_seconds: Time between price points in seconds (default 300 = 5 min).
            n_paths: Number of simulation paths (default 100, matching Synth).
            current_price: Starting price level.
            asset_type: 'equity' (252 trading days × 6.5 hours) or
                'crypto' (365.25 × 24 hours) for annualization.
            market_data: Optional MarketSnapshot for Bayesian regime adjustment.
            market_adjust_strength: Strength of market-implied adjustment.

        Returns:
            ForecastResult with paths of shape ``(n_paths, n_steps + 1)``.
        """
        n_steps = int(horizon_hours * 3600 / interval_seconds)

        seconds_per_year = 365.25 * 24 * 3600 if asset_type == "crypto" else 252 * 6.5 * 3600

        dt = interval_seconds / seconds_per_year

        # Temporarily override n_paths for this forecast
        orig_n_paths = self.n_paths
        self.n_paths = n_paths
        try:
            result = self.forecast(
                horizon=n_steps,
                current_price=current_price,
                dt=dt,
                market_data=market_data,
                market_adjust_strength=market_adjust_strength,
            )
        finally:
            self.n_paths = orig_n_paths

        # Update metadata with intraday info
        result.metadata["intraday"] = True
        result.metadata["horizon_hours"] = horizon_hours
        result.metadata["interval_seconds"] = interval_seconds
        result.metadata["asset_type"] = asset_type
        result.metadata["n_price_points"] = n_steps + 1

        return result

    def score(
        self,
        forecast_result: ForecastResult,
        actual_returns: np.ndarray,
    ) -> dict[str, Any]:
        """Score a forecast against realised returns.

        Args:
            forecast_result: A previous ForecastResult.
            actual_returns: Realised daily log returns over the horizon.

        Returns:
            Dict with CRPS, coverage, KS test, and calibration metrics.
        """
        actual = np.asarray(actual_returns, dtype=float)
        cum_actual = np.cumsum(actual)
        actual_terminal = float(cum_actual[-1]) if len(actual) > 0 else 0.0
        sim_terminal = forecast_result.terminal_returns

        forecasts_2d = sim_terminal.reshape(1, -1)
        observations_1d = np.array([actual_terminal])

        crps = ForecastScorer.crps(forecasts_2d, observations_1d)
        coverage = ForecastScorer.coverage_test(forecasts_2d, observations_1d)

        sim_daily = np.log(forecast_result.paths[:, 1:] / forecast_result.paths[:, :-1]).flatten()
        ks = ForecastScorer.ks_test(sim_daily, actual)

        realised_vol = float(np.std(actual) * np.sqrt(252))
        rv_samples = forecast_result.realized_vol_samples
        forecast_vol = float(np.mean(rv_samples))
        rv_crps = ForecastScorer.crps(rv_samples.reshape(1, -1), np.array([realised_vol]))

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
                forecast_result.quantile(0.05) <= actual_terminal <= forecast_result.quantile(0.95)
            ),
            "actual_in_50pct_interval": bool(
                forecast_result.quantile(0.25) <= actual_terminal <= forecast_result.quantile(0.75)
            ),
        }

    # ------------------------------------------------------------------ #
    # Regime-conditional model calibration
    # ------------------------------------------------------------------ #

    def _calibrate_heston(
        self, regime_returns: np.ndarray, regime_sigma: float
    ) -> dict[str, float]:
        """Calibrate Heston parameters by mapping a per-regime GARCH to Heston.

        Fits GARCH(1,1) on the regime's returns to extract:
        - v0: latest conditional variance (current vol state)
        - theta: unconditional variance (long-run target)
        - kappa: mean-reversion speed from GARCH persistence
        - sigma_v: vol-of-vol from conditional variance dynamics
        - rho: leverage effect from return-vs-variance-change correlation
        """
        ann_var = float(regime_sigma**2) * 252
        b = self.bounds

        if len(regime_returns) < b.min_regime_obs:
            return {
                "v0": ann_var,
                "theta": ann_var,
                "kappa": 3.0,
                "sigma_v": max(0.30, min(np.sqrt(ann_var) * 0.8, b.heston_sigma_v[1])),
                "rho": -0.65,
            }

        try:
            from arch import arch_model

            scaled = regime_returns * 100.0
            am = arch_model(scaled, vol="GARCH", p=1, q=1, mean="Zero", dist="normal")
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = am.fit(
                    disp="off",
                    options={"maxiter": 300},
                )

            omega = res.params.get("omega", 0.01)
            alpha = res.params.get("alpha[1]", 0.05)
            beta = res.params.get("beta[1]", 0.90)
            persistence = alpha + beta

            # theta: unconditional annualised variance
            if 0 < persistence < 0.999:
                uncond_var_daily = (omega / (1 - persistence)) / 1e4
                theta = uncond_var_daily * 252
            else:
                theta = ann_var
            theta = max(theta, 1e-6)

            # v0: last conditional variance (annualised)
            cond_vol_pct = np.asarray(res.conditional_volatility)
            v0 = (cond_vol_pct[-1] / 100.0) ** 2 * 252
            v0 = max(v0, 1e-6)

            # kappa: mean-reversion from GARCH persistence
            kappa = (1.0 - persistence) * 252
            kappa = float(np.clip(kappa, *b.heston_kappa))

            # sigma_v: vol-of-vol from conditional variance series
            cond_var_daily = (cond_vol_pct / 100.0) ** 2
            if len(cond_var_daily) > 10:
                var_of_var = float(np.var(cond_var_daily))
                mean_var = float(np.mean(cond_var_daily))
                if mean_var > 1e-12:
                    sigma_v = np.sqrt(var_of_var / mean_var) * np.sqrt(252)
                    sigma_v = float(np.clip(sigma_v, *b.heston_sigma_v))
                else:
                    sigma_v = 0.35
            else:
                sigma_v = 0.35

            # rho: leverage effect (return vs variance-change correlation)
            if len(cond_var_daily) > 10:
                dv = np.diff(cond_var_daily)
                r_align = regime_returns[: len(dv)]
                if len(r_align) == len(dv) and np.std(r_align) > 1e-10 and np.std(dv) > 1e-10:
                    rho = float(np.corrcoef(r_align, dv)[0, 1])
                    if np.isnan(rho):
                        rho = -0.65
                    rho = float(np.clip(rho, *b.heston_rho))
                else:
                    rho = -0.65
            else:
                rho = -0.65

        except (ValueError, np.linalg.LinAlgError, RuntimeError, KeyError):
            logger.warning("Heston calibration failed, using fallback params", exc_info=True)
            v0 = ann_var
            theta = ann_var
            kappa = 3.0
            sigma_v = max(0.30, min(np.sqrt(ann_var) * 0.8, 1.5))
            rho = -0.65

        # Enforce Feller condition: 2*kappa*theta > sigma_v^2
        if 2 * kappa * theta <= sigma_v**2:
            sigma_v = float(np.sqrt(b.feller_safety_factor * kappa * theta))

        return {
            "v0": float(v0),
            "theta": float(theta),
            "kappa": float(kappa),
            "sigma_v": float(sigma_v),
            "rho": float(rho),
        }

    def _calibrate_jumps(self, regime_returns: np.ndarray, regime_sigma: float) -> dict[str, float]:
        """Calibrate Merton jump parameters via outlier detection + method-of-moments.

        Combines explicit jump identification (returns beyond 2 sigma) with
        distribution shape analysis (excess kurtosis, skewness) to produce
        meaningfully fat-tailed distributions.
        """
        b = self.bounds
        if len(regime_returns) < b.min_regime_obs:
            return {
                "lambda_j": 2.0,
                "mu_j": -0.02,
                "sigma_j": max(regime_sigma * 3, 0.03),
            }

        safe_sigma = max(regime_sigma, 1e-8)
        std_returns = regime_returns / safe_sigma
        excess_kurt = float(np.mean(std_returns**4) - 3.0)
        skewness = float(np.mean(std_returns**3))

        # Explicit jump detection (lower threshold for sensitivity)
        threshold = 2.0 * safe_sigma
        jump_mask = np.abs(regime_returns) > threshold
        jump_returns = regime_returns[jump_mask]
        n_jumps = int(np.sum(jump_mask))
        n_obs = len(regime_returns)

        if n_jumps >= 5:
            lambda_j = (n_jumps / n_obs) * 252
            mu_j = float(np.mean(jump_returns))
            sigma_j = float(np.std(jump_returns))
            sigma_j = max(sigma_j, safe_sigma * 2.0)
        elif excess_kurt > 0.3:
            # Fat tails but few explicit outliers -> method-of-moments
            mu_j = skewness * safe_sigma * 0.5 if skewness < -0.1 else -0.01
            lambda_j = max(2.0, min(excess_kurt * 1.5, 15.0))
            sigma_j = safe_sigma * max(2.0, np.sqrt(max(excess_kurt, 0.5)))
        else:
            # Near-Gaussian: baseline jump component for tail risk
            lambda_j = 1.0
            mu_j = -0.005
            sigma_j = safe_sigma * 2.5

        lambda_j = float(np.clip(lambda_j, *b.jump_lambda))
        mu_j = float(np.clip(mu_j, *b.jump_mu))
        sigma_j = float(np.clip(sigma_j, safe_sigma * b.jump_sigma_floor_mult, b.jump_sigma_ceil))

        return {
            "lambda_j": lambda_j,
            "mu_j": mu_j,
            "sigma_j": sigma_j,
        }

    def _calibrate_rough_bergomi(
        self, regime_returns: np.ndarray, regime_sigma: float
    ) -> dict[str, float]:
        """Calibrate Rough Bergomi parameters from regime returns.

        Estimates the Hurst exponent H via variogram method (log-log regression
        of realized variance at different time scales) and maps regime volatility
        to Rough Bergomi parameters.

        Args:
            regime_returns: Returns within this regime.
            regime_sigma: Daily volatility of this regime.

        Returns:
            Dict with keys: H, xi, eta, rho.
        """
        n = len(regime_returns)
        if n < self.bounds.min_regime_obs:
            return {"H": 0.1, "xi": regime_sigma**2, "eta": 1.9, "rho": -0.7}

        # Variogram method for Hurst exponent estimation
        # RV at scale q scales as q^(2H): log(RV_q) = 2H * log(q) + const
        scales = [1, 2, 5, 10, 21]
        log_scales = []
        log_rvs = []

        for q in scales:
            if n < q * 3:
                continue
            # Aggregate returns at scale q
            n_blocks = n // q
            if n_blocks < 3:
                continue
            agg = np.sum(regime_returns[: n_blocks * q].reshape(n_blocks, q), axis=1)
            rv_q = float(np.mean(agg**2))
            if rv_q > 1e-20:
                log_scales.append(np.log(q))
                log_rvs.append(np.log(rv_q))

        if len(log_scales) >= 3:
            # OLS: log(RV_q) = 2H * log(q) + const
            log_scales_arr = np.array(log_scales)
            log_rvs_arr = np.array(log_rvs)
            slope = float(np.polyfit(log_scales_arr, log_rvs_arr, 1)[0])
            H_est = max(0.01, min(slope / 2, 0.49))
        else:
            H_est = 0.1

        # Map regime vol to Rough Bergomi parameters
        xi = regime_sigma**2  # forward variance ~ realized variance
        eta = float(np.clip(1.0 / (H_est + 0.1), 1.0, 3.0))  # higher eta for rougher

        # Leverage: return-vs-|return| correlation as proxy
        if n > 10:
            rho_est = float(np.corrcoef(regime_returns[1:], np.abs(regime_returns[:-1]))[0, 1])
            if np.isnan(rho_est):
                rho_est = -0.7
            rho_est = float(np.clip(-abs(rho_est) - 0.3, -0.95, -0.3))
        else:
            rho_est = -0.7

        return {"H": H_est, "xi": xi, "eta": eta, "rho": rho_est}

    # ------------------------------------------------------------------ #
    # Engine weighting
    # ------------------------------------------------------------------ #

    def _compute_engine_weights(
        self,
        regime_proba: np.ndarray,
        current_regime: int,
    ) -> dict[str, float]:
        """Compute regime-conditional engine weights.

        Uses linear interpolation between two extremes:
        - Full calm:   Heston 45%, Merton 15%, RS-GBM 40%
        - Full crisis: Heston 20%, Merton 50%, RS-GBM 30%
        This produces meaningful differentiation: in crisis, Merton JD
        dominates with jump dynamics; in calm, Heston dominates with
        stochastic vol.
        """
        # Identify which regime is "crisis" (highest sigma)
        if self.n_regimes == 2:
            sigmas = [self._calibrated[i].sigma for i in range(self.n_regimes)]
            crisis_regime = int(np.argmax(sigmas))
            calm_regime = 1 - crisis_regime

            crisis_prob = float(regime_proba[crisis_regime])
            calm_prob = float(regime_proba[calm_regime])
        else:
            mean_sigma = np.mean([self._calibrated[i].sigma for i in range(self.n_regimes)])
            current_sigma = self._calibrated[current_regime].sigma
            crisis_prob = float(np.clip(current_sigma / mean_sigma - 0.5, 0, 1))
            calm_prob = 1.0 - crisis_prob

        # Linear interpolation: calm -> crisis
        w_heston = 0.45 * calm_prob + 0.20 * crisis_prob
        w_merton = 0.15 * calm_prob + 0.50 * crisis_prob
        w_regime = 0.40 * calm_prob + 0.30 * crisis_prob

        total = w_heston + w_merton + w_regime
        return {
            "heston": w_heston / total,
            "merton_jd": w_merton / total,
            "regime_switching": w_regime / total,
        }

    def optimize_engine_weights(
        self,
        returns: np.ndarray,
        *,
        n_folds: int = 5,
        train_frac: float = 0.7,
        horizon: int = 21,
        current_price: float = 100.0,
    ) -> dict[str, np.ndarray]:
        """Find CRPS-optimal engine weights via walk-forward cross-validation.

        Runs each engine separately on validation folds, then finds weights
        that minimize composite CRPS using Nelder-Mead optimization.

        Args:
            returns: Full return series for cross-validation.
            n_folds: Number of walk-forward folds.
            train_frac: Fraction of each fold used for training.
            horizon: Forecast horizon per fold (in steps).
            current_price: Starting price for forecasts.

        Returns:
            Dict with 'optimal_weights' (array), 'crps_by_fold' (array),
            and 'engine_names' (list).
        """
        from scipy.optimize import minimize

        r = np.asarray(returns, dtype=float).ravel()
        n = len(r)
        fold_size = n // n_folds

        engine_names = ["heston", "merton_jd", "regime_switching"]
        if self.use_rough_bergomi:
            engine_names.append("rough_bergomi")
        n_engines = len(engine_names)

        fold_crps = np.zeros((n_folds, n_engines))

        for fold in range(n_folds):
            start = fold * fold_size
            end = min(start + fold_size, n)
            train_end = start + int((end - start) * train_frac)

            if train_end - start < 60 or end - train_end < horizon:
                continue

            train_r = r[start:train_end]
            val_r = r[train_end : train_end + horizon]

            # Fit on training fold
            self.fit(train_r)
            actual_terminal = float(np.sum(val_r))

            # Run each engine separately and compute CRPS
            rp = self._calibrated[int(np.argmax(self._detector.predict_proba(self._obs)[-1]))]
            dt = 1 / 252

            rng = np.random.default_rng(self.random_state)
            seeds = rng.integers(0, 2**31, size=4)

            h = rp.heston
            garch_v0 = self._garch_current_annual_variance()
            hv0 = garch_v0 if garch_v0 is not None else h["v0"]

            engines_paths = []
            # Heston
            engines_paths.append(
                HestonEngine().simulate(
                    n_paths=500,
                    n_steps=horizon,
                    dt=dt,
                    S0=current_price,
                    v0=hv0,
                    mu=rp.mu * 252,
                    kappa=h["kappa"],
                    theta=h["theta"],
                    sigma_v=h["sigma_v"],
                    rho=h["rho"],
                    random_state=int(seeds[0]),
                )
            )
            # Merton
            j = rp.jumps
            tv = (rp.sigma * np.sqrt(252)) ** 2
            jv = j["lambda_j"] * (j["mu_j"] ** 2 + j["sigma_j"] ** 2)
            ms = np.sqrt(max(tv - jv, tv * 0.25))
            engines_paths.append(
                MertonJDEngine().simulate(
                    n_paths=500,
                    n_steps=horizon,
                    dt=dt,
                    S0=current_price,
                    mu=rp.mu * 252,
                    sigma=ms,
                    lambda_j=j["lambda_j"],
                    mu_j=j["mu_j"],
                    sigma_j=j["sigma_j"],
                    random_state=int(seeds[1]),
                )
            )
            # Regime-Switching
            ann_params = [
                {
                    "mu": self._calibrated[i].mu * 252,
                    "sigma": self._calibrated[i].sigma * np.sqrt(252),
                }
                for i in range(self.n_regimes)
            ]
            assert self._transition_matrix is not None
            engines_paths.append(
                RegimeSwitchingEngine(
                    regime_params=ann_params,
                    transition_matrix=self._transition_matrix,
                ).simulate(
                    n_paths=500,
                    n_steps=horizon,
                    dt=dt,
                    S0=current_price,
                    random_state=int(seeds[2]),
                )
            )
            # Rough Bergomi (if enabled)
            if self.use_rough_bergomi:
                rb = rp.rough_bergomi
                engines_paths.append(
                    RoughBergomiEngine().simulate(
                        n_paths=500,
                        n_steps=horizon,
                        dt=dt,
                        S0=current_price,
                        H=rb.get("H", 0.1),
                        xi=rb.get("xi", 0.04),
                        eta=rb.get("eta", 1.9),
                        rho=rb.get("rho", -0.7),
                        mu=rp.mu * 252,
                        random_state=int(seeds[3]),
                    )
                )

            for eng_idx, eng_paths in enumerate(engines_paths):
                terminal_rets = np.log(eng_paths[:, -1] / current_price)
                mae = float(np.mean(np.abs(terminal_rets - actual_terminal)))
                spread = float(np.mean(np.abs(terminal_rets[:, None] - terminal_rets[None, :]))) / 2
                fold_crps[fold, eng_idx] = mae - spread

        # Optimize weights to minimize weighted CRPS
        def objective(w: np.ndarray) -> float:
            w_pos = np.abs(w)
            w_norm = w_pos / w_pos.sum()
            return float(np.mean(fold_crps @ w_norm))

        x0 = np.ones(n_engines) / n_engines
        result = minimize(
            objective, x0, method="Nelder-Mead", options={"maxiter": 500, "xatol": 1e-4}
        )

        optimal = np.abs(result.x)
        optimal /= optimal.sum()

        # Store for use in _compute_engine_weights
        self._optimal_weights = dict(zip(engine_names, optimal, strict=False))
        logger.info(f"CRPS-optimal weights: {self._optimal_weights}")

        return {
            "optimal_weights": optimal,
            "crps_by_fold": fold_crps,
            "engine_names": engine_names,
        }

    # ------------------------------------------------------------------ #
    # Shared helpers
    # ------------------------------------------------------------------ #

    def _garch_scale(self) -> float:
        """GARCH conditional vol / unconditional vol ratio."""
        if self._garch is None or self._garch._result is None:
            return 1.0
        garch_vol = self._garch.forecast(horizon=1)
        cond_vol_daily = garch_vol[0] / np.sqrt(252)
        uncond_vol = float(np.std(self._returns))
        scale = cond_vol_daily / uncond_vol if uncond_vol > 1e-10 else 1.0
        return float(np.clip(scale, *self.bounds.garch_scale_clamp))

    def _garch_current_annual_variance(self) -> float | None:
        """Return GARCH 1-step forecast as annualised variance.

        Used to set Heston v0 to the best available estimate of the
        current instantaneous variance, without uniformly scaling all
        engine parameters.
        """
        if self._garch is None or self._garch._result is None:
            return None
        try:
            vol_forecast = self._garch.forecast(horizon=1)
            return float(vol_forecast[0] ** 2)  # annualised vol^2 = annualised var
        except (ValueError, np.linalg.LinAlgError, RuntimeError):
            logger.warning("GARCH variance forecast failed", exc_info=True)
            return None

    def _get_regime_params(self) -> dict[int, dict[str, Any]]:
        """Extract and normalise regime params from the detector."""
        raw = self._detector.get_regime_params()
        params: dict[int, dict[str, Any]] = {}
        for i, rp in raw.items():
            if "mu" in rp and "sigma" in rp:
                params[i] = {"mu": float(rp["mu"]), "sigma": float(rp["sigma"])}
            elif "means" in rp:
                mu = float(rp["means"][0]) if isinstance(rp["means"], list) else float(rp["means"])
                cov = rp.get("covariances", rp.get("sigma", 0.01))
                if isinstance(cov, list):
                    sigma = float(np.sqrt(abs(cov[0][0] if isinstance(cov[0], list) else cov[0])))
                else:
                    sigma = float(np.sqrt(abs(cov)))
                params[i] = {"mu": mu, "sigma": sigma}
            else:
                mu = float(rp.get("mu", rp.get("mean", 0.0)))
                sigma = float(rp.get("sigma", rp.get("std", 0.01)))
                params[i] = {"mu": mu, "sigma": sigma}
        return params

    def _build_detector(self) -> Any:
        if self.detector_type == "hmm":
            return HMMRegimeDetector(n_components=self.n_regimes, random_state=self.random_state)
        elif self.detector_type == "ms_garch":
            return MarkovSwitchingGARCH(k_regimes=self.n_regimes)
        elif self.detector_type == "ensemble":
            return EnsembleRegimeDetector(
                detectors=[
                    HMMRegimeDetector(n_components=self.n_regimes, random_state=self.random_state),
                    MarkovSwitchingGARCH(k_regimes=self.n_regimes),
                ],
                method="proba_average",
                n_regimes=self.n_regimes,
            )
        else:
            raise ValueError(f"Unknown detector type: {self.detector_type}")

    def _build_obs(self, returns: np.ndarray) -> np.ndarray:
        """Build the observation matrix for the regime detector.

        If ``use_features`` is True and the detector is HMM, this
        constructs a multivariate feature matrix.  Otherwise returns
        the raw 1-D array.
        """
        if self.use_features and self.detector_type in ("hmm", "ensemble"):
            from synthquant.regime.features import build_hmm_features

            return build_hmm_features(returns, feature_names=self.feature_names)
        return returns

    def _estimate_transition_matrix(self, returns: np.ndarray, labels: np.ndarray) -> np.ndarray:
        if hasattr(self._detector, "_model") and hasattr(
            getattr(self._detector, "_model", None), "transmat_"
        ):
            return np.array(self._detector._model.transmat_)
        n = self.n_regimes
        trans = np.zeros((n, n))
        for t in range(len(labels) - 1):
            trans[labels[t], labels[t + 1]] += 1
        trans += 1e-6
        return trans / trans.sum(axis=1, keepdims=True)

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("Forecaster has not been fitted. Call fit() first.")
