"""Tests for model calibration (MLE and Bayesian)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from augury.models.calibration import (
    BayesianCalibrator,
    CalibratedParams,
    Calibrator,
    MLECalibrator,
)

# ─── CalibratedParams dataclass ───────────────────────────────────────


def test_calibrated_params_stores_values() -> None:
    """CalibratedParams stores all provided fields."""
    cp = CalibratedParams(
        params={"mu": 0.05, "sigma": 0.2},
        method="MLE",
        log_likelihood=-100.0,
        convergence_info={"success": True},
    )
    assert cp.params["mu"] == 0.05
    assert cp.method == "MLE"
    assert cp.log_likelihood == -100.0
    assert cp.convergence_info["success"] is True


def test_calibrated_params_defaults() -> None:
    """CalibratedParams has sensible defaults."""
    cp = CalibratedParams(params={"x": 1.0}, method="test")
    assert cp.log_likelihood is None
    assert cp.convergence_info == {}
    assert cp.posterior_samples is None


# ─── Abstract base class ──────────────────────────────────────────────


def test_calibrator_is_abstract() -> None:
    """Cannot instantiate Calibrator directly."""
    with pytest.raises(TypeError):
        Calibrator()


# ─── MLECalibrator ─────────────────────────────────────────────────────


class TestMLECalibrator:
    def test_calibrate_quadratic(self) -> None:
        """MLE finds the optimum of a simple quadratic log-likelihood."""

        def ll(mu: float, sigma: float) -> float:
            return -((mu - 3.0) ** 2) - (sigma - 1.5) ** 2

        cal = MLECalibrator()
        result = cal.calibrate(
            log_likelihood_fn=ll,
            initial_params={"mu": 0.0, "sigma": 0.5},
        )
        assert isinstance(result, CalibratedParams)
        assert result.method == "MLE"
        np.testing.assert_allclose(result.params["mu"], 3.0, atol=0.01)
        np.testing.assert_allclose(result.params["sigma"], 1.5, atol=0.01)

    def test_calibrate_with_bounds(self) -> None:
        """MLE respects parameter bounds."""

        def ll(x: float) -> float:
            return -((x - 10.0) ** 2)  # optimum at 10.0

        cal = MLECalibrator()
        result = cal.calibrate(
            log_likelihood_fn=ll,
            initial_params={"x": 1.0},
            bounds={"x": (0.0, 5.0)},  # constrain to [0, 5]
        )
        # Should be at the upper bound since optimum is outside
        np.testing.assert_allclose(result.params["x"], 5.0, atol=0.01)

    def test_calibrate_returns_log_likelihood(self) -> None:
        """Result includes the log-likelihood at the optimum."""

        def ll(x: float) -> float:
            return -((x - 2.0) ** 2)

        cal = MLECalibrator()
        result = cal.calibrate(ll, initial_params={"x": 0.0})
        assert result.log_likelihood is not None
        np.testing.assert_allclose(result.log_likelihood, 0.0, atol=0.01)

    def test_calibrate_convergence_info(self) -> None:
        """Result includes convergence metadata."""

        def ll(x: float) -> float:
            return -((x - 1.0) ** 2)

        cal = MLECalibrator()
        result = cal.calibrate(ll, initial_params={"x": 0.0})
        assert "success" in result.convergence_info
        assert "n_eval" in result.convergence_info
        assert result.convergence_info["success"] is True

    def test_calibrate_normal_distribution(self) -> None:
        """Recover mu and sigma from normal samples via MLE."""
        rng = np.random.default_rng(42)
        true_mu, true_sigma = 2.0, 0.5
        data = rng.normal(true_mu, true_sigma, 5000)

        def normal_ll(mu: float, sigma: float) -> float:
            if sigma <= 0:
                return -1e10
            n = len(data)
            return -0.5 * n * np.log(2 * np.pi * sigma**2) - np.sum((data - mu) ** 2) / (
                2 * sigma**2
            )

        cal = MLECalibrator()
        result = cal.calibrate(
            log_likelihood_fn=normal_ll,
            initial_params={"mu": 0.0, "sigma": 1.0},
            bounds={"mu": (-10.0, 10.0), "sigma": (0.01, 10.0)},
        )
        np.testing.assert_allclose(result.params["mu"], true_mu, atol=0.05)
        np.testing.assert_allclose(result.params["sigma"], true_sigma, atol=0.05)

    def test_custom_method(self) -> None:
        """Can use a different optimization method."""

        def ll(x: float) -> float:
            return -((x - 1.0) ** 2)

        cal = MLECalibrator(method="Nelder-Mead")
        result = cal.calibrate(ll, initial_params={"x": 5.0})
        np.testing.assert_allclose(result.params["x"], 1.0, atol=0.01)

    def test_multi_parameter_optimization(self) -> None:
        """Optimizes over multiple parameters simultaneously."""

        def ll(a: float, b: float, c: float) -> float:
            return -((a - 1.0) ** 2) - (b - 2.0) ** 2 - (c - 3.0) ** 2

        cal = MLECalibrator()
        result = cal.calibrate(ll, initial_params={"a": 0.0, "b": 0.0, "c": 0.0})
        np.testing.assert_allclose(result.params["a"], 1.0, atol=0.01)
        np.testing.assert_allclose(result.params["b"], 2.0, atol=0.01)
        np.testing.assert_allclose(result.params["c"], 3.0, atol=0.01)


# ─── BayesianCalibrator ───────────────────────────────────────────────


def test_bayesian_calibrator_requires_pymc() -> None:
    """BayesianCalibrator can be instantiated but calibrate() needs PyMC."""
    cal = BayesianCalibrator()
    assert cal.n_draws == 1000
    assert cal.n_chains == 2
    # We don't actually run calibrate() here since PyMC may not be installed.
    # The constructor should always work.


def test_bayesian_calibrator_custom_params() -> None:
    """BayesianCalibrator stores custom n_draws, n_chains, target_accept."""
    cal = BayesianCalibrator(n_draws=500, n_chains=4, target_accept=0.95)
    assert cal.n_draws == 500
    assert cal.n_chains == 4
    assert cal.target_accept == 0.95


class TestBayesianCalibratorWithMock:
    """Test BayesianCalibrator.calibrate() with mocked PyMC."""

    def _build_mocks(self, param_names: list[str]) -> tuple[MagicMock, MagicMock]:
        """Create mock pymc and pytensor modules with working Model context."""
        mock_pm = MagicMock()
        mock_pt = MagicMock()

        # pm.Model() needs to work as a context manager
        mock_model = MagicMock()
        mock_pm.Model.return_value.__enter__ = MagicMock(return_value=mock_model)
        mock_pm.Model.return_value.__exit__ = MagicMock(return_value=False)

        # pm.Uniform / pm.Normal return mock tensors
        for name in param_names:
            mock_pm.Uniform.return_value = MagicMock(name=name)
            mock_pm.Normal.return_value = MagicMock(name=name)

        # pm.potential is a decorator — just return the function
        mock_pm.potential = lambda fn: fn

        # Build a fake trace with posterior samples
        posterior_data = {}
        rng = np.random.default_rng(42)
        for name in param_names:
            posterior_data[name] = MagicMock()
            posterior_data[name].values.flatten.return_value = rng.normal(1.0, 0.1, 100)

        mock_trace = MagicMock()
        mock_trace.posterior.__getitem__ = lambda self, key: posterior_data[key]
        mock_pm.sample.return_value = mock_trace

        return mock_pm, mock_pt

    def test_calibrate_returns_bayesian_result(self) -> None:
        """BayesianCalibrator.calibrate returns CalibratedParams with method='Bayesian'."""
        mock_pm, mock_pt = self._build_mocks(["mu", "sigma"])

        with patch.dict(
            "sys.modules", {"pymc": mock_pm, "pytensor": MagicMock(), "pytensor.tensor": mock_pt}
        ):
            cal = BayesianCalibrator(n_draws=100, n_chains=1)
            result = cal.calibrate(
                log_likelihood_fn=lambda mu, sigma: -((mu - 1.0) ** 2) - (sigma - 0.5) ** 2,
                initial_params={"mu": 0.0, "sigma": 1.0},
            )

        assert isinstance(result, CalibratedParams)
        assert result.method == "Bayesian"
        assert "mu" in result.params
        assert "sigma" in result.params
        assert result.posterior_samples is not None
        assert "mu" in result.posterior_samples
        assert len(result.posterior_samples["mu"]) == 100

    def test_calibrate_with_bounds_uses_uniform(self) -> None:
        """When bounds are finite, BayesianCalibrator uses pm.Uniform priors."""
        mock_pm, mock_pt = self._build_mocks(["x"])

        with patch.dict(
            "sys.modules", {"pymc": mock_pm, "pytensor": MagicMock(), "pytensor.tensor": mock_pt}
        ):
            cal = BayesianCalibrator(n_draws=50, n_chains=1)
            cal.calibrate(
                log_likelihood_fn=lambda x: -((x - 1.0) ** 2),
                initial_params={"x": 0.5},
                bounds={"x": (0.0, 2.0)},
            )

        mock_pm.Uniform.assert_called_once_with("x", lower=0.0, upper=2.0)

    def test_calibrate_without_bounds_uses_normal(self) -> None:
        """Without bounds, BayesianCalibrator uses pm.Normal priors."""
        mock_pm, mock_pt = self._build_mocks(["x"])

        with patch.dict(
            "sys.modules", {"pymc": mock_pm, "pytensor": MagicMock(), "pytensor.tensor": mock_pt}
        ):
            cal = BayesianCalibrator(n_draws=50, n_chains=1)
            cal.calibrate(
                log_likelihood_fn=lambda x: -((x - 1.0) ** 2),
                initial_params={"x": 2.0},
            )

        # For unbounded param, Normal prior with mu=initial and sigma=|initial|+1
        mock_pm.Normal.assert_called_once_with("x", mu=2.0, sigma=3.0)

    def test_calibrate_calls_pm_sample_with_correct_args(self) -> None:
        """pm.sample is called with the configured n_draws, n_chains, target_accept."""
        mock_pm, mock_pt = self._build_mocks(["a"])

        with patch.dict(
            "sys.modules", {"pymc": mock_pm, "pytensor": MagicMock(), "pytensor.tensor": mock_pt}
        ):
            cal = BayesianCalibrator(n_draws=200, n_chains=3, target_accept=0.85)
            cal.calibrate(
                log_likelihood_fn=lambda a: -(a**2),
                initial_params={"a": 0.0},
            )

        mock_pm.sample.assert_called_once()
        call_kwargs = mock_pm.sample.call_args[1]
        assert call_kwargs["draws"] == 200
        assert call_kwargs["chains"] == 3
        assert call_kwargs["target_accept"] == 0.85
        assert call_kwargs["progressbar"] is False

    def test_calibrate_importerror_when_pymc_missing(self) -> None:
        """calibrate() raises ImportError with helpful message when PyMC is absent."""
        import sys

        # Temporarily remove pymc from sys.modules if present, and make import fail
        saved = sys.modules.get("pymc")
        sys.modules["pymc"] = None  # type: ignore[assignment]
        try:
            cal = BayesianCalibrator()
            with pytest.raises(ImportError, match="PyMC is required"):
                cal.calibrate(
                    log_likelihood_fn=lambda x: -(x**2),
                    initial_params={"x": 0.0},
                )
        finally:
            if saved is not None:
                sys.modules["pymc"] = saved
            else:
                sys.modules.pop("pymc", None)
