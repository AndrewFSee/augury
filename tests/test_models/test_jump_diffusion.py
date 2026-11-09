"""Tests for Merton jump-diffusion and Kou double-exponential models."""

from __future__ import annotations

import numpy as np
import pytest

from augury.models.jump_diffusion import KouModel, MertonJumpDiffusion

# ─── Merton Jump-Diffusion ────────────────────────────────────────────


class TestMertonJumpDiffusion:
    @pytest.fixture()
    def model(self) -> MertonJumpDiffusion:
        return MertonJumpDiffusion()

    def test_simulate_shape(self, model) -> None:
        """Output has shape (n_paths, n_steps+1)."""
        paths = model.simulate(S0=100.0, T=1.0, n_paths=100, n_steps=50, random_state=42)
        assert paths.shape == (100, 51)

    def test_initial_price(self, model) -> None:
        """All paths start at S0."""
        S0 = 120.0
        paths = model.simulate(S0=S0, T=1.0, n_paths=200, n_steps=10, random_state=42)
        np.testing.assert_allclose(paths[:, 0], S0)

    def test_all_prices_positive(self, model) -> None:
        """Price paths are strictly positive."""
        paths = model.simulate(S0=100.0, T=1.0, n_paths=500, n_steps=50, random_state=42)
        assert np.all(paths > 0)

    def test_reproducibility(self, model) -> None:
        """Same seed gives identical paths."""
        p1 = model.simulate(S0=100.0, T=1.0, n_paths=50, n_steps=30, random_state=7)
        p2 = model.simulate(S0=100.0, T=1.0, n_paths=50, n_steps=30, random_state=7)
        np.testing.assert_array_equal(p1, p2)

    def test_different_seeds_differ(self, model) -> None:
        """Different seeds produce different paths."""
        p1 = model.simulate(S0=100.0, T=1.0, n_paths=20, n_steps=20, random_state=1)
        p2 = model.simulate(S0=100.0, T=1.0, n_paths=20, n_steps=20, random_state=2)
        assert not np.array_equal(p1, p2)

    def test_zero_jump_intensity_resembles_gbm(self) -> None:
        """lambda_j=0 reduces to GBM-like dynamics."""
        m = MertonJumpDiffusion(lambda_j=0.0)
        paths = m.simulate(S0=100.0, T=1.0, n_paths=5000, n_steps=252, random_state=42)
        log_returns = np.diff(np.log(paths), axis=1).flatten()
        # Should be approximately normal with no excess kurtosis from jumps
        assert abs(float(np.mean(log_returns))) < 0.01

    def test_higher_jump_intensity_increases_kurtosis(self) -> None:
        """More jumps produce fatter tails (higher kurtosis)."""
        from scipy.stats import kurtosis

        p_low = MertonJumpDiffusion(lambda_j=0.5).simulate(
            S0=100.0, T=1.0, n_paths=5000, n_steps=252, random_state=42
        )
        p_high = MertonJumpDiffusion(lambda_j=5.0).simulate(
            S0=100.0, T=1.0, n_paths=5000, n_steps=252, random_state=42
        )
        rets_low = np.diff(np.log(p_low), axis=1).flatten()
        rets_high = np.diff(np.log(p_high), axis=1).flatten()
        assert kurtosis(rets_high) > kurtosis(rets_low)

    def test_negative_mean_jump_lowers_terminal(self) -> None:
        """Negative mu_j should lower the average terminal price."""
        p_neg = MertonJumpDiffusion(mu_j=-0.10, lambda_j=2.0).simulate(
            S0=100.0, T=1.0, n_paths=3000, n_steps=252, random_state=42
        )
        p_pos = MertonJumpDiffusion(mu_j=0.10, lambda_j=2.0).simulate(
            S0=100.0, T=1.0, n_paths=3000, n_steps=252, random_state=42
        )
        assert p_neg[:, -1].mean() < p_pos[:, -1].mean()

    # ─── Closed-form pricing ──────────────────────────────────────────

    def test_european_call_non_negative(self, model) -> None:
        """Call price is non-negative."""
        price = model.european_call_price(S0=100.0, K=100.0, T=1.0, r=0.05)
        assert price >= 0

    def test_european_call_deep_itm(self) -> None:
        """Deep ITM call approaches intrinsic value."""
        m = MertonJumpDiffusion(sigma=0.01, lambda_j=0.0)
        price = m.european_call_price(S0=150.0, K=100.0, T=1.0, r=0.05)
        intrinsic = 150.0 - 100.0 * np.exp(-0.05)
        np.testing.assert_allclose(price, intrinsic, atol=2.0)

    def test_european_call_increases_with_spot(self) -> None:
        """Higher S0 -> higher call price."""
        m = MertonJumpDiffusion()
        low = m.european_call_price(S0=90.0, K=100.0, T=1.0, r=0.05)
        high = m.european_call_price(S0=110.0, K=100.0, T=1.0, r=0.05)
        assert high > low

    def test_european_call_increases_with_vol(self) -> None:
        """Higher sigma -> higher call price (vega positive)."""
        low = MertonJumpDiffusion(sigma=0.10).european_call_price(S0=100.0, K=100.0, T=1.0, r=0.05)
        high = MertonJumpDiffusion(sigma=0.40).european_call_price(S0=100.0, K=100.0, T=1.0, r=0.05)
        assert high > low

    def test_kappa_j_compensator(self) -> None:
        """Internal _kappa_j is calculated correctly."""
        m = MertonJumpDiffusion(mu_j=-0.05, sigma_j=0.10)
        expected = np.exp(-0.05 + 0.5 * 0.10**2) - 1
        np.testing.assert_allclose(m._kappa_j, expected)


# ─── Kou Model ─────────────────────────────────────────────────────────


class TestKouModel:
    @pytest.fixture()
    def model(self) -> KouModel:
        return KouModel()

    def test_simulate_shape(self, model) -> None:
        paths = model.simulate(S0=100.0, T=1.0, n_paths=100, n_steps=50, random_state=42)
        assert paths.shape == (100, 51)

    def test_initial_price(self, model) -> None:
        S0 = 80.0
        paths = model.simulate(S0=S0, T=1.0, n_paths=200, n_steps=10, random_state=42)
        np.testing.assert_allclose(paths[:, 0], S0)

    def test_all_prices_positive(self, model) -> None:
        paths = model.simulate(S0=100.0, T=1.0, n_paths=500, n_steps=50, random_state=42)
        assert np.all(paths > 0)

    def test_reproducibility(self, model) -> None:
        p1 = model.simulate(S0=100.0, T=1.0, n_paths=50, n_steps=30, random_state=7)
        p2 = model.simulate(S0=100.0, T=1.0, n_paths=50, n_steps=30, random_state=7)
        np.testing.assert_array_equal(p1, p2)

    def test_different_seeds_differ(self, model) -> None:
        p1 = model.simulate(S0=100.0, T=1.0, n_paths=20, n_steps=20, random_state=1)
        p2 = model.simulate(S0=100.0, T=1.0, n_paths=20, n_steps=20, random_state=2)
        assert not np.array_equal(p1, p2)

    def test_asymmetric_jumps(self) -> None:
        """Mostly downward jumps (low p_up) should lower terminal mean."""
        p_down = KouModel(p_up=0.1, lambda_j=3.0).simulate(
            S0=100.0, T=1.0, n_paths=10000, n_steps=252, random_state=42
        )
        p_up = KouModel(p_up=0.9, lambda_j=3.0).simulate(
            S0=100.0, T=1.0, n_paths=10000, n_steps=252, random_state=42
        )
        assert p_down[:, -1].mean() < p_up[:, -1].mean()

    def test_kappa_j_compensator(self) -> None:
        """Internal _kappa_j is calculated correctly."""
        m = KouModel(p_up=0.4, eta1=10.0, eta2=5.0)
        expected = 0.4 * 10.0 / 9.0 + 0.6 * 5.0 / 6.0 - 1
        np.testing.assert_allclose(m._kappa_j, expected, rtol=1e-10)
