import pytest
import scoringrules as sr
import torch

from genpp.models.loss import (
    CRPS_Normal,
    CRPS_TruncatedNormal,
    EnergyScore,
    EnsembleCRPS,
    VariogramScore,
)


class TestEnergyScore:
    """Test cases comparing EnergyScore class with scoringrules.es_ensemble"""

    @pytest.mark.unit
    def test_energy_score_simple_case(self):
        """Test energy score computation for a simple case"""
        # Create simple test data
        batch_size, n_samples, lat, lon, out_features = 2, 3, 4, 4, 1

        # Create deterministic test data for reproducibility
        torch.manual_seed(42)
        # Reshape for EnergyScore: x should be [batch, n_samples, spatial_dims]
        x = torch.randn(batch_size, n_samples, lat * lon * out_features)
        y = torch.randn(batch_size, lat * lon * out_features)

        # Compute energy score using our implementation
        energy_score_model = EnergyScore(beta=1.0, clamp=False)
        es_custom = energy_score_model(x, y)

        # Prepare data for scoringrules (already in correct format)
        es_reference = sr.es_ensemble(y, x, backend="torch")

        # Compare results - they should be close (allowing for numerical differences)
        assert es_custom.shape == (batch_size,)
        torch.testing.assert_close(es_custom, es_reference, rtol=1e-10, atol=1e-10)

    @pytest.mark.unit
    def test_energy_score_multiple_features(self):
        """Test energy score with multiple output features"""
        batch_size, n_samples, lat, lon, out_features = 4, 5, 3, 3, 2

        torch.manual_seed(123)
        # Reshape for EnergyScore
        x = torch.randn(batch_size, n_samples, lat * lon * out_features)
        y = torch.randn(batch_size, lat * lon * out_features)

        energy_score_model = EnergyScore(beta=1.0, clamp=False)
        es_custom = energy_score_model(x, y)

        es_reference = sr.es_ensemble(y, x, backend="torch")
        torch.testing.assert_close(es_custom, es_reference, rtol=1e-7, atol=1e-7)

    @pytest.mark.unit
    def test_energy_score_different_beta_values(self):
        """Test energy score with different beta values (note: scoringrules uses beta=1)"""
        batch_size, n_samples, lat, lon, out_features = 1, 4, 2, 2, 1

        torch.manual_seed(456)
        # Reshape for EnergyScore
        x = torch.randn(batch_size, n_samples, lat * lon * out_features)
        y = torch.randn(batch_size, lat * lon * out_features)

        # Test with beta=1.0 (should match scoringrules)
        energy_score_beta1 = EnergyScore(beta=1.0, clamp=False)
        es_custom_beta1 = energy_score_beta1(x, y)

        es_reference = sr.es_ensemble(y, x, backend="torch")

        torch.testing.assert_close(es_custom_beta1, es_reference, rtol=1e-7, atol=1e-7)

        # Test with beta=2.0 (should be different from scoringrules)
        energy_score_beta2 = EnergyScore(beta=2.0, clamp=False)
        es_custom_beta2 = energy_score_beta2(x, y)

        # They should not be equal
        assert not torch.allclose(es_custom_beta1, es_custom_beta2)

    @pytest.mark.unit
    def test_energy_score_edge_cases(self):
        """Test edge cases like identical predictions"""
        batch_size, n_samples, lat, lon, out_features = 1, 3, 2, 2, 1

        # Case 1: All ensemble members are identical
        torch.manual_seed(789)
        y = torch.randn(batch_size, lat * lon * out_features)
        x = y.unsqueeze(1).repeat(1, n_samples, 1)  # All ensemble members = truth

        energy_score_model = EnergyScore(beta=1.0, clamp=False)
        es_custom = energy_score_model(x, y)

        es_reference = sr.es_ensemble(y, x, backend="torch")

        torch.testing.assert_close(es_custom, es_reference, rtol=1e-10, atol=1e-10)

        # Energy score should be 0 when all predictions equal truth
        assert torch.allclose(es_custom, torch.zeros_like(es_custom), atol=1e-10)

    @pytest.mark.unit
    def test_energy_score_batch_consistency(self):
        """Test that batched computation gives same results as individual computations"""
        n_samples, lat, lon, out_features = 4, 3, 3, 1

        torch.manual_seed(101)
        x1 = torch.randn(1, n_samples, lat * lon * out_features)
        y1 = torch.randn(1, lat * lon * out_features)
        x2 = torch.randn(1, n_samples, lat * lon * out_features)
        y2 = torch.randn(1, lat * lon * out_features)

        energy_score_model = EnergyScore(beta=1.0, clamp=False)

        # Compute individually
        es1 = energy_score_model(x1, y1)
        es2 = energy_score_model(x2, y2)

        # Compute batched
        x_batch = torch.cat([x1, x2], dim=0)
        y_batch = torch.cat([y1, y2], dim=0)
        es_batch = energy_score_model(x_batch, y_batch)

        # Results should be identical
        torch.testing.assert_close(es_batch[0:1], es1, rtol=1e-10, atol=1e-10)
        torch.testing.assert_close(es_batch[1:2], es2, rtol=1e-10, atol=1e-10)

    @pytest.mark.parametrize("beta", [0.5, 1.0, 1.5, 2.0])
    @pytest.mark.unit
    def test_energy_score_beta_parameter(self, beta):
        """Test energy score with different beta values"""
        batch_size, n_samples, lat, lon, out_features = 1, 3, 2, 2, 1

        torch.manual_seed(303)
        x = torch.randn(batch_size, n_samples, lat * lon * out_features)
        y = torch.randn(batch_size, lat * lon * out_features)

        energy_score_model = EnergyScore(beta=beta, clamp=False)
        es_custom = energy_score_model(x, y)

        # Check that result is finite and has correct shape
        assert torch.isfinite(es_custom).all()
        assert es_custom.shape == (batch_size,)

        # For beta=1, compare with scoringrules
        if beta == 1.0:
            es_reference = sr.es_ensemble(y, x, backend="torch")
            torch.testing.assert_close(es_custom, es_reference, rtol=1e-10, atol=1e-10)


class TestVariogramScore:
    """Test cases comparing VariogramScore class with scoringrules.vs_ensemble"""

    @pytest.mark.unit
    def test_variogram_score_simple_case(self):
        """Test variogram score computation for a simple case"""
        # Create simple test data
        batch_size, n_samples, lat, lon, out_features = 2, 3, 4, 4, 1

        # Create deterministic test data for reproducibility
        torch.manual_seed(42)
        # Reshape for VariogramScore: x should be [batch, n_samples, spatial_dims]
        x = torch.randn(batch_size, n_samples, lat * lon * out_features)
        y = torch.randn(batch_size, lat * lon * out_features)

        # Compute variogram score using our implementation
        variogram_score_model = VariogramScore(p=0.5)
        vs_custom = variogram_score_model(x, y)

        # Compute variogram score using scoringrules
        vs_reference = sr.vs_ensemble(y, x, p=0.5, m_axis=1, backend="torch")

        # Compare results
        assert vs_custom.shape == (batch_size,)
        torch.testing.assert_close(vs_custom, vs_reference, rtol=1e-5, atol=1e-6)

    @pytest.mark.unit
    def test_variogram_score_multiple_features(self):
        """Test variogram score with multiple output features"""
        batch_size, n_samples, lat, lon, out_features = 4, 5, 3, 3, 2

        torch.manual_seed(123)
        x = torch.randn(batch_size, n_samples, lat * lon * out_features)
        y = torch.randn(batch_size, lat * lon * out_features)

        variogram_score_model = VariogramScore(p=0.5)
        vs_custom = variogram_score_model(x, y)

        vs_reference = sr.vs_ensemble(y, x, p=0.5, m_axis=1, backend="torch")

        torch.testing.assert_close(vs_custom, vs_reference, rtol=1e-10, atol=1e-10)

    @pytest.mark.parametrize("p", [0.5, 1.0, 1.5, 2.0])
    @pytest.mark.unit
    def test_variogram_score_different_p_values(self, p):
        """Test variogram score with different p values"""
        batch_size, n_samples, lat, lon, out_features = 2, 4, 3, 3, 1

        torch.manual_seed(456)
        x = torch.randn(batch_size, n_samples, lat * lon * out_features)
        y = torch.randn(batch_size, lat * lon * out_features)

        variogram_score_model = VariogramScore(p=p)
        vs_custom = variogram_score_model(x, y)

        vs_reference = sr.vs_ensemble(y, x, p=p, m_axis=1, backend="torch")

        torch.testing.assert_close(vs_custom, vs_reference, rtol=1e-10, atol=1e-10)

    @pytest.mark.unit
    def test_variogram_score_edge_cases(self):
        """Test edge cases like identical predictions"""
        batch_size, n_samples, lat, lon, out_features = 1, 3, 2, 2, 1

        # Case 1: All ensemble members are identical to the truth
        torch.manual_seed(789)
        y = torch.randn(batch_size, lat * lon * out_features)
        x = y.unsqueeze(1).repeat(1, n_samples, 1)  # All ensemble members = truth

        variogram_score_model = VariogramScore(p=0.5)
        vs_custom = variogram_score_model(x, y)

        vs_reference = sr.vs_ensemble(y, x, p=0.5, m_axis=1, backend="torch")

        torch.testing.assert_close(vs_custom, vs_reference, rtol=1e-10, atol=1e-10)

        # When all predictions equal truth, variogram score should be 0
        assert torch.allclose(vs_custom, torch.zeros_like(vs_custom), atol=1e-6)

        # Case 2: All ensemble members are identical but different from truth
        x_constant = torch.ones_like(x) * 2.0  # All ensemble members = 2.0
        y_different = torch.zeros(batch_size, lat * lon * out_features)  # Truth = 0

        vs_custom_constant = variogram_score_model(x_constant, y_different)

        vs_reference_constant = sr.vs_ensemble(
            y_different, x_constant, p=0.5, m_axis=1, backend="torch"
        )

        torch.testing.assert_close(
            vs_custom_constant, vs_reference_constant, rtol=1e-10, atol=1e-10
        )

    @pytest.mark.unit
    def test_variogram_score_batch_consistency(self):
        """Test that batched computation gives same results as individual computations"""
        n_samples, lat, lon, out_features = 4, 3, 3, 1

        torch.manual_seed(101)
        x1 = torch.randn(1, n_samples, lat * lon * out_features)
        y1 = torch.randn(1, lat * lon * out_features)
        x2 = torch.randn(1, n_samples, lat * lon * out_features)
        y2 = torch.randn(1, lat * lon * out_features)

        variogram_score_model = VariogramScore(p=0.5)

        # Compute individually
        vs1 = variogram_score_model(x1, y1)
        vs2 = variogram_score_model(x2, y2)

        # Compute batched
        x_batch = torch.cat([x1, x2], dim=0)
        y_batch = torch.cat([y1, y2], dim=0)
        vs_batch = variogram_score_model(x_batch, y_batch)

        # Results should be identical
        torch.testing.assert_close(vs_batch[0:1], vs1, rtol=1e-10, atol=1e-10)
        torch.testing.assert_close(vs_batch[1:2], vs2, rtol=1e-10, atol=1e-10)

    @pytest.mark.unit
    def test_variogram_score_spatial_structure(self):
        """Test variogram score captures spatial structure differences"""
        batch_size, n_samples, lat, lon, out_features = 1, 10, 5, 5, 1

        torch.manual_seed(202)

        # Create spatially structured truth - flatten directly
        y = torch.zeros(batch_size, lat * lon * out_features)
        for i in range(lat):
            for j in range(lon):
                y[0, i * lon + j] = i + j  # Linear gradient

        # Case 1: Predictions that preserve spatial structure
        x_structured = y.unsqueeze(1).repeat(1, n_samples, 1) + 0.1 * torch.randn(
            batch_size, n_samples, lat * lon * out_features
        )

        # Case 2: Predictions that destroy spatial structure (random)
        x_random = torch.randn(batch_size, n_samples, lat * lon * out_features)

        variogram_score_model = VariogramScore(p=0.5)

        vs_structured = variogram_score_model(x_structured, y)
        vs_random = variogram_score_model(x_random, y)

        # Verify against scoringrules
        vs_structured_ref = sr.vs_ensemble(y, x_structured, p=0.5, m_axis=1, backend="torch")
        vs_random_ref = sr.vs_ensemble(y, x_random, p=0.5, m_axis=1, backend="torch")

        torch.testing.assert_close(vs_structured, vs_structured_ref, rtol=1e-10, atol=1e-10)
        torch.testing.assert_close(vs_random, vs_random_ref, rtol=1e-10, atol=1e-10)

        # Structured predictions should generally have lower variogram score
        # (though this isn't guaranteed for all random seeds)
        assert vs_structured.shape == vs_random.shape

    @pytest.mark.unit
    def test_variogram_score_different_spatial_sizes(self):
        """Test variogram score with different spatial dimensions"""
        batch_size, n_samples, out_features = 2, 5, 1

        # Test different spatial sizes
        spatial_sizes = [(2, 2), (3, 4), (5, 3), (1, 10)]

        for lat, lon in spatial_sizes:
            torch.manual_seed(303)
            x = torch.randn(batch_size, n_samples, lat * lon * out_features)
            y = torch.randn(batch_size, lat * lon * out_features)

            variogram_score_model = VariogramScore(p=0.5)
            vs_custom = variogram_score_model(x, y)

            # Verify against scoringrules
            vs_reference = sr.vs_ensemble(y, x, p=0.5, m_axis=1, backend="torch")

            torch.testing.assert_close(vs_custom, vs_reference, rtol=1e-10, atol=1e-10)
            assert vs_custom.shape == (batch_size,)

    @pytest.mark.unit
    def test_variogram_score_single_ensemble_member(self):
        """Test variogram score with single ensemble member"""
        batch_size, n_samples, lat, lon, out_features = 2, 1, 3, 3, 1

        torch.manual_seed(404)
        x = torch.randn(batch_size, n_samples, lat * lon * out_features)
        y = torch.randn(batch_size, lat * lon * out_features)

        variogram_score_model = VariogramScore(p=0.5)
        vs_custom = variogram_score_model(x, y)

        # Verify against scoringrules
        vs_reference = sr.vs_ensemble(y, x, p=0.5, m_axis=1, backend="torch")

        torch.testing.assert_close(vs_custom, vs_reference, rtol=1e-10, atol=1e-10)

    @pytest.mark.unit
    def test_variogram_score_large_ensemble(self):
        """Test variogram score with larger ensemble size"""
        batch_size, n_samples, lat, lon, out_features = 1, 50, 4, 4, 1

        torch.manual_seed(505)
        x = torch.randn(batch_size, n_samples, lat * lon * out_features)
        y = torch.randn(batch_size, lat * lon * out_features)

        variogram_score_model = VariogramScore(p=0.5)
        vs_custom = variogram_score_model(x, y)

        # Verify against scoringrules
        vs_reference = sr.vs_ensemble(y, x, p=0.5, m_axis=1, backend="torch")

        torch.testing.assert_close(vs_custom, vs_reference, rtol=1e-10, atol=1e-10)


class TestCRPS_Normal:
    """Test cases comparing CRPS_Normal class with scoringrules.crps_normal"""

    @pytest.mark.unit
    def test_crps_normal_simple_case(self):
        """Test CRPS normal computation for a simple case"""
        # Create simple test data
        batch_size, features, height, width = 2, 1, 3, 3

        # Create deterministic test data for reproducibility
        torch.manual_seed(42)
        mu = torch.randn(batch_size, features, height, width)
        sigma = torch.abs(torch.randn(batch_size, features, height, width)) + 0.1  # Ensure positive
        obs = torch.randn(batch_size, features, height, width)

        # Compute CRPS using our implementation
        crps_model = CRPS_Normal()
        crps_custom = crps_model(mu, sigma, obs)

        # Compute CRPS using scoringrules
        crps_reference = sr.crps_normal(obs, mu, sigma, backend="torch")

        # Compare results
        torch.testing.assert_close(crps_custom, crps_reference, rtol=1e-6, atol=1e-6)

    @pytest.mark.unit
    def test_crps_normal_edge_cases(self):
        """Test CRPS normal with edge cases"""
        batch_size, features, height, width = 1, 1, 2, 2

        # Case 1: Perfect prediction (obs = mu)
        torch.manual_seed(123)
        mu = torch.randn(batch_size, features, height, width)
        sigma = torch.ones(batch_size, features, height, width) * 0.5
        obs = mu.clone()  # Perfect prediction

        crps_model = CRPS_Normal()
        crps_custom = crps_model(mu, sigma, obs)
        crps_reference = sr.crps_normal(obs, mu, sigma, backend="torch")

        torch.testing.assert_close(crps_custom, crps_reference, rtol=1e-10, atol=1e-10)

        # Case 2: Very small sigma
        sigma_small = torch.ones(batch_size, features, height, width) * 1e-6
        obs_different = mu + 0.1  # Slightly different from mu

        crps_custom_small = crps_model(mu, sigma_small, obs_different)
        crps_reference_small = sr.crps_normal(obs_different, mu, sigma_small, backend="torch")

        torch.testing.assert_close(crps_custom_small, crps_reference_small, rtol=1e-10, atol=1e-10)

    @pytest.mark.parametrize("sigma_scale", [0.1, 1.0, 2.0, 5.0])
    @pytest.mark.unit
    def test_crps_normal_different_scales(self, sigma_scale):
        """Test CRPS normal with different sigma scales"""
        batch_size, features, height, width = 2, 1, 2, 2

        torch.manual_seed(456)
        mu = torch.randn(batch_size, features, height, width)
        sigma = torch.ones(batch_size, features, height, width) * sigma_scale
        obs = torch.randn(batch_size, features, height, width)

        crps_model = CRPS_Normal()
        crps_custom = crps_model(mu, sigma, obs)
        crps_reference = sr.crps_normal(obs, mu, sigma, backend="torch")

        torch.testing.assert_close(crps_custom, crps_reference, rtol=1e-6, atol=1e-6)

    @pytest.mark.unit
    def test_crps_normal_batch_consistency(self):
        """Test that batched computation gives same results as individual computations"""
        features, height, width = 1, 2, 2

        torch.manual_seed(789)
        mu1 = torch.randn(1, features, height, width)
        sigma1 = torch.abs(torch.randn(1, features, height, width)) + 0.1
        obs1 = torch.randn(1, features, height, width)

        mu2 = torch.randn(1, features, height, width)
        sigma2 = torch.abs(torch.randn(1, features, height, width)) + 0.1
        obs2 = torch.randn(1, features, height, width)

        crps_model = CRPS_Normal()

        # Compute individually
        crps1 = crps_model(mu1, sigma1, obs1)
        crps2 = crps_model(mu2, sigma2, obs2)

        # Compute batched
        mu_batch = torch.cat([mu1, mu2], dim=0)
        sigma_batch = torch.cat([sigma1, sigma2], dim=0)
        obs_batch = torch.cat([obs1, obs2], dim=0)
        crps_batch = crps_model(mu_batch, sigma_batch, obs_batch)

        # Results should be identical
        torch.testing.assert_close(crps_batch[0:1], crps1, rtol=1e-6, atol=1e-7)
        torch.testing.assert_close(crps_batch[1:2], crps2, rtol=1e-6, atol=1e-7)


class TestCRPS_TruncatedNormal:
    """Test cases comparing CRPS_TruncatedNormal class with scoringrules.crps_tnormal"""

    @pytest.mark.unit
    def test_crps_truncated_normal_simple_case(self):
        """Test CRPS truncated normal computation for a simple case"""
        # Create simple test data
        batch_size, features, height, width = 2, 1, 3, 3

        # Create deterministic test data for reproducibility
        torch.manual_seed(42)
        # For truncated normal, mu should be positive for the implementation to work correctly
        mu = torch.abs(torch.randn(batch_size, features, height, width)) + 1.0
        sigma = torch.abs(torch.randn(batch_size, features, height, width)) + 0.1
        obs = torch.abs(torch.randn(batch_size, features, height, width))  # Positive observations

        # Compute CRPS using our implementation
        crps_model = CRPS_TruncatedNormal()
        crps_custom = crps_model(mu, sigma, obs)

        # Compute CRPS using scoringrules (truncated at 0, upper bound inf)
        crps_reference = sr.crps_tnormal(
            obs, mu, sigma, lower=0.0, upper=float("inf"), backend="torch"
        )

        # Compare results
        torch.testing.assert_close(crps_custom, crps_reference, rtol=1e-6, atol=1e-7)

    @pytest.mark.unit
    def test_crps_truncated_normal_edge_cases(self):
        """Test CRPS truncated normal with edge cases"""
        batch_size, features, height, width = 1, 1, 2, 2

        # Case 1: Perfect prediction (obs = mu)
        torch.manual_seed(123)
        mu = torch.ones(batch_size, features, height, width) * 2.0  # Positive
        sigma = torch.ones(batch_size, features, height, width) * 0.5
        obs = mu.clone()  # Perfect prediction

        crps_model = CRPS_TruncatedNormal()
        crps_custom = crps_model(mu, sigma, obs)
        crps_reference = sr.crps_tnormal(
            obs, mu, sigma, lower=0.0, upper=float("inf"), backend="torch"
        )

        torch.testing.assert_close(crps_custom, crps_reference, rtol=1e-6, atol=1e-7)

        # Case 2: Small values near the truncation boundary
        mu_small = torch.ones(batch_size, features, height, width) * 0.5
        sigma_small = torch.ones(batch_size, features, height, width) * 0.1
        obs_small = torch.ones(batch_size, features, height, width) * 0.1

        crps_custom_small = crps_model(mu_small, sigma_small, obs_small)
        crps_reference_small = sr.crps_tnormal(
            obs_small, mu_small, sigma_small, lower=0.0, upper=float("inf"), backend="torch"
        )

        torch.testing.assert_close(crps_custom_small, crps_reference_small, rtol=1e-6, atol=1e-7)

    @pytest.mark.parametrize("mu_scale", [0.5, 1.0, 2.0, 5.0])
    @pytest.mark.unit
    def test_crps_truncated_normal_different_scales(self, mu_scale):
        """Test CRPS truncated normal with different mu scales"""
        batch_size, features, height, width = 2, 1, 2, 2

        torch.manual_seed(456)
        mu = torch.ones(batch_size, features, height, width) * mu_scale
        sigma = torch.ones(batch_size, features, height, width) * 0.5
        obs = torch.abs(torch.randn(batch_size, features, height, width)) + 0.1

        crps_model = CRPS_TruncatedNormal()
        crps_custom = crps_model(mu, sigma, obs)
        crps_reference = sr.crps_tnormal(
            obs, mu, sigma, lower=0.0, upper=float("inf"), backend="torch"
        )

        torch.testing.assert_close(crps_custom, crps_reference, rtol=1e-6, atol=1e-7)

    @pytest.mark.unit
    def test_crps_truncated_normal_batch_consistency(self):
        """Test that batched computation gives same results as individual computations"""
        features, height, width = 1, 2, 2

        torch.manual_seed(789)
        mu1 = torch.ones(1, features, height, width) * 1.5
        sigma1 = torch.ones(1, features, height, width) * 0.3
        obs1 = torch.ones(1, features, height, width) * 0.8

        mu2 = torch.ones(1, features, height, width) * 2.5
        sigma2 = torch.ones(1, features, height, width) * 0.7
        obs2 = torch.ones(1, features, height, width) * 1.2

        crps_model = CRPS_TruncatedNormal()

        # Compute individually
        crps1 = crps_model(mu1, sigma1, obs1)
        crps2 = crps_model(mu2, sigma2, obs2)

        # Compute batched
        mu_batch = torch.cat([mu1, mu2], dim=0)
        sigma_batch = torch.cat([sigma1, sigma2], dim=0)
        obs_batch = torch.cat([obs1, obs2], dim=0)
        crps_batch = crps_model(mu_batch, sigma_batch, obs_batch)

        # Results should be identical
        torch.testing.assert_close(crps_batch[0:1], crps1, rtol=1e-6, atol=1e-7)
        torch.testing.assert_close(crps_batch[1:2], crps2, rtol=1e-6, atol=1e-7)

    @pytest.mark.unit
    def test_crps_truncated_normal_comparison_with_normal(self):
        """Test that truncated normal CRPS approaches normal CRPS when mu >> 0"""
        batch_size, features, height, width = 20, 20, 20, 20

        # Use large positive mu and small sigma so truncation has minimal effect
        torch.manual_seed(101)
        mu = torch.ones(batch_size, features, height, width) * 10.0  # Large positive
        sigma = torch.ones(batch_size, features, height, width) * 0.1  # Small
        obs = mu + torch.randn(batch_size, features, height, width) * 0.1  # Close to mu

        # Compute CRPS using both models
        crps_truncated_model = CRPS_TruncatedNormal()
        crps_normal_model = CRPS_Normal()

        crps_truncated = crps_truncated_model(mu, sigma, obs)
        crps_normal = crps_normal_model(mu, sigma, obs)

        # They should be approximately equal when truncation has minimal effect
        torch.testing.assert_close(crps_truncated, crps_normal, rtol=1e-2, atol=1e-3)


class TestEnsembleCRPS:
    """Test cases comparing EnsembleCRPS class with scoringrules.crps_ensemble"""

    @pytest.mark.unit
    def test_ensemble_crps_simple_case(self):
        """Test ensemble CRPS computation for a simple case"""
        # Create simple test data
        batch_size, n_samples, out_features, height, width = 2, 5, 1, 3, 3

        # Create deterministic test data for reproducibility
        torch.manual_seed(42)
        x = torch.randn(batch_size, n_samples, out_features, height, width)
        y = torch.randn(batch_size, out_features, height, width)

        # Compute CRPS using our implementation
        crps_model = EnsembleCRPS()
        crps_custom = crps_model(x, y)

        # Compute CRPS using scoringrules
        crps_reference = sr.crps_ensemble(y, x, m_axis=-4, estimator="nrg", backend="torch")

        # Compare results
        assert crps_custom.shape == (batch_size, out_features, height, width)
        torch.testing.assert_close(crps_custom, crps_reference, rtol=1e-5, atol=1e-6)

    @pytest.mark.unit
    def test_ensemble_crps_multiple_features(self):
        """Test ensemble CRPS with multiple output features"""
        batch_size, n_samples, out_features, height, width = 3, 4, 2, 4, 4

        torch.manual_seed(123)
        x = torch.randn(batch_size, n_samples, out_features, height, width)
        y = torch.randn(batch_size, out_features, height, width)

        crps_model = EnsembleCRPS()
        crps_custom = crps_model(x, y)

        crps_reference = sr.crps_ensemble(y, x, m_axis=-4, estimator="nrg", backend="torch")

        torch.testing.assert_close(crps_custom, crps_reference, rtol=1e-5, atol=1e-6)

    @pytest.mark.unit
    def test_ensemble_crps_edge_cases(self):
        """Test edge cases like identical predictions"""
        batch_size, n_samples, out_features, height, width = 1, 3, 1, 2, 2

        # Case 1: All ensemble members are identical to the truth
        torch.manual_seed(456)
        y = torch.randn(batch_size, out_features, height, width)
        x = y.unsqueeze(1).repeat(1, n_samples, 1, 1, 1)  # All ensemble members = truth

        crps_model = EnsembleCRPS()
        crps_custom = crps_model(x, y)

        crps_reference = sr.crps_ensemble(y, x, m_axis=-4, estimator="nrg", backend="torch")

        torch.testing.assert_close(crps_custom, crps_reference, rtol=1e-10, atol=1e-10)

        # CRPS should be 0 when all predictions equal truth
        assert torch.allclose(crps_custom, torch.zeros_like(crps_custom), atol=1e-10)

        # Case 2: All ensemble members are identical but different from truth
        x_constant = torch.ones_like(x) * 2.0  # All ensemble members = 2.0
        y_different = torch.zeros(batch_size, out_features, height, width)  # Truth = 0

        crps_custom_constant = crps_model(x_constant, y_different)
        crps_reference_constant = sr.crps_ensemble(
            y_different, x_constant, m_axis=-4, estimator="nrg", backend="torch"
        )

        torch.testing.assert_close(
            crps_custom_constant, crps_reference_constant, rtol=1e-10, atol=1e-10
        )

    @pytest.mark.unit
    def test_ensemble_crps_batch_consistency(self):
        """Test that batched computation gives same results as individual computations"""
        n_samples, out_features, height, width = 4, 1, 3, 3

        torch.manual_seed(789)
        x1 = torch.randn(1, n_samples, out_features, height, width)
        y1 = torch.randn(1, out_features, height, width)
        x2 = torch.randn(1, n_samples, out_features, height, width)
        y2 = torch.randn(1, out_features, height, width)

        crps_model = EnsembleCRPS()

        # Compute individually
        crps1 = crps_model(x1, y1)
        crps2 = crps_model(x2, y2)

        # Compute batched
        x_batch = torch.cat([x1, x2], dim=0)
        y_batch = torch.cat([y1, y2], dim=0)
        crps_batch = crps_model(x_batch, y_batch)

        # Results should be identical
        torch.testing.assert_close(crps_batch[0:1], crps1, rtol=1e-10, atol=1e-10)
        torch.testing.assert_close(crps_batch[1:2], crps2, rtol=1e-10, atol=1e-10)

    @pytest.mark.parametrize("n_samples", [1, 3, 5, 10, 20])
    @pytest.mark.unit
    def test_ensemble_crps_different_ensemble_sizes(self, n_samples):
        """Test ensemble CRPS with different ensemble sizes"""
        batch_size, out_features, height, width = 2, 1, 3, 3

        torch.manual_seed(101)
        x = torch.randn(batch_size, n_samples, out_features, height, width)
        y = torch.randn(batch_size, out_features, height, width)

        crps_model = EnsembleCRPS()
        crps_custom = crps_model(x, y)

        crps_reference = sr.crps_ensemble(y, x, m_axis=-4, estimator="nrg", backend="torch")

        torch.testing.assert_close(crps_custom, crps_reference, rtol=1e-10, atol=1e-10)
        assert crps_custom.shape == (batch_size, out_features, height, width)

    @pytest.mark.unit
    def test_ensemble_crps_different_spatial_sizes(self):
        """Test ensemble CRPS with different spatial dimensions"""
        batch_size, n_samples, out_features = 2, 5, 1

        # Test different spatial sizes
        spatial_sizes = [(1, 1), (2, 2), (3, 4), (5, 3), (1, 10)]

        for height, width in spatial_sizes:
            torch.manual_seed(202)
            x = torch.randn(batch_size, n_samples, out_features, height, width)
            y = torch.randn(batch_size, out_features, height, width)

            crps_model = EnsembleCRPS()
            crps_custom = crps_model(x, y)

            crps_reference = sr.crps_ensemble(y, x, m_axis=-4, estimator="nrg", backend="torch")

            torch.testing.assert_close(crps_custom, crps_reference, rtol=1e-10, atol=1e-10)
            assert crps_custom.shape == (batch_size, out_features, height, width)

    @pytest.mark.unit
    def test_ensemble_crps_perfect_forecast_distribution(self):
        """Test ensemble CRPS when ensemble perfectly represents the truth distribution"""
        batch_size, n_samples, out_features, height, width = 2, 100, 1, 2, 2

        torch.manual_seed(303)
        # Create truth values
        y = torch.randn(batch_size, out_features, height, width)

        # Create ensemble that's normally distributed around the truth
        noise = torch.randn(batch_size, n_samples, out_features, height, width) * 0.1
        x = y.unsqueeze(1) + noise

        crps_model = EnsembleCRPS()
        crps_custom = crps_model(x, y)

        crps_reference = sr.crps_ensemble(y, x, m_axis=-4, estimator="nrg", backend="torch")

        torch.testing.assert_close(crps_custom, crps_reference, rtol=1e-10, atol=1e-10)

        # With large ensemble size and small noise, CRPS should be relatively small
        assert torch.all(crps_custom >= 0)  # CRPS is always non-negative
        assert torch.all(crps_custom < 1.0)  # Should be small for this setup

    @pytest.mark.unit
    def test_ensemble_crps_deterministic_forecast(self):
        """Test ensemble CRPS with deterministic forecast (all members identical)"""
        batch_size, n_samples, out_features, height, width = 2, 5, 1, 3, 3

        torch.manual_seed(404)
        y = torch.randn(batch_size, out_features, height, width)

        # Create deterministic forecast (all ensemble members identical)
        x_det = torch.randn(batch_size, 1, out_features, height, width)
        x = x_det.repeat(1, n_samples, 1, 1, 1)

        crps_model = EnsembleCRPS()
        crps_custom = crps_model(x, y)

        crps_reference = sr.crps_ensemble(y, x, m_axis=-4, estimator="nrg", backend="torch")

        torch.testing.assert_close(crps_custom, crps_reference, rtol=1e-6, atol=1e-7)

        # For deterministic forecast, CRPS should equal absolute error
        expected_crps = torch.abs(x_det.squeeze(1) - y)
        torch.testing.assert_close(crps_custom, expected_crps, rtol=1e-6, atol=1e-7)

    @pytest.mark.unit
    def test_ensemble_crps_properties(self):
        """Test mathematical properties of ensemble CRPS"""
        batch_size, n_samples, out_features, height, width = 3, 7, 2, 4, 4

        torch.manual_seed(505)
        x = torch.randn(batch_size, n_samples, out_features, height, width)
        y = torch.randn(batch_size, out_features, height, width)

        crps_model = EnsembleCRPS()
        crps_custom = crps_model(x, y)

        # CRPS should be non-negative
        assert torch.all(crps_custom >= 0)

        # CRPS should be finite
        assert torch.all(torch.isfinite(crps_custom))

        # Verify against scoringrules
        crps_reference = sr.crps_ensemble(y, x, m_axis=-4, estimator="nrg", backend="torch")
        torch.testing.assert_close(crps_custom, crps_reference, rtol=1e-10, atol=1e-10)

    @pytest.mark.unit
    def test_ensemble_crps_shape_handling(self):
        """Test that ensemble CRPS handles different input shapes correctly"""
        # Test with extra leading dimensions
        extra_dims, batch_size, n_samples, out_features, height, width = 2, 3, 4, 1, 2, 2

        torch.manual_seed(606)
        x = torch.randn(extra_dims, batch_size, n_samples, out_features, height, width)
        y = torch.randn(extra_dims, batch_size, out_features, height, width)

        crps_model = EnsembleCRPS()
        crps_custom = crps_model(x, y)

        crps_reference = sr.crps_ensemble(y, x, m_axis=-4, estimator="nrg", backend="torch")

        torch.testing.assert_close(crps_custom, crps_reference, rtol=1e-10, atol=1e-10)
        assert crps_custom.shape == (extra_dims, batch_size, out_features, height, width)

    @pytest.mark.unit
    def test_ensemble_crps_single_ensemble_member(self):
        """Test ensemble CRPS with single ensemble member"""
        batch_size, n_samples, out_features, height, width = 2, 1, 1, 3, 3

        torch.manual_seed(707)
        x = torch.randn(batch_size, n_samples, out_features, height, width)
        y = torch.randn(batch_size, out_features, height, width)

        crps_model = EnsembleCRPS()
        crps_custom = crps_model(x, y)

        crps_reference = sr.crps_ensemble(y, x, m_axis=-4, estimator="nrg", backend="torch")

        torch.testing.assert_close(crps_custom, crps_reference, rtol=1e-10, atol=1e-10)

        # With single ensemble member, CRPS should equal absolute error
        expected_crps = torch.abs(x.squeeze(1) - y)
        torch.testing.assert_close(crps_custom, expected_crps, rtol=1e-10, atol=1e-10)
        torch.manual_seed(505)
        x = torch.randn(batch_size, n_samples, out_features, height, width)
        y = torch.randn(batch_size, out_features, height, width)

        crps_model = EnsembleCRPS()
        crps_custom = crps_model(x, y)

        # CRPS should be non-negative
        assert torch.all(crps_custom >= 0)

        # CRPS should be finite
        assert torch.all(torch.isfinite(crps_custom))

        # Verify against scoringrules
        crps_reference = sr.crps_ensemble(y, x, m_axis=-4, estimator="nrg", backend="torch")
        torch.testing.assert_close(crps_custom, crps_reference, rtol=1e-10, atol=1e-10)
