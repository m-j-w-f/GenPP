import pytest
import scoringrules as sr
import torch

from genpp.models.loss import EnergyScore


class TestEnergyScore:
    """Test cases comparing EnergyScore class with scoringrules.es_ensemble"""

    def test_energy_score_simple_case(self):
        """Test energy score computation for a simple case"""
        # Create simple test data
        batch_size, n_samples, lat, lon, out_features = 2, 3, 4, 4, 1

        # Create deterministic test data for reproducibility
        torch.manual_seed(42)
        x = torch.randn(batch_size, n_samples, out_features, lon, lat)
        y = torch.randn(batch_size, out_features, lon, lat)

        # Compute energy score using our implementation
        energy_score_model = EnergyScore(beta=1.0, clamp=False)
        es_custom = energy_score_model(x, y)

        # Prepare data for scoringrules (needs different shape)
        # scoringrules expects: obs shape [batch, variables], fct shape [batch, ensemble, variables]
        x_flat = x.view(batch_size, n_samples, -1)  # [batch, ensemble, spatial*features]
        y_flat = y.view(batch_size, -1)  # [batch, spatial*features]

        # Compute energy score using scoringrules
        es_reference = sr.es_ensemble(y_flat, x_flat, backend="torch")

        # Compare results - they should be close (allowing for numerical differences)
        assert es_custom.shape == (batch_size, out_features)
        torch.testing.assert_close(es_custom.flatten(), es_reference, rtol=1e-5, atol=1e-6)

    def test_energy_score_multiple_features(self):
        """Test energy score with multiple output features"""
        batch_size, n_samples, lat, lon, out_features = 1, 5, 3, 3, 2

        torch.manual_seed(123)
        x = torch.randn(batch_size, n_samples, out_features, lon, lat)
        y = torch.randn(batch_size, out_features, lon, lat)

        energy_score_model = EnergyScore(beta=1.0, clamp=False)
        es_custom = energy_score_model(x, y)

        # Test each feature separately since scoringrules handles them independently
        for feature_idx in range(out_features):
            x_feat = x[:, :, feature_idx : feature_idx + 1, ...]  # Keep dimension
            y_feat = y[:, feature_idx : feature_idx + 1, ...]

            x_flat = x_feat.view(batch_size, n_samples, -1)
            y_flat = y_feat.view(batch_size, -1)

            es_reference = sr.es_ensemble(y_flat, x_flat, backend="torch")

            torch.testing.assert_close(
                es_custom[0, feature_idx],
                es_reference[0],
                rtol=1e-5,
                atol=1e-6,
                msg=f"Mismatch for feature {feature_idx}",
            )

    def test_energy_score_different_beta_values(self):
        """Test energy score with different beta values (note: scoringrules uses beta=1)"""
        batch_size, n_samples, lat, lon, out_features = 1, 4, 2, 2, 1

        torch.manual_seed(456)
        x = torch.randn(batch_size, n_samples, out_features, lon, lat)
        y = torch.randn(batch_size, out_features, lon, lat)

        # Test with beta=1.0 (should match scoringrules)
        energy_score_beta1 = EnergyScore(beta=1.0, clamp=False)
        es_custom_beta1 = energy_score_beta1(x, y)

        x_flat = x.view(batch_size, n_samples, -1)
        y_flat = y.view(batch_size, -1)
        es_reference = sr.es_ensemble(y_flat, x_flat, backend="torch")

        torch.testing.assert_close(es_custom_beta1.flatten(), es_reference, rtol=1e-5, atol=1e-6)

        # Test with beta=2.0 (should be different from scoringrules)
        energy_score_beta2 = EnergyScore(beta=2.0, clamp=False)
        es_custom_beta2 = energy_score_beta2(x, y)

        # They should not be equal
        assert not torch.allclose(es_custom_beta1, es_custom_beta2)

    def test_energy_score_edge_cases(self):
        """Test edge cases like identical predictions"""
        batch_size, n_samples, lat, lon, out_features = 1, 3, 2, 2, 1

        # Case 1: All ensemble members are identical
        torch.manual_seed(789)
        y = torch.randn(batch_size, out_features, lon, lat)
        x = y.unsqueeze(1).repeat(1, n_samples, 1, 1, 1)  # All ensemble members = truth

        energy_score_model = EnergyScore(beta=1.0, clamp=False)
        es_custom = energy_score_model(x, y)

        x_flat = x.view(batch_size, n_samples, -1)
        y_flat = y.view(batch_size, -1)
        es_reference = sr.es_ensemble(y_flat, x_flat, backend="torch")

        torch.testing.assert_close(es_custom.flatten(), es_reference, rtol=1e-5, atol=1e-6)

        # Energy score should be 0 when all predictions equal truth
        assert torch.allclose(es_custom, torch.zeros_like(es_custom), atol=1e-6)

    def test_energy_score_batch_consistency(self):
        """Test that batched computation gives same results as individual computations"""
        n_samples, lat, lon, out_features = 4, 3, 3, 1

        torch.manual_seed(101)
        x1 = torch.randn(1, n_samples, out_features, lon, lat)
        y1 = torch.randn(1, out_features, lon, lat)
        x2 = torch.randn(1, n_samples, out_features, lon, lat)
        y2 = torch.randn(1, out_features, lon, lat)

        energy_score_model = EnergyScore(beta=1.0, clamp=False)

        # Compute individually
        es1 = energy_score_model(x1, y1)
        es2 = energy_score_model(x2, y2)

        # Compute batched
        x_batch = torch.cat([x1, x2], dim=0)
        y_batch = torch.cat([y1, y2], dim=0)
        es_batch = energy_score_model(x_batch, y_batch)

        # Results should be identical
        torch.testing.assert_close(es_batch[0], es1[0], rtol=1e-10, atol=1e-10)
        torch.testing.assert_close(es_batch[1], es2[0], rtol=1e-10, atol=1e-10)

    @pytest.mark.parametrize("beta", [0.5, 1.0, 1.5, 2.0])
    def test_energy_score_beta_parameter(self, beta):
        """Test energy score with different beta values"""
        batch_size, n_samples, lat, lon, out_features = 1, 3, 2, 2, 1

        torch.manual_seed(303)
        x = torch.randn(batch_size, n_samples, out_features, lon, lat)
        y = torch.randn(batch_size, out_features, lon, lat)

        energy_score_model = EnergyScore(beta=beta, clamp=False)
        es_custom = energy_score_model(x, y)

        # Check that result is finite and has correct shape
        assert torch.isfinite(es_custom).all()
        assert es_custom.shape == (batch_size, out_features)

        # For beta=1, compare with scoringrules
        if beta == 1.0:
            x_flat = x.view(batch_size, n_samples, -1)
            y_flat = y.view(batch_size, -1)
            es_reference = sr.es_ensemble(y_flat, x_flat, backend="torch")

            torch.testing.assert_close(es_custom.flatten(), es_reference, rtol=1e-5, atol=1e-6)
