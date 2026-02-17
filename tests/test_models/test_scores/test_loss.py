import pytest
import scoringrules as sr
import torch

from genpp.models.scores import (
    CRPS_Normal,
    CRPS_TruncatedNormal,
    EnergyScore,
    EnsembleCRPS,
    MultiScaleEnergyScore,
    MultiScalePatchwiseRBFScore,
    MultiScaleRBFScore,
    PatchwiseEnergyScore,
    PatchwiseRBFScore,
    RBFScore,
    VariogramScore,
)


class TestMultiScaleEnergyScore:
    """Test cases for MultiScaleEnergyScore class"""

    @pytest.mark.unit
    def test_multiscale_energy_score_simple_case(self):
        """Test multi-scale Energy score computation"""
        batch_size, n_samples, lat, lon, out_features = 2, 3, 16, 16, 1

        torch.manual_seed(42)
        x = torch.randn(batch_size, n_samples, out_features, lat, lon)
        y = torch.randn(batch_size, out_features, lat, lon)

        multi_scale_es = MultiScaleEnergyScore(blur_kernel_sizes=[3, 5, 7])
        es = multi_scale_es(x, y, mode="complete")

        assert es.shape == (batch_size,)
        assert torch.isfinite(es).all()

    @pytest.mark.unit
    def test_multiscale_energy_score_per_var_mode(self):
        """Test multi-scale Energy score in per_var mode"""
        batch_size, n_samples, lat, lon, out_features = 2, 3, 16, 16, 3

        torch.manual_seed(123)
        x = torch.randn(batch_size, n_samples, out_features, lat, lon)
        y = torch.randn(batch_size, out_features, lat, lon)

        multi_scale_es = MultiScaleEnergyScore(blur_kernel_sizes=[3, 7])
        es = multi_scale_es(x, y, mode="per_var")

        assert es.shape == (batch_size, out_features)
        assert torch.isfinite(es).all()


class TestEnergyScore:
    """Test cases comparing EnergyScore class with scoringrules.es_ensemble"""

    @pytest.mark.unit
    def test_energy_score_simple_case(self):
        """Test energy score computation for a simple case"""
        # Create simple test data
        batch_size, n_samples, lat, lon, out_features = 2, 3, 4, 4, 1

        # Create deterministic test data for reproducibility
        torch.manual_seed(42)
        # New shape: x [batch, n_samples, c, h, w], y [batch, c, h, w]
        x = torch.randn(batch_size, n_samples, out_features, lat, lon)
        y = torch.randn(batch_size, out_features, lat, lon)

        # Compute energy score using our implementation
        energy_score_model = EnergyScore(beta=1.0, clamp=False, unbiased=False)
        es_custom = energy_score_model(x, y, mode="complete")

        # Prepare data for scoringrules (needs flattened format)
        x_flat = x.reshape(batch_size, n_samples, -1)
        y_flat = y.reshape(batch_size, -1)
        es_reference = sr.es_ensemble(y_flat, x_flat, backend="torch")

        # Compare results - they should be close (allowing for numerical differences)
        assert es_custom.shape == (batch_size,)
        torch.testing.assert_close(es_custom, es_reference, rtol=1e-10, atol=1e-10)

    @pytest.mark.unit
    def test_energy_score_multiple_features(self):
        """Test energy score with multiple output features"""
        batch_size, n_samples, lat, lon, out_features = 4, 5, 3, 3, 2

        torch.manual_seed(123)
        # New shape: x [batch, n_samples, c, h, w], y [batch, c, h, w]
        x = torch.randn(batch_size, n_samples, out_features, lat, lon)
        y = torch.randn(batch_size, out_features, lat, lon)

        energy_score_model = EnergyScore(beta=1.0, clamp=False, unbiased=False)
        es_custom = energy_score_model(x, y, mode="complete")

        # Prepare data for scoringrules (needs flattened format)
        x_flat = x.reshape(batch_size, n_samples, -1)
        y_flat = y.reshape(batch_size, -1)
        es_reference = sr.es_ensemble(y_flat, x_flat, backend="torch")
        torch.testing.assert_close(es_custom, es_reference, rtol=1e-7, atol=1e-7)

    @pytest.mark.unit
    def test_energy_score_different_beta_values(self):
        """Test energy score with different beta values (note: scoringrules uses beta=1)"""
        batch_size, n_samples, lat, lon, out_features = 1, 4, 2, 2, 1

        torch.manual_seed(456)
        # New shape: x [batch, n_samples, c, h, w], y [batch, c, h, w]
        x = torch.randn(batch_size, n_samples, out_features, lat, lon)
        y = torch.randn(batch_size, out_features, lat, lon)

        # Test with beta=1.0 (should match scoringrules)
        energy_score_beta1 = EnergyScore(beta=1.0, clamp=False, unbiased=False)
        es_custom_beta1 = energy_score_beta1(x, y, mode="complete")

        # Prepare data for scoringrules
        x_flat = x.reshape(batch_size, n_samples, -1)
        y_flat = y.reshape(batch_size, -1)
        es_reference = sr.es_ensemble(y_flat, x_flat, backend="torch")

        torch.testing.assert_close(es_custom_beta1, es_reference, rtol=1e-7, atol=1e-7)

        # Test with beta=2.0 (should be different from scoringrules)
        energy_score_beta2 = EnergyScore(beta=2.0, clamp=False)
        es_custom_beta2 = energy_score_beta2(x, y, mode="complete")

        # They should not be equal
        assert not torch.allclose(es_custom_beta1, es_custom_beta2)

    @pytest.mark.unit
    def test_energy_score_edge_cases(self):
        """Test edge cases like identical predictions"""
        batch_size, n_samples, lat, lon, out_features = 1, 3, 2, 2, 1

        # Case 1: All ensemble members are identical
        torch.manual_seed(789)
        y = torch.randn(batch_size, out_features, lat, lon)
        x = y.unsqueeze(1).repeat(1, n_samples, 1, 1, 1)  # All ensemble members = truth

        energy_score_model = EnergyScore(beta=1.0, clamp=False, unbiased=False)
        es_custom = energy_score_model(x, y, mode="complete")

        # Prepare data for scoringrules
        x_flat = x.reshape(batch_size, n_samples, -1)
        y_flat = y.reshape(batch_size, -1)
        es_reference = sr.es_ensemble(y_flat, x_flat, backend="torch")

        torch.testing.assert_close(es_custom, es_reference, rtol=1e-10, atol=1e-10)

        # Energy score should be 0 when all predictions equal truth
        assert torch.allclose(es_custom, torch.zeros_like(es_custom), atol=1e-10)

    @pytest.mark.unit
    def test_energy_score_batch_consistency(self):
        """Test that batched computation gives same results as individual computations"""
        n_samples, lat, lon, out_features = 4, 3, 3, 1

        torch.manual_seed(101)
        x1 = torch.randn(1, n_samples, out_features, lat, lon)
        y1 = torch.randn(1, out_features, lat, lon)
        x2 = torch.randn(1, n_samples, out_features, lat, lon)
        y2 = torch.randn(1, out_features, lat, lon)

        energy_score_model = EnergyScore(beta=1.0, clamp=False, unbiased=False)

        # Compute individually
        es1 = energy_score_model(x1, y1, mode="complete")
        es2 = energy_score_model(x2, y2, mode="complete")

        # Compute batched
        x_batch = torch.cat([x1, x2], dim=0)
        y_batch = torch.cat([y1, y2], dim=0)
        es_batch = energy_score_model(x_batch, y_batch, mode="complete")

        # Results should be identical
        torch.testing.assert_close(es_batch[0:1], es1, rtol=1e-10, atol=1e-10)
        torch.testing.assert_close(es_batch[1:2], es2, rtol=1e-10, atol=1e-10)

    @pytest.mark.parametrize("beta", [0.5, 1.0, 1.5, 2.0])
    @pytest.mark.unit
    def test_energy_score_beta_parameter(self, beta):
        """Test energy score with different beta values"""
        batch_size, n_samples, lat, lon, out_features = 1, 3, 2, 2, 1

        torch.manual_seed(303)
        x = torch.randn(batch_size, n_samples, out_features, lat, lon)
        y = torch.randn(batch_size, out_features, lat, lon)

        energy_score_model = EnergyScore(beta=beta, clamp=False, unbiased=False)
        es_custom = energy_score_model(x, y, mode="complete")

        # Check that result is finite and has correct shape
        assert torch.isfinite(es_custom).all()
        assert es_custom.shape == (batch_size,)

        # For beta=1, compare with scoringrules
        if beta == 1.0:
            x_flat = x.reshape(batch_size, n_samples, -1)
            y_flat = y.reshape(batch_size, -1)
            es_reference = sr.es_ensemble(y_flat, x_flat, backend="torch")
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
        # New shape: x [batch, n_samples, c, h, w], y [batch, c, h, w]
        x = torch.randn(batch_size, n_samples, out_features, lat, lon)
        y = torch.randn(batch_size, out_features, lat, lon)

        # Compute variogram score using our implementation
        variogram_score_model = VariogramScore(p=0.5)
        vs_custom = variogram_score_model(x, y, mode="complete")

        # Prepare data for scoringrules (needs flattened format)
        x_flat = x.reshape(batch_size, n_samples, -1)
        y_flat = y.reshape(batch_size, -1)
        vs_reference = sr.vs_ensemble(y_flat, x_flat, p=0.5, m_axis=1, backend="torch")

        # Compare results
        assert vs_custom.shape == (batch_size,)
        torch.testing.assert_close(vs_custom, vs_reference, rtol=1e-5, atol=1e-6)

    @pytest.mark.unit
    def test_variogram_score_multiple_features(self):
        """Test variogram score with multiple output features"""
        batch_size, n_samples, lat, lon, out_features = 4, 5, 3, 3, 2

        torch.manual_seed(123)
        x = torch.randn(batch_size, n_samples, out_features, lat, lon)
        y = torch.randn(batch_size, out_features, lat, lon)

        variogram_score_model = VariogramScore(p=0.5)
        vs_custom = variogram_score_model(x, y, mode="complete")

        x_flat = x.reshape(batch_size, n_samples, -1)
        y_flat = y.reshape(batch_size, -1)
        vs_reference = sr.vs_ensemble(y_flat, x_flat, p=0.5, m_axis=1, backend="torch")

        torch.testing.assert_close(vs_custom, vs_reference, rtol=1e-10, atol=1e-10)

    @pytest.mark.parametrize("p", [0.5, 1.0, 1.5, 2.0])
    @pytest.mark.unit
    def test_variogram_score_different_p_values(self, p):
        """Test variogram score with different p values"""
        batch_size, n_samples, lat, lon, out_features = 2, 4, 3, 3, 1

        torch.manual_seed(456)
        x = torch.randn(batch_size, n_samples, out_features, lat, lon)
        y = torch.randn(batch_size, out_features, lat, lon)

        variogram_score_model = VariogramScore(p=p)
        vs_custom = variogram_score_model(x, y, mode="complete")

        x_flat = x.reshape(batch_size, n_samples, -1)
        y_flat = y.reshape(batch_size, -1)
        vs_reference = sr.vs_ensemble(y_flat, x_flat, p=p, m_axis=1, backend="torch")

        torch.testing.assert_close(vs_custom, vs_reference, rtol=1e-10, atol=1e-10)

    @pytest.mark.unit
    def test_variogram_score_edge_cases(self):
        """Test edge cases like identical predictions"""
        batch_size, n_samples, lat, lon, out_features = 1, 3, 2, 2, 1

        # Case 1: All ensemble members are identical to the truth
        torch.manual_seed(789)
        y = torch.randn(batch_size, out_features, lat, lon)
        x = y.unsqueeze(1).repeat(1, n_samples, 1, 1, 1)  # All ensemble members = truth

        variogram_score_model = VariogramScore(p=0.5)
        vs_custom = variogram_score_model(x, y, mode="complete")

        x_flat = x.reshape(batch_size, n_samples, -1)
        y_flat = y.reshape(batch_size, -1)
        vs_reference = sr.vs_ensemble(y_flat, x_flat, p=0.5, m_axis=1, backend="torch")

        torch.testing.assert_close(vs_custom, vs_reference, rtol=1e-10, atol=1e-10)

        # When all predictions equal truth, variogram score should be 0
        assert torch.allclose(vs_custom, torch.zeros_like(vs_custom), atol=1e-6)

        # Case 2: All ensemble members are identical but different from truth
        x_constant = torch.ones_like(x) * 2.0  # All ensemble members = 2.0
        y_different = torch.zeros(batch_size, out_features, lat, lon)  # Truth = 0

        vs_custom_constant = variogram_score_model(x_constant, y_different, mode="complete")

        x_const_flat = x_constant.reshape(batch_size, n_samples, -1)
        y_diff_flat = y_different.reshape(batch_size, -1)
        vs_reference_constant = sr.vs_ensemble(
            y_diff_flat, x_const_flat, p=0.5, m_axis=1, backend="torch"
        )

        torch.testing.assert_close(
            vs_custom_constant, vs_reference_constant, rtol=1e-10, atol=1e-10
        )

    @pytest.mark.unit
    def test_variogram_score_batch_consistency(self):
        """Test that batched computation gives same results as individual computations"""
        n_samples, lat, lon, out_features = 4, 3, 3, 1

        torch.manual_seed(101)
        x1 = torch.randn(1, n_samples, out_features, lat, lon)
        y1 = torch.randn(1, out_features, lat, lon)
        x2 = torch.randn(1, n_samples, out_features, lat, lon)
        y2 = torch.randn(1, out_features, lat, lon)

        variogram_score_model = VariogramScore(p=0.5)

        # Compute individually
        vs1 = variogram_score_model(x1, y1, mode="complete")
        vs2 = variogram_score_model(x2, y2, mode="complete")

        # Compute batched
        x_batch = torch.cat([x1, x2], dim=0)
        y_batch = torch.cat([y1, y2], dim=0)
        vs_batch = variogram_score_model(x_batch, y_batch, mode="complete")

        # Results should be identical
        torch.testing.assert_close(vs_batch[0:1], vs1, rtol=1e-10, atol=1e-10)
        torch.testing.assert_close(vs_batch[1:2], vs2, rtol=1e-10, atol=1e-10)

    @pytest.mark.unit
    def test_variogram_score_spatial_structure(self):
        """Test variogram score captures spatial structure differences"""
        batch_size, n_samples, lat, lon, out_features = 1, 10, 5, 5, 1

        torch.manual_seed(202)

        # Create spatially structured truth with shape [b, c, h, w]
        y = torch.zeros(batch_size, out_features, lat, lon)
        for i in range(lat):
            for j in range(lon):
                y[0, 0, i, j] = i + j  # Linear gradient

        # Case 1: Predictions that preserve spatial structure
        x_structured = y.unsqueeze(1).repeat(1, n_samples, 1, 1, 1) + 0.1 * torch.randn(
            batch_size, n_samples, out_features, lat, lon
        )

        # Case 2: Predictions that destroy spatial structure (random)
        x_random = torch.randn(batch_size, n_samples, out_features, lat, lon)

        variogram_score_model = VariogramScore(p=0.5)

        vs_structured = variogram_score_model(x_structured, y, mode="complete")
        vs_random = variogram_score_model(x_random, y, mode="complete")

        # Verify against scoringrules (needs flattened format)
        x_struct_flat = x_structured.reshape(batch_size, n_samples, -1)
        x_rand_flat = x_random.reshape(batch_size, n_samples, -1)
        y_flat = y.reshape(batch_size, -1)
        vs_structured_ref = sr.vs_ensemble(y_flat, x_struct_flat, p=0.5, m_axis=1, backend="torch")
        vs_random_ref = sr.vs_ensemble(y_flat, x_rand_flat, p=0.5, m_axis=1, backend="torch")

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
            x = torch.randn(batch_size, n_samples, out_features, lat, lon)
            y = torch.randn(batch_size, out_features, lat, lon)

            variogram_score_model = VariogramScore(p=0.5)
            vs_custom = variogram_score_model(x, y, mode="complete")

            # Verify against scoringrules
            x_flat = x.reshape(batch_size, n_samples, -1)
            y_flat = y.reshape(batch_size, -1)
            vs_reference = sr.vs_ensemble(y_flat, x_flat, p=0.5, m_axis=1, backend="torch")

            torch.testing.assert_close(vs_custom, vs_reference, rtol=1e-10, atol=1e-10)
            assert vs_custom.shape == (batch_size,)

    @pytest.mark.unit
    def test_variogram_score_single_ensemble_member(self):
        """Test variogram score with single ensemble member"""
        batch_size, n_samples, lat, lon, out_features = 2, 1, 3, 3, 1

        torch.manual_seed(404)
        x = torch.randn(batch_size, n_samples, out_features, lat, lon)
        y = torch.randn(batch_size, out_features, lat, lon)

        variogram_score_model = VariogramScore(p=0.5)
        vs_custom = variogram_score_model(x, y, mode="complete")

        # Verify against scoringrules
        x_flat = x.reshape(batch_size, n_samples, -1)
        y_flat = y.reshape(batch_size, -1)
        vs_reference = sr.vs_ensemble(y_flat, x_flat, p=0.5, m_axis=1, backend="torch")

        torch.testing.assert_close(vs_custom, vs_reference, rtol=1e-10, atol=1e-10)

    @pytest.mark.unit
    def test_variogram_score_large_ensemble(self):
        """Test variogram score with larger ensemble size"""
        batch_size, n_samples, lat, lon, out_features = 1, 50, 4, 4, 1

        torch.manual_seed(505)
        x = torch.randn(batch_size, n_samples, out_features, lat, lon)
        y = torch.randn(batch_size, out_features, lat, lon)

        variogram_score_model = VariogramScore(p=0.5)
        vs_custom = variogram_score_model(x, y, mode="complete")

        # Verify against scoringrules
        x_flat = x.reshape(batch_size, n_samples, -1)
        y_flat = y.reshape(batch_size, -1)
        vs_reference = sr.vs_ensemble(y_flat, x_flat, p=0.5, m_axis=1, backend="torch")

        torch.testing.assert_close(vs_custom, vs_reference, rtol=1e-10, atol=1e-10)


class TestVariogramScoreChunked:
    """Test that the chunked VariogramScore produces the same results as the full implementation.

    The chunked version accumulates partial sums in a different order than the full version,
    which causes float32 rounding differences up to ~1e-6 relative. This is expected and
    verified by the float64 test below which shows the implementations are mathematically
    identical (differences < 1e-11).
    """

    @pytest.mark.unit
    def test_chunked_matches_full_complete_mode(self):
        """Test chunked vs full computation in complete mode."""
        batch_size, n_samples, out_features, lat, lon = 2, 5, 2, 8, 8

        torch.manual_seed(42)
        x = torch.randn(batch_size, n_samples, out_features, lat, lon)
        y = torch.randn(batch_size, out_features, lat, lon)

        vs_full = VariogramScore(p=0.5, chunk_size=None)
        vs_chunked = VariogramScore(p=0.5, chunk_size=4)

        result_full = vs_full(x, y, mode="complete")
        result_chunked = vs_chunked(x, y, mode="complete")

        torch.testing.assert_close(result_chunked, result_full, rtol=1e-5, atol=1e-5)

    @pytest.mark.unit
    def test_chunked_matches_full_per_var_mode(self):
        """Test chunked vs full computation in per_var mode."""
        batch_size, n_samples, out_features, lat, lon = 2, 5, 2, 8, 8

        torch.manual_seed(42)
        x = torch.randn(batch_size, n_samples, out_features, lat, lon)
        y = torch.randn(batch_size, out_features, lat, lon)

        vs_full = VariogramScore(p=0.5, chunk_size=None)
        vs_chunked = VariogramScore(p=0.5, chunk_size=4)

        result_full = vs_full(x, y, mode="per_var")
        result_chunked = vs_chunked(x, y, mode="per_var")

        torch.testing.assert_close(result_chunked, result_full, rtol=1e-5, atol=1e-5)

    @pytest.mark.parametrize("p", [0.5, 1.0, 2.0])
    @pytest.mark.unit
    def test_chunked_matches_full_different_p(self, p):
        """Test chunked vs full with different p values."""
        batch_size, n_samples, out_features, lat, lon = 2, 4, 1, 6, 6

        torch.manual_seed(123)
        x = torch.randn(batch_size, n_samples, out_features, lat, lon)
        y = torch.randn(batch_size, out_features, lat, lon)

        vs_full = VariogramScore(p=p, chunk_size=None)
        vs_chunked = VariogramScore(p=p, chunk_size=4)

        result_full = vs_full(x, y, mode="complete")
        result_chunked = vs_chunked(x, y, mode="complete")

        torch.testing.assert_close(result_chunked, result_full, rtol=1e-5, atol=1e-5)

    @pytest.mark.parametrize("chunk_size", [2, 4, 8, 16])
    @pytest.mark.unit
    def test_chunked_matches_full_different_chunk_sizes(self, chunk_size):
        """Test that different chunk sizes all produce the same result."""
        batch_size, n_samples, out_features, lat, lon = 2, 3, 1, 10, 10

        torch.manual_seed(456)
        x = torch.randn(batch_size, n_samples, out_features, lat, lon)
        y = torch.randn(batch_size, out_features, lat, lon)

        vs_full = VariogramScore(p=0.5, chunk_size=None)
        vs_chunked = VariogramScore(p=0.5, chunk_size=chunk_size)

        result_full = vs_full(x, y, mode="complete")
        result_chunked = vs_chunked(x, y, mode="complete")

        torch.testing.assert_close(result_chunked, result_full, rtol=1e-5, atol=1e-5)

    @pytest.mark.unit
    def test_chunked_identical_predictions(self):
        """Test chunked computation when predictions equal truth (score should be 0)."""
        batch_size, n_samples, out_features, lat, lon = 1, 3, 1, 8, 8

        torch.manual_seed(789)
        y = torch.randn(batch_size, out_features, lat, lon)
        x = y.unsqueeze(1).repeat(1, n_samples, 1, 1, 1)

        vs_chunked = VariogramScore(p=0.5, chunk_size=4)
        result = vs_chunked(x, y, mode="complete")

        assert torch.allclose(result, torch.zeros_like(result), atol=1e-6)

    @pytest.mark.unit
    def test_chunked_batch_consistency(self):
        """Test that batched chunked computation matches individual computations."""
        n_samples, out_features, lat, lon = 4, 1, 8, 8

        torch.manual_seed(101)
        x1 = torch.randn(1, n_samples, out_features, lat, lon)
        y1 = torch.randn(1, out_features, lat, lon)
        x2 = torch.randn(1, n_samples, out_features, lat, lon)
        y2 = torch.randn(1, out_features, lat, lon)

        vs_chunked = VariogramScore(p=0.5, chunk_size=4)

        vs1 = vs_chunked(x1, y1, mode="complete")
        vs2 = vs_chunked(x2, y2, mode="complete")

        x_batch = torch.cat([x1, x2], dim=0)
        y_batch = torch.cat([y1, y2], dim=0)
        vs_batch = vs_chunked(x_batch, y_batch, mode="complete")

        torch.testing.assert_close(vs_batch[0:1], vs1, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(vs_batch[1:2], vs2, rtol=1e-5, atol=1e-5)

    @pytest.mark.unit
    def test_chunked_exact_in_float64(self):
        """Verify chunked and full are mathematically identical using float64.

        Float32 differences (~1e-6 relative) are due to accumulation order, not
        algorithmic error. In float64 the difference drops to ~1e-12.
        """
        batch_size, n_samples, out_features, lat, lon = 2, 5, 2, 8, 8

        torch.manual_seed(42)
        x = torch.randn(batch_size, n_samples, out_features, lat, lon, dtype=torch.float64)
        y = torch.randn(batch_size, out_features, lat, lon, dtype=torch.float64)

        vs_full = VariogramScore(p=0.5, chunk_size=None)
        vs_chunked = VariogramScore(p=0.5, chunk_size=4)

        result_full = vs_full(x, y, mode="complete")
        result_chunked = vs_chunked(x, y, mode="complete")

        torch.testing.assert_close(result_chunked, result_full, rtol=1e-10, atol=1e-10)


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


class TestRBFScore:
    """Test cases for RBFScore class"""

    @pytest.mark.unit
    def test_rbf_score_simple_case(self):
        """Test RBF score computation for a simple case"""
        batch_size, n_samples, lat, lon, out_features = 2, 3, 4, 4, 1

        torch.manual_seed(42)
        x = torch.randn(batch_size, n_samples, out_features, lat, lon)
        y = torch.randn(batch_size, out_features, lat, lon)

        rbf_score_model = RBFScore(lengthscales=1.0)
        rbf = rbf_score_model(x, y, mode="complete")

        assert rbf.shape == (batch_size,)
        assert torch.isfinite(rbf).all()

    @pytest.mark.unit
    def test_rbf_score_perfect_prediction(self):
        """Test RBF score when all predictions equal the target"""
        batch_size, n_samples, lat, lon, out_features = 1, 3, 2, 2, 1

        torch.manual_seed(789)
        y = torch.randn(batch_size, out_features, lat, lon)
        x = y.unsqueeze(1).repeat(1, n_samples, 1, 1, 1)

        rbf_score_model = RBFScore(lengthscales=1.0)
        rbf = rbf_score_model(x, y, mode="complete")

        # Perfect prediction should give score close to -0.5
        assert torch.allclose(rbf, torch.ones_like(rbf) * -0.5, atol=1e-6)

    @pytest.mark.unit
    def test_rbf_score_multiple_lengthscales(self):
        """Test RBF score with multiple lengthscales"""
        batch_size, n_samples, lat, lon, out_features = 2, 4, 3, 3, 1

        torch.manual_seed(123)
        x = torch.randn(batch_size, n_samples, out_features, lat, lon)
        y = torch.randn(batch_size, out_features, lat, lon)

        rbf_score_model = RBFScore(lengthscales=[0.5, 1.0, 2.0])
        rbf = rbf_score_model(x, y, mode="complete")

        assert rbf.shape == (batch_size,)
        assert torch.isfinite(rbf).all()

    @pytest.mark.unit
    def test_rbf_score_per_var_mode(self):
        """Test RBF score in per_var mode"""
        batch_size, n_samples, lat, lon, out_features = 2, 3, 4, 4, 3

        torch.manual_seed(456)
        x = torch.randn(batch_size, n_samples, out_features, lat, lon)
        y = torch.randn(batch_size, out_features, lat, lon)

        rbf_score_model = RBFScore(lengthscales=1.0)
        rbf = rbf_score_model(x, y, mode="per_var")

        assert rbf.shape == (batch_size, out_features)
        assert torch.isfinite(rbf).all()

    @pytest.mark.unit
    def test_rbf_score_batch_consistency(self):
        """Test that batched computation gives same results as individual computations"""
        n_samples, lat, lon, out_features = 4, 3, 3, 1

        torch.manual_seed(101)
        x1 = torch.randn(1, n_samples, out_features, lat, lon)
        y1 = torch.randn(1, out_features, lat, lon)
        x2 = torch.randn(1, n_samples, out_features, lat, lon)
        y2 = torch.randn(1, out_features, lat, lon)

        rbf_score_model = RBFScore(lengthscales=1.0)

        rbf1 = rbf_score_model(x1, y1, mode="complete")
        rbf2 = rbf_score_model(x2, y2, mode="complete")

        x_batch = torch.cat([x1, x2], dim=0)
        y_batch = torch.cat([y1, y2], dim=0)
        rbf_batch = rbf_score_model(x_batch, y_batch, mode="complete")

        torch.testing.assert_close(rbf_batch[0:1], rbf1, rtol=1e-10, atol=1e-10)
        torch.testing.assert_close(rbf_batch[1:2], rbf2, rtol=1e-10, atol=1e-10)


class TestPatchwiseEnergyScore:
    """Test cases for PatchwiseEnergyScore class"""

    @pytest.mark.unit
    def test_patchwise_energy_score_simple_case(self):
        """Test patchwise energy score computation"""
        batch_size, n_samples, lat, lon, out_features = 2, 3, 8, 8, 1

        torch.manual_seed(42)
        x = torch.randn(batch_size, n_samples, out_features, lat, lon)
        y = torch.randn(batch_size, out_features, lat, lon)

        patchwise_es = PatchwiseEnergyScore(patch_size=3)
        es = patchwise_es(x, y, mode="complete")

        assert es.shape == (batch_size,)
        assert torch.isfinite(es).all()

    @pytest.mark.unit
    def test_patchwise_energy_score_per_var_mode(self):
        """Test patchwise energy score in per_var mode"""
        batch_size, n_samples, lat, lon, out_features = 2, 3, 8, 8, 3

        torch.manual_seed(123)
        x = torch.randn(batch_size, n_samples, out_features, lat, lon)
        y = torch.randn(batch_size, out_features, lat, lon)

        patchwise_es = PatchwiseEnergyScore(patch_size=3)
        es = patchwise_es(x, y, mode="per_var")

        assert es.shape == (batch_size, out_features)
        assert torch.isfinite(es).all()

    @pytest.mark.unit
    def test_patchwise_energy_score_different_patch_sizes(self):
        """Test patchwise energy score with different patch sizes"""
        batch_size, n_samples, lat, lon, out_features = 2, 3, 12, 12, 1

        torch.manual_seed(456)
        x = torch.randn(batch_size, n_samples, out_features, lat, lon)
        y = torch.randn(batch_size, out_features, lat, lon)

        for patch_size in [3, 5, 7]:
            patchwise_es = PatchwiseEnergyScore(patch_size=patch_size)
            es = patchwise_es(x, y, mode="complete")

            assert es.shape == (batch_size,)
            assert torch.isfinite(es).all()

    @pytest.mark.unit
    def test_patchwise_energy_score_normalization(self):
        """Test that normalization affects the score magnitude"""
        batch_size, n_samples, lat, lon, out_features = 2, 3, 8, 8, 1

        torch.manual_seed(789)
        x = torch.randn(batch_size, n_samples, out_features, lat, lon)
        y = torch.randn(batch_size, out_features, lat, lon)

        patchwise_es_normalized = PatchwiseEnergyScore(patch_size=3, normalize=True)
        patchwise_es_unnormalized = PatchwiseEnergyScore(patch_size=3, normalize=False)

        es_norm = patchwise_es_normalized(x, y, mode="complete")
        es_unnorm = patchwise_es_unnormalized(x, y, mode="complete")

        # Normalized and unnormalized should be different
        assert not torch.allclose(es_norm, es_unnorm)


class TestPatchwiseRBFScore:
    """Test cases for PatchwiseRBFScore class"""

    @pytest.mark.unit
    def test_patchwise_rbf_score_simple_case(self):
        """Test patchwise RBF score computation"""
        batch_size, n_samples, lat, lon, out_features = 2, 3, 8, 8, 1

        torch.manual_seed(42)
        x = torch.randn(batch_size, n_samples, out_features, lat, lon)
        y = torch.randn(batch_size, out_features, lat, lon)

        patchwise_rbf = PatchwiseRBFScore(patch_size=3)
        rbf = patchwise_rbf(x, y, mode="complete")

        assert rbf.shape == (batch_size,)
        assert torch.isfinite(rbf).all()

    @pytest.mark.unit
    def test_patchwise_rbf_score_per_var_mode(self):
        """Test patchwise RBF score in per_var mode"""
        batch_size, n_samples, lat, lon, out_features = 2, 3, 8, 8, 3

        torch.manual_seed(123)
        x = torch.randn(batch_size, n_samples, out_features, lat, lon)
        y = torch.randn(batch_size, out_features, lat, lon)

        patchwise_rbf = PatchwiseRBFScore(patch_size=3)
        rbf = patchwise_rbf(x, y, mode="per_var")

        assert rbf.shape == (batch_size, out_features)
        assert torch.isfinite(rbf).all()

    @pytest.mark.unit
    def test_patchwise_rbf_score_custom_lengthscales(self):
        """Test patchwise RBF score with custom lengthscales"""
        batch_size, n_samples, lat, lon, out_features = 2, 3, 8, 8, 1

        torch.manual_seed(456)
        x = torch.randn(batch_size, n_samples, out_features, lat, lon)
        y = torch.randn(batch_size, out_features, lat, lon)

        patchwise_rbf = PatchwiseRBFScore(patch_size=3, lengthscales=[1.0, 5.0, 10.0])
        rbf = patchwise_rbf(x, y, mode="complete")

        assert rbf.shape == (batch_size,)
        assert torch.isfinite(rbf).all()

    @pytest.mark.unit
    def test_patchwise_rbf_score_perfect_prediction(self):
        """Test patchwise RBF score when predictions equal target"""
        batch_size, n_samples, lat, lon, out_features = 1, 3, 8, 8, 1

        torch.manual_seed(789)
        y = torch.randn(batch_size, out_features, lat, lon)
        x = y.unsqueeze(1).repeat(1, n_samples, 1, 1, 1)

        patchwise_rbf = PatchwiseRBFScore(patch_size=3)
        rbf = patchwise_rbf(x, y, mode="complete")

        # Perfect prediction should give score close to -0.5
        assert torch.allclose(rbf, torch.ones_like(rbf) * -0.5, atol=1e-6)


class TestMultiPatchwiseRBFScore:
    """Test cases for MultiPatchwiseRBFScore class"""

    @pytest.mark.unit
    def test_multipatchwise_rbf_score_simple_case(self):
        """Test multi-patchwise RBF score computation"""
        batch_size, n_samples, lat, lon, out_features = 2, 3, 16, 16, 1

        torch.manual_seed(42)
        x = torch.randn(batch_size, n_samples, out_features, lat, lon)
        y = torch.randn(batch_size, out_features, lat, lon)

        multi_rbf = MultiScalePatchwiseRBFScore(blur_kernel_sizes=[3, 7], patch_size=3)
        rbf = multi_rbf(x, y, mode="complete")

        assert rbf.shape == (batch_size,)
        assert torch.isfinite(rbf).all()

    @pytest.mark.unit
    def test_multipatchwise_rbf_score_per_var_mode(self):
        """Test multi-patchwise RBF score in per_var mode"""
        batch_size, n_samples, lat, lon, out_features = 2, 3, 16, 16, 3

        torch.manual_seed(123)
        x = torch.randn(batch_size, n_samples, out_features, lat, lon)
        y = torch.randn(batch_size, out_features, lat, lon)

        multi_rbf = MultiScalePatchwiseRBFScore(blur_kernel_sizes=[3, 7], patch_size=3)
        rbf = multi_rbf(x, y, mode="per_var")

        assert rbf.shape == (batch_size, out_features)
        assert torch.isfinite(rbf).all()

    @pytest.mark.unit
    def test_multipatchwise_rbf_score_custom_lengthscales(self):
        """Test multi-patchwise RBF score with custom lengthscales per patch size"""
        batch_size, n_samples, lat, lon, out_features = 2, 3, 16, 16, 1

        torch.manual_seed(456)
        x = torch.randn(batch_size, n_samples, out_features, lat, lon)
        y = torch.randn(batch_size, out_features, lat, lon)

        multi_rbf = MultiScalePatchwiseRBFScore(
            blur_kernel_sizes=[3, 5],
            lengthscales=[1.0, 5.0],  # Different lengthscales per patch size
        )
        rbf = multi_rbf(x, y, mode="complete")

        assert rbf.shape == (batch_size,)
        assert torch.isfinite(rbf).all()


class TestMultiScaleRBFScore:
    """Test cases for MultiScaleRBFScore class"""

    @pytest.mark.unit
    def test_multiscale_rbf_score_simple_case(self):
        """Test multi-scale RBF score computation"""
        batch_size, n_samples, lat, lon, out_features = 2, 3, 16, 16, 1

        torch.manual_seed(42)
        x = torch.randn(batch_size, n_samples, out_features, lat, lon)
        y = torch.randn(batch_size, out_features, lat, lon)

        multi_scale_rbf = MultiScaleRBFScore(blur_kernel_sizes=[3, 7])
        rbf = multi_scale_rbf(x, y, mode="complete")

        assert rbf.shape == (batch_size,)
        assert torch.isfinite(rbf).all()
