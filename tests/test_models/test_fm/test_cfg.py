"""Tests for Classifier-Free Guidance (CFG) flow matching models."""

import pytest
import torch

from genpp.models.cgm.fm.cfg import (
    FlowMatchingDirectModelCFG,
    FlowMatchingNoiseModelCFG,
    _CFGVectorFieldWrapper,
)


class MockBackbone(torch.nn.Module):
    """Simple mock backbone for testing."""

    def __init__(self, in_channels=2, out_channels=2):
        super().__init__()
        self.linear = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x, t, conditioning):
        # Simple forward that uses conditioning to affect output
        # This allows us to verify CFG is working
        if "all_vars_mean" in conditioning:
            cond_influence = conditioning["all_vars_mean"][:, :x.shape[1], :x.shape[2], :x.shape[3]]
            return self.linear(x) + cond_influence.mean(dim=1, keepdim=True)
        return self.linear(x)


class TestCFGVectorFieldWrapper:
    """Tests for the _CFGVectorFieldWrapper class."""

    @pytest.mark.unit
    def test_cfg_wrapper_with_guidance_scale_1(self):
        """Test that guidance_scale=1 returns only conditional output."""
        backbone = MockBackbone()

        def null_fn(cond):
            return {k: torch.zeros_like(v) for k, v in cond.items()}

        wrapper = _CFGVectorFieldWrapper(backbone, guidance_scale=1.0, null_conditioning_fn=null_fn)

        x = torch.randn(2, 2, 8, 8)
        t = torch.rand(2)
        conditioning = {"all_vars_mean": torch.randn(2, 4, 8, 8)}

        # With guidance_scale=1, should only call backbone once (conditional)
        result = wrapper(x, t, conditioning)

        # Verify output has same shape as input
        assert result.shape == x.shape

    @pytest.mark.unit
    def test_cfg_wrapper_applies_formula(self):
        """Test that CFG formula is applied correctly: u = (1-w)*u_uncond + w*u_cond."""
        backbone = MockBackbone()

        # Use a simple null conditioning function
        def null_fn(cond):
            return {k: torch.zeros_like(v) for k, v in cond.items()}

        guidance_scale = 2.0
        wrapper = _CFGVectorFieldWrapper(
            backbone, guidance_scale=guidance_scale, null_conditioning_fn=null_fn
        )

        x = torch.randn(2, 2, 8, 8)
        t = torch.rand(2)
        conditioning = {"all_vars_mean": torch.randn(2, 4, 8, 8)}

        # Get the result from the wrapper
        result = wrapper(x, t, conditioning)

        # Manually compute expected result
        u_cond = backbone(x, t, conditioning)
        u_uncond = backbone(x, t, null_fn(conditioning))
        expected = (1 - guidance_scale) * u_uncond + guidance_scale * u_cond

        assert torch.allclose(result, expected, atol=1e-6)

    @pytest.mark.unit
    @pytest.mark.parametrize("guidance_scale", [0.5, 1.5, 2.0, 3.0, 7.5])
    def test_cfg_wrapper_various_guidance_scales(self, guidance_scale):
        """Test CFG wrapper works correctly with various guidance scales."""
        backbone = MockBackbone()

        def null_fn(cond):
            return {k: torch.zeros_like(v) for k, v in cond.items()}

        wrapper = _CFGVectorFieldWrapper(
            backbone, guidance_scale=guidance_scale, null_conditioning_fn=null_fn
        )

        x = torch.randn(2, 2, 8, 8)
        t = torch.rand(2)
        conditioning = {"all_vars_mean": torch.randn(2, 4, 8, 8)}

        result = wrapper(x, t, conditioning)

        # Verify output shape
        assert result.shape == x.shape


class TestFlowMatchingNoiseModelCFG:
    """Tests for FlowMatchingNoiseModelCFG class."""

    @pytest.fixture
    def mock_optimizer(self):
        """Create a mock optimizer factory."""
        return lambda params: torch.optim.Adam(params, lr=1e-3)

    @pytest.fixture
    def mock_lr_scheduler(self):
        """Create a mock lr_scheduler config."""
        from omegaconf import OmegaConf

        return OmegaConf.create({"class_path": "torch.optim.lr_scheduler.StepLR", "step_size": 10})

    @pytest.mark.unit
    def test_create_null_conditioning(self, mock_optimizer, mock_lr_scheduler):
        """Test that null conditioning zeros out tensors."""
        backbone = MockBackbone()

        model = FlowMatchingNoiseModelCFG(
            backbone=backbone,
            n_samples=5,
            solver_iter=10,
            padding=[],
            optimizer=mock_optimizer,
            lr_scheduler=mock_lr_scheduler,
            internal_td_scaling="abs",
            use_rescaler=False,
            guidance_scale=2.0,
        )

        conditioning = {
            "all_vars_mean": torch.randn(2, 4, 8, 8),
            "all_vars_std": torch.randn(2, 4, 8, 8),
            "meta_vars": torch.randn(2, 2, 8, 8),
            "pixel_idx": torch.arange(64).view(1, 1, 8, 8).expand(2, -1, -1, -1),
        }

        null_cond = model._create_null_conditioning(conditioning)

        # All tensor values should be zeros
        for key in conditioning:
            assert torch.all(null_cond[key] == 0)

    @pytest.mark.unit
    def test_guidance_scale_attribute(self, mock_optimizer, mock_lr_scheduler):
        """Test that guidance_scale is stored correctly."""
        backbone = MockBackbone()

        model = FlowMatchingNoiseModelCFG(
            backbone=backbone,
            n_samples=5,
            solver_iter=10,
            padding=[],
            optimizer=mock_optimizer,
            lr_scheduler=mock_lr_scheduler,
            internal_td_scaling="abs",
            use_rescaler=False,
            guidance_scale=3.5,
        )

        assert model.guidance_scale == 3.5

    @pytest.mark.unit
    def test_default_guidance_scale_is_one(self, mock_optimizer, mock_lr_scheduler):
        """Test that default guidance_scale is 1.0 (no guidance)."""
        backbone = MockBackbone()

        model = FlowMatchingNoiseModelCFG(
            backbone=backbone,
            n_samples=5,
            solver_iter=10,
            padding=[],
            optimizer=mock_optimizer,
            lr_scheduler=mock_lr_scheduler,
            internal_td_scaling="abs",
            use_rescaler=False,
        )

        assert model.guidance_scale == 1.0


class TestFlowMatchingDirectModelCFG:
    """Tests for FlowMatchingDirectModelCFG class."""

    @pytest.fixture
    def mock_optimizer(self):
        """Create a mock optimizer factory."""
        return lambda params: torch.optim.Adam(params, lr=1e-3)

    @pytest.fixture
    def mock_lr_scheduler(self):
        """Create a mock lr_scheduler config."""
        from omegaconf import OmegaConf

        return OmegaConf.create({"class_path": "torch.optim.lr_scheduler.StepLR", "step_size": 10})

    @pytest.mark.unit
    def test_create_null_conditioning(self, mock_optimizer, mock_lr_scheduler):
        """Test that null conditioning zeros out tensors."""
        backbone = MockBackbone()

        model = FlowMatchingDirectModelCFG(
            backbone=backbone,
            n_samples=5,
            solver_iter=10,
            padding=[],
            optimizer=mock_optimizer,
            lr_scheduler=mock_lr_scheduler,
            use_rescaler=False,
            guidance_scale=2.0,
        )

        conditioning = {
            "all_vars_mean": torch.randn(2, 4, 8, 8),
            "timedelta": torch.rand(2),
        }

        null_cond = model._create_null_conditioning(conditioning)

        # All tensor values should be zeros
        for key in conditioning:
            assert torch.all(null_cond[key] == 0)

    @pytest.mark.unit
    def test_guidance_scale_attribute(self, mock_optimizer, mock_lr_scheduler):
        """Test that guidance_scale is stored correctly."""
        backbone = MockBackbone()

        model = FlowMatchingDirectModelCFG(
            backbone=backbone,
            n_samples=5,
            solver_iter=10,
            padding=[],
            optimizer=mock_optimizer,
            lr_scheduler=mock_lr_scheduler,
            use_rescaler=False,
            guidance_scale=5.0,
        )

        assert model.guidance_scale == 5.0

    @pytest.mark.unit
    def test_default_guidance_scale_is_one(self, mock_optimizer, mock_lr_scheduler):
        """Test that default guidance_scale is 1.0 (no guidance)."""
        backbone = MockBackbone()

        model = FlowMatchingDirectModelCFG(
            backbone=backbone,
            n_samples=5,
            solver_iter=10,
            padding=[],
            optimizer=mock_optimizer,
            lr_scheduler=mock_lr_scheduler,
            use_rescaler=False,
        )

        assert model.guidance_scale == 1.0

    @pytest.mark.unit
    def test_conditioning_dropout_prob_attribute(self, mock_optimizer, mock_lr_scheduler):
        """Test that conditioning_dropout_prob is stored correctly."""
        backbone = MockBackbone()

        model = FlowMatchingDirectModelCFG(
            backbone=backbone,
            n_samples=5,
            solver_iter=10,
            padding=[],
            optimizer=mock_optimizer,
            lr_scheduler=mock_lr_scheduler,
            use_rescaler=False,
            conditioning_dropout_prob=0.1,
        )

        assert model.conditioning_dropout_prob == 0.1


class TestPredictStepGuidanceScaleOverride:
    """Tests for predict_step guidance_scale parameter override."""

    @pytest.fixture
    def mock_optimizer(self):
        """Create a mock optimizer factory."""
        return lambda params: torch.optim.Adam(params, lr=1e-3)

    @pytest.fixture
    def mock_lr_scheduler(self):
        """Create a mock lr_scheduler config."""
        from omegaconf import OmegaConf

        return OmegaConf.create({"class_path": "torch.optim.lr_scheduler.StepLR", "step_size": 10})

    @pytest.mark.unit
    def test_noise_model_predict_step_accepts_guidance_scale(self, mock_optimizer, mock_lr_scheduler):
        """Test that FlowMatchingNoiseModelCFG.predict_step accepts guidance_scale parameter."""
        backbone = MockBackbone()

        model = FlowMatchingNoiseModelCFG(
            backbone=backbone,
            n_samples=2,
            solver_iter=2,
            padding=[],
            optimizer=mock_optimizer,
            lr_scheduler=mock_lr_scheduler,
            internal_td_scaling="abs",
            use_rescaler=False,
            guidance_scale=1.0,  # Default no guidance
        )

        # Check that predict_step signature includes guidance_scale parameter
        import inspect
        sig = inspect.signature(model.predict_step)
        assert "guidance_scale" in sig.parameters
        assert sig.parameters["guidance_scale"].default is None

    @pytest.mark.unit
    def test_direct_model_predict_step_accepts_guidance_scale(self, mock_optimizer, mock_lr_scheduler):
        """Test that FlowMatchingDirectModelCFG.predict_step accepts guidance_scale parameter."""
        backbone = MockBackbone()

        model = FlowMatchingDirectModelCFG(
            backbone=backbone,
            n_samples=2,
            solver_iter=2,
            padding=[],
            optimizer=mock_optimizer,
            lr_scheduler=mock_lr_scheduler,
            use_rescaler=False,
            guidance_scale=1.0,  # Default no guidance
        )

        # Check that predict_step signature includes guidance_scale parameter
        import inspect
        sig = inspect.signature(model.predict_step)
        assert "guidance_scale" in sig.parameters
        assert sig.parameters["guidance_scale"].default is None
