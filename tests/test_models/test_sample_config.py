"""Tests for n_samples configuration (train vs predict)."""

from functools import partial

import pytest
import torch

from genpp.models.cgm import CNNChenNoiseModel, CNNEngressionNoiseModel
from genpp.models.loss import EnergyScore


class TestNSamplesConfiguration:
    """Test n_samples configuration for generative models."""

    @pytest.mark.unit
    def test_chen_backwards_compatibility_n_samples_only(self):
        """Test Chen model with only n_samples (backwards compatibility)."""
        model = CNNChenNoiseModel(
            in_features=2,
            meta_features=6,
            out_features=2,
            width=32,
            height=32,
            noise_dim=5,
            embedding_dim=5,
            n_samples=50,  # Old API
            final_activation=torch.nn.Identity(),
            loss_fn=EnergyScore(),
            optimizer=partial(torch.optim.Adam, lr=1e-3),
            lr_scheduler={"scheduler": None},  # type: ignore[arg-type]
            internal_td_scaling="learned",
            use_rescaler=False,
            rescaler=None,
            padding=(2, 2, 2, 2),
        )
        
        # Both should be set to 50
        assert model.n_samples_train == 50
        assert model.n_samples_predict == 50
        assert model.n_samples == 50

    @pytest.mark.unit
    def test_chen_separate_train_predict_samples(self):
        """Test Chen model with separate train/predict samples."""
        model = CNNChenNoiseModel(
            in_features=2,
            meta_features=6,
            out_features=2,
            width=32,
            height=32,
            noise_dim=5,
            embedding_dim=5,
            n_samples=50,  # Default
            n_samples_train=30,  # Override for training
            n_samples_predict=100,  # Override for prediction
            final_activation=torch.nn.Identity(),
            loss_fn=EnergyScore(),
            optimizer=partial(torch.optim.Adam, lr=1e-3),
            lr_scheduler={"scheduler": None},  # type: ignore[arg-type]
            internal_td_scaling="learned",
            use_rescaler=False,
            rescaler=None,
            padding=(2, 2, 2, 2),
        )
        
        assert model.n_samples_train == 30
        assert model.n_samples_predict == 100
        assert model.n_samples == 100  # Should equal predict

    @pytest.mark.unit
    def test_chen_new_api_without_n_samples(self):
        """Test Chen model with new API (no n_samples)."""
        model = CNNChenNoiseModel(
            in_features=2,
            meta_features=6,
            out_features=2,
            width=32,
            height=32,
            noise_dim=5,
            embedding_dim=5,
            n_samples_train=25,
            n_samples_predict=75,
            final_activation=torch.nn.Identity(),
            loss_fn=EnergyScore(),
            optimizer=partial(torch.optim.Adam, lr=1e-3),
            lr_scheduler={"scheduler": None},  # type: ignore[arg-type]
            internal_td_scaling="learned",
            use_rescaler=False,
            rescaler=None,
            padding=(2, 2, 2, 2),
        )
        
        assert model.n_samples_train == 25
        assert model.n_samples_predict == 75
        assert model.n_samples == 75

    @pytest.mark.unit
    def test_engression_backwards_compatibility_n_samples_only(self):
        """Test Engression model with only n_samples (backwards compatibility)."""
        model = CNNEngressionNoiseModel(
            in_channels=10,
            out_channels=2,
            height=32,
            width=32,
            embedding_dim=5,
            channels=[32, 64],
            noise_channels=16,
            num_layers_per_block=2,
            use_resblock=False,
            kernel_size=3,
            add_bn=True,
            n_samples=50,  # Old API
            padding=[2, 2, 2, 2],
            optimizer=partial(torch.optim.Adam, lr=1e-3),
            lr_scheduler={"scheduler": None},  # type: ignore[arg-type]
            internal_td_scaling="learned",
            use_rescaler=False,
            loss_fn=EnergyScore(),
        )
        
        # Both should be set to 50
        assert model.n_samples_train == 50
        assert model.n_samples_predict == 50
        assert model.n_samples == 50

    @pytest.mark.unit
    def test_engression_separate_train_predict_samples(self):
        """Test Engression model with separate train/predict samples."""
        model = CNNEngressionNoiseModel(
            in_channels=10,
            out_channels=2,
            height=32,
            width=32,
            embedding_dim=5,
            channels=[32, 64],
            noise_channels=16,
            num_layers_per_block=2,
            use_resblock=False,
            kernel_size=3,
            add_bn=True,
            n_samples=50,  # Default
            n_samples_train=20,  # Override for training
            n_samples_predict=80,  # Override for prediction
            padding=[2, 2, 2, 2],
            optimizer=partial(torch.optim.Adam, lr=1e-3),
            lr_scheduler={"scheduler": None},  # type: ignore[arg-type]
            internal_td_scaling="learned",
            use_rescaler=False,
            loss_fn=EnergyScore(),
        )
        
        assert model.n_samples_train == 20
        assert model.n_samples_predict == 80
        assert model.n_samples == 80  # Should equal predict

    @pytest.mark.unit
    def test_chen_forward_respects_n_samples_train(self):
        """Test that Chen forward generates correct number of samples during training."""
        model = CNNChenNoiseModel(
            in_features=2,
            meta_features=6,
            out_features=2,
            width=32,
            height=32,
            noise_dim=5,
            embedding_dim=5,
            n_samples_train=10,
            n_samples_predict=20,
            final_activation=torch.nn.Identity(),
            loss_fn=EnergyScore(),
            optimizer=partial(torch.optim.Adam, lr=1e-3),
            lr_scheduler={"scheduler": None},  # type: ignore[arg-type]
            internal_td_scaling="learned",
            use_rescaler=False,
            rescaler=None,
            padding=(2, 2, 2, 2),
        )
        
        # Set up TD scaling
        model.internal_td_scaling.is_fitted = torch.tensor(True)
        model.internal_td_scaling.model = torch.nn.Linear(1, 2)
        
        batch_size = 2
        x = {
            "predicted_vars": torch.randn(batch_size, 2, 32, 32),
            "auxiliary_vars": torch.randn(batch_size, 2, 32, 32),
            "meta_vars": torch.randn(batch_size, 6, 32, 32),
            "pixel_idx": torch.zeros(batch_size, 1, 32, 32, dtype=torch.long),
        }
        td = torch.rand(batch_size)
        
        # Test with train samples
        out_train = model.forward(x, td, n_samples=model.n_samples_train)
        assert out_train.shape[1] == 10, f"Expected 10 samples, got {out_train.shape[1]}"
        
        # Test with predict samples
        out_predict = model.forward(x, td, n_samples=model.n_samples_predict)
        assert out_predict.shape[1] == 20, f"Expected 20 samples, got {out_predict.shape[1]}"

    @pytest.mark.unit
    def test_engression_forward_respects_n_samples_train(self):
        """Test that Engression forward generates correct number of samples."""
        model = CNNEngressionNoiseModel(
            in_channels=10,
            out_channels=2,
            height=32,
            width=32,
            embedding_dim=5,
            channels=[32, 64],
            noise_channels=16,
            num_layers_per_block=2,
            use_resblock=False,
            kernel_size=3,
            add_bn=True,
            n_samples_train=15,
            n_samples_predict=25,
            padding=[2, 2, 2, 2],
            optimizer=partial(torch.optim.Adam, lr=1e-3),
            lr_scheduler={"scheduler": None},  # type: ignore[arg-type]
            internal_td_scaling="learned",
            use_rescaler=False,
            loss_fn=EnergyScore(),
        )
        
        # Set up TD scaling
        model.internal_td_scaling.is_fitted = torch.tensor(True)
        model.internal_td_scaling.model = torch.nn.Linear(1, 2)
        
        batch_size = 2
        x = {
            "predicted_vars": torch.randn(batch_size, 2, 32, 32),
            "auxiliary_vars": torch.randn(batch_size, 4, 32, 32),
            "meta_vars": torch.randn(batch_size, 4, 32, 32),
            "pixel_idx": torch.zeros(batch_size, 1, 32, 32, dtype=torch.long),
        }
        td = torch.rand(batch_size)
        
        # Test with train samples
        out_train = model.forward(x, td, n_samples=model.n_samples_train)
        assert out_train.shape[1] == 15, f"Expected 15 samples, got {out_train.shape[1]}"
        
        # Test with predict samples
        out_predict = model.forward(x, td, n_samples=model.n_samples_predict)
        assert out_predict.shape[1] == 25, f"Expected 25 samples, got {out_predict.shape[1]}"
