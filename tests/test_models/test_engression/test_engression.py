"""Tests for the engression model components."""

from functools import partial

import pytest
import torch

from genpp.models.cgm.engression.base import (
    StochasticLayer2D,
    StochasticResBlock2D,
)
from genpp.models.cgm.engression.cnn import (
    CNNEngressionModel,
    StochasticDecoder,
    StochasticEncoder,
    StochasticUNet,
)
from genpp.models.scores import EnergyScore


class TestStochasticLayer2D:
    """Tests for StochasticLayer2D."""

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "batch_size,in_channels,out_channels,height,width,noise_channels",
        [
            (2, 16, 32, 32, 32, 16),
            (1, 8, 16, 48, 32, 8),
            (4, 32, 64, 16, 16, 32),
        ],
    )
    def test_output_shape(
        self, batch_size, in_channels, out_channels, height, width, noise_channels
    ):
        """Test that StochasticLayer2D produces correct output shape."""
        layer = StochasticLayer2D(
            in_channels=in_channels,
            out_channels=out_channels,
            noise_channels=noise_channels,
            kernel_size=3,
            add_bn=True,
            activation=torch.nn.ReLU(),
        )
        x = torch.randn(batch_size, in_channels, height, width)
        out = layer(x)
        assert out.shape == (batch_size, out_channels, height, width)

    @pytest.mark.unit
    def test_stochastic_output(self):
        """Test that StochasticLayer2D produces different outputs for same input."""
        layer = StochasticLayer2D(
            in_channels=16,
            out_channels=32,
            noise_channels=16,
        )
        layer.eval()  # Ensure consistent behavior
        x = torch.randn(2, 16, 32, 32)

        out1 = layer(x)
        out2 = layer(x)

        # Outputs should be different due to noise injection
        assert not torch.allclose(out1, out2)


class TestStochasticResBlock2D:
    """Tests for StochasticResBlock2D."""

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "batch_size,channels,height,width,noise_channels",
        [
            (2, 32, 32, 32, 16),
            (1, 64, 48, 32, 8),
            (4, 16, 16, 16, 32),
        ],
    )
    def test_output_shape(self, batch_size, channels, height, width, noise_channels):
        """Test that StochasticResBlock2D produces correct output shape."""
        block = StochasticResBlock2D(
            channels=channels,
            noise_channels=noise_channels,
        )
        x = torch.randn(batch_size, channels, height, width)
        out = block(x)
        assert out.shape == (batch_size, channels, height, width)

    @pytest.mark.unit
    def test_residual_connection(self):
        """Test that residual connection works."""
        block = StochasticResBlock2D(
            channels=32,
            noise_channels=0,  # No noise to test pure residual
        )
        x = torch.randn(2, 32, 16, 16)
        out = block(x)

        # Output should not be zero due to residual connection
        assert torch.any(out != 0)


class TestStochasticEncoder:
    """Tests for StochasticEncoder."""

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "batch_size,in_channels,out_channels,height,width",
        [
            (2, 32, 64, 32, 32),
            (1, 16, 32, 48, 48),
        ],
    )
    def test_output_shape(self, batch_size, in_channels, out_channels, height, width):
        """Test that StochasticEncoder produces correct output shapes."""
        encoder = StochasticEncoder(
            in_channels=in_channels,
            out_channels=out_channels,
            noise_channels=16,
            num_layers=2,
        )
        x = torch.randn(batch_size, in_channels, height, width)
        downsampled, skip = encoder(x)

        # Downsampled should be half the spatial size
        assert downsampled.shape == (batch_size, out_channels, height // 2, width // 2)
        # Skip should have same spatial size as input
        assert skip.shape == (batch_size, out_channels, height, width)


class TestStochasticDecoder:
    """Tests for StochasticDecoder."""

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "batch_size,in_channels,skip_channels,out_channels,height,width",
        [
            (2, 64, 64, 32, 16, 16),  # skip_channels = in_channels (typical UNet)
            (1, 128, 128, 64, 8, 8),
        ],
    )
    def test_output_shape(
        self, batch_size, in_channels, skip_channels, out_channels, height, width
    ):
        """Test that StochasticDecoder produces correct output shapes."""
        decoder = StochasticDecoder(
            in_channels=in_channels,
            skip_channels=skip_channels,
            out_channels=out_channels,
            noise_channels=16,
            num_layers=2,
        )
        # Input is the downsampled feature map
        x = torch.randn(batch_size, in_channels, height, width)
        # Skip has skip_channels and double the spatial size
        skip = torch.randn(batch_size, skip_channels, height * 2, width * 2)

        out = decoder(x, skip)
        assert out.shape == (batch_size, out_channels, height * 2, width * 2)


class TestStochasticUNet:
    """Tests for StochasticUNet."""

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "batch_size,in_channels,out_channels,height,width,channels",
        [
            (2, 16, 2, 32, 32, [32, 64]),
            (1, 8, 4, 64, 64, [32, 64, 128]),
        ],
    )
    def test_output_shape(self, batch_size, in_channels, out_channels, height, width, channels):
        """Test that StochasticUNet produces correct output shape."""
        unet = StochasticUNet(
            in_channels=in_channels,
            out_channels=out_channels,
            channels=channels,
            noise_channels=16,
        )
        x = torch.randn(batch_size, in_channels, height, width)
        out = unet(x)
        assert out.shape == (batch_size, out_channels, height, width)

    @pytest.mark.unit
    def test_sample_method(self):
        """Test that sample method produces correct shape."""
        unet = StochasticUNet(
            in_channels=16,
            out_channels=2,
            channels=[32, 64],
            noise_channels=16,
        )
        batch_size, n_samples = 2, 5
        x = torch.randn(batch_size, 16, 32, 32)

        samples = unet.sample(x, n_samples)
        assert samples.shape == (batch_size, n_samples, 2, 32, 32)

    @pytest.mark.unit
    def test_samples_are_different(self):
        """Test that different samples are actually different."""
        unet = StochasticUNet(
            in_channels=16,
            out_channels=2,
            channels=[32, 64],
            noise_channels=16,
        )
        x = torch.randn(1, 16, 32, 32)

        samples = unet.sample(x, n_samples=3)

        # Check that samples are different from each other
        assert not torch.allclose(samples[0, 0], samples[0, 1])
        assert not torch.allclose(samples[0, 1], samples[0, 2])


class TestCNNEngressionModel:
    """Tests for CNNEngressionModel."""

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "batch_size,height,width,pred_channels,aux_channels,meta_channels,out_channels,n_samples,padding",
        [
            (2, 48, 32, 2, 8, 6, 2, 10, [2, 2, 4, 4]),
            (1, 64, 64, 4, 10, 4, 4, 5, [4, 4, 4, 4]),
        ],
    )
    def test_forward_output_shape(
        self,
        batch_size,
        height,
        width,
        pred_channels,
        aux_channels,
        meta_channels,
        out_channels,
        n_samples,
        padding,
    ):
        """Test that CNNEngressionModel forward pass produces correct output shape."""
        # Calculate expected output dimensions after cropping
        expected_height = height - padding[2] - padding[3]
        expected_width = width - padding[0] - padding[1]

        embedding_dim = 5
        in_channels = pred_channels + aux_channels + meta_channels

        # Create the model
        model = CNNEngressionModel(
            in_channels=in_channels,
            out_channels=out_channels,
            height=height,
            width=width,
            embedding_dim=embedding_dim,
            channels=[32, 64],
            noise_channels=16,
            num_layers_per_block=2,
            use_resblock=False,
            kernel_size=3,
            add_bn=True,
            n_samples=n_samples,
            padding=padding,
            optimizer=partial(torch.optim.Adam, lr=1e-3),
            lr_scheduler={"scheduler": None},  # pyright: ignore[reportArgumentType]
            internal_td_scaling="learned",
            use_rescaler=False,
            loss_fn=EnergyScore(),
        )

        # Mark TD scaling as fitted to avoid errors during forward pass
        model.internal_td_scaling.is_fitted = torch.tensor(True)
        # Set up the internal TD scaling model for out_channels variables
        model.internal_td_scaling.model = torch.nn.Linear(1, out_channels)

        # Create input tensors with correct shapes
        # Note: predicted_vars must have out_channels since scale is computed based on it
        x = {
            "predicted_vars": torch.randn(batch_size, out_channels, height, width),
            "auxiliary_vars": torch.randn(batch_size, aux_channels, height, width),
            "meta_vars": torch.randn(batch_size, meta_channels, height, width),
            "pixel_idx": torch.zeros(batch_size, 1, height, width, dtype=torch.long),
        }
        # Fill pixel_idx properly
        for i in range(height):
            for j in range(width):
                x["pixel_idx"][:, 0, i, j] = i * width + j

        td = torch.rand(batch_size)

        # Forward pass
        out = model.forward(x, td, n_samples=n_samples)

        # Check output shape: [batch, n_samples, out_channels, cropped_height, cropped_width]
        assert out.shape == (
            batch_size,
            n_samples,
            out_channels,
            expected_height,
            expected_width,
        ), (
            f"Expected shape {(batch_size, n_samples, out_channels, expected_height, expected_width)}, got {out.shape}"
        )

    @pytest.mark.unit
    def test_samples_are_stochastic(self):
        """Test that multiple forward passes produce different samples due to noise injection."""
        batch_size = 2
        height, width = 32, 32
        aux_channels, meta_channels = 4, 2
        out_channels = 2
        n_samples = 5
        padding = [2, 2, 2, 2]
        embedding_dim = 5

        model = CNNEngressionModel(
            in_channels=out_channels + aux_channels + meta_channels,
            out_channels=out_channels,
            height=height,
            width=width,
            embedding_dim=embedding_dim,
            channels=[32, 64],
            noise_channels=16,
            num_layers_per_block=2,
            use_resblock=False,
            kernel_size=3,
            add_bn=True,
            n_samples=n_samples,
            padding=padding,
            optimizer=partial(torch.optim.Adam, lr=1e-3),
            lr_scheduler={"scheduler": None},  # pyright: ignore[reportArgumentType]
            internal_td_scaling="learned",
            use_rescaler=False,
            loss_fn=EnergyScore(),
        )

        model.internal_td_scaling.is_fitted = torch.tensor(True)
        # Set up the internal TD scaling model for out_channels variables
        model.internal_td_scaling.model = torch.nn.Linear(1, out_channels)

        x = {
            "predicted_vars": torch.randn(batch_size, out_channels, height, width),
            "auxiliary_vars": torch.randn(batch_size, aux_channels, height, width),
            "meta_vars": torch.randn(batch_size, meta_channels, height, width),
            "pixel_idx": torch.zeros(batch_size, 1, height, width, dtype=torch.long),
        }
        for i in range(height):
            for j in range(width):
                x["pixel_idx"][:, 0, i, j] = i * width + j

        td = torch.rand(batch_size)

        out = model.forward(x, td, n_samples=n_samples)

        # Check that different samples are different (due to stochastic noise)
        assert not torch.allclose(out[0, 0], out[0, 1]), (
            "Different samples should be different due to noise injection"
        )
