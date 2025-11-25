"""Tests for the engression model components."""

import pytest
import torch

from genpp.models.engression.base import (
    StochasticLayer2D,
    StochasticResBlock2D,
)
from genpp.models.engression.cnn import (
    StochasticEncoder,
    StochasticDecoder,
    StochasticUNet,
)


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
            (2, 64, 32, 32, 16, 16),
            (1, 32, 16, 16, 24, 24),
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
