import pytest
import torch

from genpp.models.fm.fm_uvit import patchify, unpatchify


class TestPatchifyUnpatchify:
    @pytest.mark.unit
    @pytest.mark.parametrize(
        "batch_size,channels,height,width,patch_h,patch_w",
        [
            (1, 2, 32, 32, 4, 4),
            (2, 3, 64, 64, 8, 8),
            (4, 2, 40, 40, 4, 4),
            (1, 1, 16, 16, 2, 2),
            (3, 2, 48, 32, 4, 4),  # non-square image
            (2, 3, 32, 48, 8, 4),  # non-square patches
        ],
    )
    def test_patchify_shape(self, batch_size, channels, height, width, patch_h, patch_w):
        """Test that patchify produces the correct output shape."""
        imgs = torch.randn(batch_size, channels, height, width)
        patch_size = (patch_h, patch_w)

        patches = patchify(imgs, patch_size)

        num_patches = (height // patch_h) * (width // patch_w)
        patch_dim = patch_h * patch_w * channels

        assert patches.shape == (batch_size, num_patches, patch_dim)

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "batch_size,channels,height,width,patch_h,patch_w",
        [
            (1, 2, 32, 32, 4, 4),
            (2, 3, 64, 64, 8, 8),
            (4, 2, 40, 40, 4, 4),
            (1, 1, 16, 16, 2, 2),
            (3, 2, 48, 32, 4, 4),  # non-square image
            (2, 3, 32, 48, 8, 4),  # non-square patches
        ],
    )
    def test_unpatchify_is_inverse_of_patchify(
        self, batch_size, channels, height, width, patch_h, patch_w
    ):
        """Test that unpatchify correctly inverts patchify."""
        imgs = torch.randn(batch_size, channels, height, width)
        patch_size = (patch_h, patch_w)
        image_size = (height, width)

        # Apply patchify then unpatchify
        patches = patchify(imgs, patch_size)
        reconstructed = unpatchify(patches, patch_size, image_size, channels)

        # Check shape matches
        assert reconstructed.shape == imgs.shape

        # Check values match (should be exact since these are invertible operations)
        assert torch.allclose(reconstructed, imgs, atol=1e-6)

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "batch_size,channels,height,width,patch_h,patch_w",
        [
            (1, 2, 32, 32, 4, 4),
            (2, 3, 64, 64, 8, 8),
            (4, 2, 40, 40, 4, 4),
        ],
    )
    def test_unpatchify_shape(self, batch_size, channels, height, width, patch_h, patch_w):
        """Test that unpatchify produces the correct output shape."""
        num_patches = (height // patch_h) * (width // patch_w)
        patch_dim = patch_h * patch_w * channels
        patches = torch.randn(batch_size, num_patches, patch_dim)

        patch_size = (patch_h, patch_w)
        image_size = (height, width)

        imgs = unpatchify(patches, patch_size, image_size, channels)

        assert imgs.shape == (batch_size, channels, height, width)

    @pytest.mark.unit
    def test_invalid_dimensions_raise_assertion(self):
        """Test that invalid dimensions cause assertions to fail."""
        batch_size, channels, height, width = 2, 3, 32, 32
        patch_h, patch_w = 4, 4

        # Create patches with wrong dimensions
        num_patches = (height // patch_h) * (width // patch_w)
        wrong_patch_dim = patch_h * patch_w * channels + 1  # intentionally wrong
        patches = torch.randn(batch_size, num_patches, wrong_patch_dim)

        patch_size = (patch_h, patch_w)
        image_size = (height // patch_h, width // patch_w)

        with pytest.raises(AssertionError):
            unpatchify(patches, patch_size, image_size, channels)
