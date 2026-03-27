import pytest
import torch

from genpp.models.layers import CropND
from genpp.preproc.transforms import Pad


@pytest.mark.unit
@pytest.mark.parametrize(
    "pad_lat_left, pad_lat_right, pad_lon_top, pad_lon_bottom",
    [
        (0, 0, 0, 0),  # No padding
        (0, 1, 1, 1),  # Exactly one value is 0 (lat_left)
        (1, 0, 1, 1),  # Exactly one value is 0 (lat_right)
        (1, 1, 0, 1),  # Exactly one value is 0 (lon_top)
        (1, 1, 1, 0),  # Exactly one value is 0 (lon_bottom)
        (1, 1, 1, 1),  # Symmetric padding
        (2, 3, 4, 5),  # Asymmetric padding
        (2000, 30000, 4000, 500),  # Large padding
    ],
)
def test_cropnd_inverse_of_pad(pad_lat_left, pad_lat_right, pad_lon_top, pad_lon_bottom):
    """Test that CropND is the inverse of Pad for a given padding configuration."""
    padding = (pad_lat_left, pad_lat_right, pad_lon_top, pad_lon_bottom)
    print(padding)
    # Create a random tensor with shape (batch, channels, lon, lat)
    # Ensure dimensions are large enough to avoid negative slicing
    size = (2, 3, 10, 8)
    original = torch.randn(size)

    # Apply Pad
    pad_transform = Pad(padding)
    padded = pad_transform.transform(original)
    print(f"Padded shape: {padded.shape}")

    # Apply CropND with the same padding
    crop_layer = CropND(padding)
    cropped = crop_layer(padded)
    print(f"Cropped shape: {cropped.shape}")

    # Assert that cropped tensor matches the original
    assert torch.allclose(original, cropped), f"Failed for padding {padding}"
