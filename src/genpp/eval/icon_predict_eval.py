"""Backward-compatible exports for ICON prediction evaluation helpers."""

import torch


def _rescale_y(y: torch.Tensor, reverse_modules: list) -> torch.Tensor:
	"""Rescale normalized y values back to original space.

	Keeps the legacy helper import path stable for tests and older scripts,
	without importing the full evaluation pipeline on module import.
	"""
	y_rescaled = y.clone()
	for i, mod in enumerate(reverse_modules):
		y_rescaled[..., i, :, :] = y_rescaled[..., i, :, :] * mod.scale + mod.mean
	return y_rescaled


__all__ = ["_rescale_y"]
