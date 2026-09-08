"""Intensity normalization shared by CNN training and inference."""

from __future__ import annotations

import numpy as np


def normalize(image: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Min-max normalize one finite, nonconstant 2-D image."""
    array = np.asarray(image, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"Expected a 2-D image, received shape {array.shape}.")
    if not np.isfinite(array).all():
        raise ValueError("The image contains NaN or infinite values.")
    minimum = float(array.min())
    maximum = float(array.max())
    if maximum <= minimum:
        raise ValueError("A constant image cannot be min-max normalized.")
    return (array - minimum) / (maximum - minimum), minimum, maximum


def denormalize(image: np.ndarray, minimum: float, maximum: float) -> np.ndarray:
    """Undo :func:`normalize` using the recorded input range."""
    if not np.isfinite([minimum, maximum]).all() or maximum <= minimum:
        raise ValueError("A valid finite normalization range is required.")
    return np.asarray(image, dtype=np.float32) * (maximum - minimum) + minimum

