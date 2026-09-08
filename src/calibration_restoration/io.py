"""TIFF discovery, validation, and safe output encoding."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile


TIFF_SUFFIXES = {".tif", ".tiff"}


def discover_tiffs(folder: str | Path) -> list[Path]:
    root = Path(folder)
    if not root.is_dir():
        raise NotADirectoryError(f"Input folder does not exist: {root}")
    files = sorted(
        (path for path in root.iterdir() if path.is_file() and path.suffix.lower() in TIFF_SUFFIXES),
        key=lambda path: path.name.casefold(),
    )
    if not files:
        raise FileNotFoundError(f"No TIFF images were found in: {root}")
    return files


def read_tiff(path: str | Path) -> tuple[np.ndarray, np.dtype]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    original = np.asarray(tifffile.imread(source))
    if original.ndim != 2:
        raise ValueError(f"Expected a single-band 2-D TIFF, received {original.shape}.")
    if not np.issubdtype(original.dtype, np.number):
        raise TypeError(f"Unsupported TIFF dtype: {original.dtype}")
    image = original.astype(np.float32)
    if not np.isfinite(image).all():
        raise ValueError("The TIFF contains NaN or infinite values.")
    return image, original.dtype


def encode_like(values: np.ndarray, dtype: np.dtype) -> np.ndarray:
    array = np.asarray(values)
    if not np.isfinite(array).all():
        raise ValueError("Cannot save NaN or infinite values.")
    target = np.dtype(dtype)
    if np.issubdtype(target, np.integer):
        limits = np.iinfo(target)
        return np.rint(np.clip(array, limits.min, limits.max)).astype(target)
    if np.issubdtype(target, np.floating):
        return array.astype(np.float32)
    raise TypeError(f"Unsupported output dtype: {target}")


def write_tiff(path: str | Path, values: np.ndarray, dtype: np.dtype | None = None) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    array = np.asarray(values, dtype=np.float32) if dtype is None else encode_like(values, dtype)
    tifffile.imwrite(destination, array, photometric="minisblack")
    return destination

