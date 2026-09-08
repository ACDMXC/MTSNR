"""Run the proposed CNN-ADMM method and create three diagnostic PDFs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from calibration_restoration.admm import ADMMConfig
from calibration_restoration.io import read_tiff
from calibration_restoration.pipeline import restore_array


TIFF_SUFFIXES = {".tif", ".tiff"}


def find_single_tiff(folder: str | Path) -> Path:
    """Return the only TIFF in *folder* and reject ambiguous input."""
    input_dir = Path(folder)
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input folder does not exist: {input_dir}")
    images = sorted(
        (path for path in input_dir.iterdir() if path.is_file() and path.suffix.lower() in TIFF_SUFFIXES),
        key=lambda path: path.name.casefold(),
    )
    if len(images) != 1:
        raise ValueError(f"NoisyImage must contain exactly one TIFF; found {len(images)} in {input_dir}.")
    return images[0]


def compute_residual(noisy: np.ndarray, proposed: np.ndarray) -> np.ndarray:
    """Return the total component removed by the proposed method: D - I."""
    noisy_array = np.asarray(noisy, dtype=np.float32)
    proposed_array = np.asarray(proposed, dtype=np.float32)
    if noisy_array.shape != proposed_array.shape:
        raise ValueError(f"Noisy and proposed shapes differ: {noisy_array.shape} vs {proposed_array.shape}.")
    if noisy_array.ndim != 2 or not np.isfinite(noisy_array).all() or not np.isfinite(proposed_array).all():
        raise ValueError("Noisy and proposed images must be finite two-dimensional arrays.")
    return np.asarray(noisy_array - proposed_array, dtype=np.float32)


def compute_mean_profiles(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return one mean value per image row and per image column."""
    array = np.asarray(image, dtype=np.float32)
    if array.ndim != 2 or not np.isfinite(array).all():
        raise ValueError("The profile image must be one finite two-dimensional array.")
    row_mean = np.mean(array, axis=1, dtype=np.float64).astype(np.float32)
    column_mean = np.mean(array, axis=0, dtype=np.float64).astype(np.float32)
    return row_mean, column_mean


def _gray_limits(image: np.ndarray) -> tuple[float, float]:
    low, high = (float(value) for value in np.percentile(image, (1.0, 99.0)))
    if high <= low:
        high = low + 1.0
    return low, high


def save_comparison_pdf(
    noisy: np.ndarray,
    proposed: np.ndarray,
    residual: np.ndarray,
    output: str | Path,
    *,
    show: bool = True,
) -> Path:
    """Save the Noisy/Proposed/Residual comparison as one horizontal PDF."""
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    low, high = _gray_limits(noisy)
    residual_limit = max(
        float(np.percentile(np.abs(residual), 99.5)),
        float(np.finfo(np.float32).eps),
    )

    figure, axes = plt.subplots(1, 3, figsize=(10.5, 3.45), constrained_layout=True)
    panels = (
        (noisy, "Noisy", "gray", low, high),
        (proposed, "Proposed", "gray", low, high),
        (residual, "Residual", "RdBu_r", -residual_limit, residual_limit),
    )
    for axis, (image, title, cmap, vmin, vmax) in zip(axes, panels):
        axis.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
        axis.set_title(title, fontsize=12)
        axis.set_axis_off()
    figure.savefig(destination, format="pdf", dpi=600, bbox_inches="tight", pad_inches=0.02)
    if show:
        plt.show(block=False)
    else:
        plt.close(figure)
    return destination


def save_convergence_pdf(
    history: dict[str, list[float]],
    output: str | Path,
    *,
    show: bool = True,
) -> Path:
    """Save normalized primal and dual ADMM residuals on linear axes."""
    primal = np.asarray(history.get("normalized_primal_residual", []), dtype=np.float64)
    dual = np.asarray(history.get("normalized_dual_residual", []), dtype=np.float64)
    if primal.ndim != 1 or dual.ndim != 1 or primal.size == 0 or primal.size != dual.size:
        raise ValueError("Convergence history must contain equally sized normalized primal and dual residuals.")
    if not np.isfinite(primal).all() or not np.isfinite(dual).all():
        raise ValueError("Convergence history contains NaN or infinite values.")

    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    iterations = np.arange(1, primal.size + 1)
    figure, axes = plt.subplots(1, 2, figsize=(8.0, 3.25), constrained_layout=True)
    panels = (
    (iterations, primal, "Normalized primal residual"),
    (iterations[1:], dual[1:], "Normalized dual residual"),
    )

    for label, axis, (x_values, values, title) in zip(("(a)", "(b)"), axes, panels):
        axis.plot(x_values,values,color="#2171b5",linewidth=1.8,)
        axis.set_xlabel("Iteration")
        axis.set_ylabel("Metric value")
        axis.set_title(title, fontsize=11)
        axis.grid(True, linestyle="--", linewidth=0.6, alpha=0.45)
        axis.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        axis.text(0.5, -0.24, label, transform=axis.transAxes, ha="center", va="top", fontweight="bold")
    figure.savefig(destination, format="pdf", dpi=600, bbox_inches="tight", pad_inches=0.03)
    if show:
        plt.show(block=False)
    else:
        plt.close(figure)
    return destination


def save_mean_profiles_pdf(
    noisy: np.ndarray,
    proposed: np.ndarray,
    output: str | Path,
    *,
    show: bool = True,
) -> Path:
    """Save Noisy and Proposed row/column mean-profile comparisons."""
    if np.asarray(noisy).shape != np.asarray(proposed).shape:
        raise ValueError("Noisy and proposed images must have the same shape for profile comparison.")
    noisy_row, noisy_column = compute_mean_profiles(noisy)
    proposed_row, proposed_column = compute_mean_profiles(proposed)
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(1, 2, figsize=(8.0, 3.25), constrained_layout=True)
    panels = (
        (noisy_row, proposed_row, "Row mean profile", "Row index"),
        (noisy_column, proposed_column, "Column mean profile", "Column index"),
    )
    for label, axis, (noisy_values, proposed_values, title, x_label) in zip(("(a)", "(b)"), axes, panels):
        indices = np.arange(noisy_values.size)
        axis.plot(indices, noisy_values, color="#777777", linewidth=1.2, label="Noisy")
        axis.plot(indices, proposed_values, color="#2171b5", linewidth=1.6, label="Proposed")
        axis.set_xlabel(x_label)
        axis.set_ylabel("Mean intensity")
        axis.set_title(title, fontsize=11)
        axis.grid(True, linestyle="--", linewidth=0.6, alpha=0.45)
        axis.legend(frameon=False, fontsize=9)
        axis.text(0.5, -0.24, label, transform=axis.transAxes, ha="center", va="top", fontweight="bold")
    figure.savefig(destination, format="pdf", dpi=600, bbox_inches="tight", pad_inches=0.03)
    if show:
        plt.show(block=False)
    else:
        plt.close(figure)
    return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "NoisyImage")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints" / "M1_Pad_Mean_best.pth")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "Results")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:0")
    parser.add_argument("--no-show", action="store_true", help="Save the PDF without opening a figure window.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = find_single_tiff(args.input_dir)
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {args.checkpoint}")

    noisy, _ = read_tiff(input_path)
    config = ADMMConfig(
        lambda_1=0.11,
        lambda_2=0.90,
        lambda_3=0.035,
        rho=0.10,
        max_iters=50,
        eps_relative=1e-6,
    )
    result = restore_array(
        noisy,
        checkpoint=args.checkpoint,
        device=args.device,
        use_cnn=True,
        admm_config=config,
    )
    proposed = np.asarray(result.restored, dtype=np.float32)
    residual = compute_residual(noisy, proposed)
    show_figures = not args.no_show
    comparison_path = args.output_dir / "noisy_proposed_residual.pdf"
    convergence_path = args.output_dir / "admm_convergence.pdf"
    profiles_path = args.output_dir / "row_column_mean_profiles.pdf"
    save_comparison_pdf(noisy, proposed, residual, comparison_path, show=show_figures)
    save_convergence_pdf(result.history, convergence_path, show=show_figures)
    save_mean_profiles_pdf(noisy, proposed, profiles_path, show=show_figures)
    if show_figures:
        plt.show()

    iterations = len(result.history.get("relative_residual", []))
    print(f"Input      : {input_path}")
    print(f"Device     : {result.device}")
    print(f"Iterations : {iterations}")
    print(f"Residual   : Noisy - Proposed")
    print(f"Saved PDF  : {comparison_path}")
    print(f"Saved PDF  : {convergence_path}")
    print(f"Saved PDF  : {profiles_path}")


if __name__ == "__main__":
    main()
