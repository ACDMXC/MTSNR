"""Explicit sparse, low-rank, and total-variation ADMM refinement."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from skimage.restoration import denoise_tv_chambolle


@dataclass(frozen=True)
class ADMMConfig:
    """Parameters adopted in the manuscript."""

    lambda_1: float = 0.11
    lambda_2: float = 0.9
    lambda_3: float = 0.035
    rho: float = 0.1
    max_iters: int = 50
    eps_relative: float = 1e-6

    def __post_init__(self) -> None:
        if min(self.lambda_1, self.lambda_2, self.lambda_3) < 0:
            raise ValueError("Regularization weights must be nonnegative.")
        if self.rho <= 0 or self.max_iters < 1 or self.eps_relative <= 0:
            raise ValueError("rho, max_iters, and eps_relative must be positive.")


@dataclass(frozen=True)
class ADMMResult:
    restored: np.ndarray
    sparse_noise: np.ndarray
    residual_stripe: np.ndarray
    history: dict[str, list[float]]


def soft(values: np.ndarray, threshold: float) -> np.ndarray:
    """Elementwise soft thresholding."""
    return np.sign(values) * np.maximum(np.abs(values) - threshold, 0.0)


def singular_value_threshold(values: np.ndarray, threshold: float) -> np.ndarray:
    """Nuclear-norm proximal mapping."""
    left, singular_values, right = np.linalg.svd(values, full_matrices=False)
    shrunk = np.maximum(singular_values - threshold, 0.0)
    return (left * shrunk) @ right


def _validate_observation(observation: np.ndarray) -> np.ndarray:
    array = np.asarray(observation, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"Expected a 2-D observation, received shape {array.shape}.")
    if not np.isfinite(array).all():
        raise ValueError("The observation contains NaN or infinite values.")
    return array


def admm_restore(
    observation: np.ndarray,
    config: ADMMConfig | None = None,
) -> ADMMResult:
    """Refine a CNN-corrected observation and return all model components."""
    cfg = config or ADMMConfig()
    target = _validate_observation(observation)
    restored = target.copy()
    sparse = np.zeros_like(target)
    stripe = np.zeros_like(target)
    auxiliary = target.copy()
    dual = np.zeros_like(target)
    history = {
        "primal_residual": [],
        "dual_residual": [],
        "relative_residual": [],
        "normalized_primal_residual": [],
        "normalized_dual_residual": [],
    }
    eps = float(np.finfo(np.float64).eps)

    for _ in range(cfg.max_iters):
        previous_auxiliary = auxiliary.copy()
        auxiliary = (
            target + cfg.rho * (restored + sparse + stripe - dual)
        ) / (1.0 + cfg.rho)
        sparse = soft(auxiliary - restored - stripe + dual, cfg.lambda_1 / cfg.rho)
        stripe = singular_value_threshold(
            auxiliary - restored - sparse + dual,
            cfg.lambda_2 / cfg.rho,
        )
        restored = denoise_tv_chambolle(
            auxiliary - sparse - stripe + dual,
            weight=cfg.lambda_3 / cfg.rho,
            channel_axis=None,
        ).astype(np.float32, copy=False)

        primal = auxiliary - restored - sparse - stripe
        dual_step = cfg.rho * (auxiliary - previous_auxiliary)
        dual += primal
        reconstruction_norm = max(float(np.linalg.norm(restored + sparse + stripe)), eps)
        auxiliary_norm = max(float(np.linalg.norm(auxiliary)), eps)
        primal_norm = float(np.linalg.norm(primal))
        dual_norm = float(np.linalg.norm(dual_step))
        relative = primal_norm / reconstruction_norm

        history["primal_residual"].append(primal_norm)
        history["dual_residual"].append(dual_norm)
        history["relative_residual"].append(relative)
        history["normalized_primal_residual"].append(primal_norm / auxiliary_norm)
        history["normalized_dual_residual"].append(dual_norm / auxiliary_norm)
        if relative <= cfg.eps_relative:
            break

    return ADMMResult(
        restored=np.asarray(restored, dtype=np.float32),
        sparse_noise=np.asarray(sparse, dtype=np.float32),
        residual_stripe=np.asarray(stripe, dtype=np.float32),
        history=history,
    )

