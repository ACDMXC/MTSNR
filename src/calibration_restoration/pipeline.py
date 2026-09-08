"""Single-pass CNN initialization followed by explicit ADMM refinement."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .admm import ADMMConfig, admm_restore
from .normalization import denormalize, normalize
from .wdcnn import MultiWaveletCNN


@dataclass(frozen=True)
class RestorationResult:
    restored: np.ndarray
    cnn_corrected: np.ndarray
    cnn_stripe: np.ndarray
    sparse_noise: np.ndarray
    residual_stripe: np.ndarray
    history: dict[str, list[float]]
    device: str


def resolve_device(name: str = "auto") -> torch.device:
    if name == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _state_dict(payload: Any) -> dict[str, torch.Tensor]:
    state = payload.get("model_state_dict") if isinstance(payload, dict) and "model_state_dict" in payload else payload
    if not isinstance(state, dict) or not state:
        raise ValueError("Checkpoint does not contain a valid state dictionary.")
    return state


def load_checkpoint(
    checkpoint: str | Path,
    device: str | torch.device = "auto",
) -> tuple[MultiWaveletCNN, torch.device]:
    source = Path(checkpoint)
    if not source.is_file():
        raise FileNotFoundError(source)
    selected_device = resolve_device(device) if isinstance(device, str) else device
    model = MultiWaveletCNN(num_blocks=1).to(selected_device)
    try:
        payload = torch.load(source, map_location=selected_device, weights_only=False)
    except TypeError:
        payload = torch.load(source, map_location=selected_device)
    model.load_state_dict(_state_dict(payload))
    model.eval()
    return model, selected_device


def estimate_cnn_stripe(
    image: np.ndarray,
    model: MultiWaveletCNN,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    normalized, minimum, maximum = normalize(image)
    tensor = torch.as_tensor(normalized, dtype=torch.float32, device=device)[None, None]
    with torch.inference_mode():
        predicted = model(tensor).detach().cpu().numpy().squeeze()
    stripe = predicted * (maximum - minimum)
    corrected = denormalize(normalized - predicted, minimum, maximum)
    return np.asarray(corrected, dtype=np.float32), np.asarray(stripe, dtype=np.float32)


def restore_array(
    image: np.ndarray,
    *,
    checkpoint: str | Path | None = None,
    model: MultiWaveletCNN | None = None,
    device: str | torch.device = "auto",
    use_cnn: bool = True,
    admm_config: ADMMConfig | None = None,
) -> RestorationResult:
    """Apply the manuscript pipeline to one preconditioned 2-D image."""
    input_image = np.asarray(image, dtype=np.float32)
    if input_image.ndim != 2 or not np.isfinite(input_image).all():
        raise ValueError("The input must be one finite 2-D image.")
    if use_cnn:
        if model is None:
            if checkpoint is None:
                raise ValueError("A checkpoint is required when use_cnn=True.")
            model, selected_device = load_checkpoint(checkpoint, device)
        else:
            selected_device = resolve_device(device) if isinstance(device, str) else device
            model = model.to(selected_device).eval()
        corrected, cnn_stripe = estimate_cnn_stripe(input_image, model, selected_device)
        device_text = str(selected_device)
    else:
        corrected = input_image.copy()
        cnn_stripe = np.zeros_like(input_image)
        device_text = "bypassed"

    refined = admm_restore(corrected, admm_config)
    return RestorationResult(
        restored=refined.restored,
        cnn_corrected=corrected,
        cnn_stripe=cnn_stripe,
        sparse_noise=refined.sparse_noise,
        residual_stripe=refined.residual_stripe,
        history=refined.history,
        device=device_text,
    )

