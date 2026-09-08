"""Final M1 wavelet CNN used for frequency-stripe estimation."""

from __future__ import annotations

import numpy as np
import pywt
import torch
from torch import nn
from torch.nn import functional as F


class WaveletConvBlock(nn.Module):
    """DWT, approximation/detail branches, fusion, and directional mean."""

    def __init__(self, wavelet: str = "haar") -> None:
        super().__init__()
        self.wavelet = wavelet
        self.conv_CA = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.conv_CH_CV_CD = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.last_conv = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
        )

    def dwt(self, tensor: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Apply the original fixed PyWavelets transform to each image."""
        batch, channels, height, width = tensor.shape
        coefficients: list[list[np.ndarray]] = [[], [], [], []]
        for batch_index in range(batch):
            for channel_index in range(channels):
                plane = tensor[batch_index, channel_index].detach().cpu().numpy()
                ll, (lh, hl, hh) = pywt.dwt2(plane, self.wavelet)
                for target, value in zip(coefficients, (ll, lh, hl, hh)):
                    target.append(np.asarray(value, dtype=np.float32))
        output_shape = (batch, channels, coefficients[0][0].shape[0], coefficients[0][0].shape[1])
        outputs = []
        for values in coefficients:
            stacked = np.stack(values).reshape(output_shape)
            outputs.append(torch.as_tensor(stacked, device=tensor.device, dtype=tensor.dtype))
        return tuple(outputs)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.ndim != 4 or tensor.shape[1] != 1:
            raise ValueError("The M1 network expects an N x 1 x H x W tensor.")
        height, width = tensor.shape[-2:]
        if min(height, width) <= 15:
            raise ValueError("Both spatial dimensions must exceed 15 pixels.")
        padded = F.pad(tensor, (15, 15, 15, 15), mode="reflect")
        ll, lh, hl, hh = self.dwt(padded)
        fused = torch.cat(
            (
                self.conv_CA(ll),
                self.conv_CH_CV_CD(lh),
                self.conv_CH_CV_CD(hl),
                self.conv_CH_CV_CD(hh),
            ),
            dim=1,
        )
        result = self.last_conv(fused)
        directional_mean = result.mean(dim=2, keepdim=True).expand_as(result)
        return directional_mean[:, :, 15 : 15 + height, 15 : 15 + width]


class MultiWaveletCNN(nn.Module):
    """Stack one or more wavelet blocks; the released checkpoint uses one."""

    def __init__(self, num_blocks: int = 1, wavelet: str = "haar") -> None:
        super().__init__()
        if num_blocks < 1:
            raise ValueError("num_blocks must be positive.")
        self.blocks = nn.ModuleList(
            [WaveletConvBlock(wavelet=wavelet) for _ in range(num_blocks)]
        )

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        output = tensor
        for block in self.blocks:
            output = block(output)
        return output

