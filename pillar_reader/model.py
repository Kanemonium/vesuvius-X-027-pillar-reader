"""Pillar reader: a 121k-parameter ink detector for flattened papyrus surface volumes.

Input  : [batch, 1, Z, H, W]  surface volume patch (Z layers around the sheet surface)
Output : [batch, 1, H, W]     raw ink logits

Four stages (see README.md for the equations and the diagram):

1. A fixed 2D wavelet scattering transform, applied layer by layer and averaged over absolute
   orientation, so the features do not depend on the direction of fibres and strokes.
   Computed with Kymatio (https://github.com/kymatio/kymatio, BSD-3-Clause). No learned weights.
2. A column encoder: log(1 + s), instance normalisation, a 1x1x1 mix of the coefficients and a
   1D convolution along depth.
3. One vector per pillar: each channel is summarised over depth by a soft maximum. This is the
   pillar idea of PointPillars (Lang et al., 2019, https://arxiv.org/abs/1812.05784), on a dense
   grid: one pillar per 8x8 px cell.
4. A small 2D U-Net over the pillar grid, then bilinear upsampling back to the input size.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def scattering_channels(J: int, L: int) -> int:
    """Coefficients kept per cell once averaged over absolute orientation.

    Order 0: 1. Order 1: one per scale. Order 2: for each pair of scales j1 < j2, one per
    orientation difference theta2 - theta1 (L values).
    """
    return 1 + J + L * J * (J - 1) // 2


class InvariantScattering(nn.Module):
    """Per-layer 2D scattering, averaged over rotations. No learned parameter."""

    def __init__(self, size: tuple[int, int], J: int = 3, L: int = 8) -> None:
        super().__init__()
        self.J, self.L = J, L
        self.channels = scattering_channels(J, L)
        # One transform per input size, built on demand (inference may use other patch sizes).
        # Kept out of the state dict: the filters are computed, not learned.
        self._transforms: dict[tuple[int, int, str], nn.Module] = {}
        self._transform((int(size[0]), int(size[1])), torch.device("cpu"))

    def _transform(self, size: tuple[int, int], device: torch.device) -> nn.Module:
        key = (size[0], size[1], str(device))
        if key not in self._transforms:
            from kymatio.torch import Scattering2D

            self._transforms[key] = Scattering2D(J=self.J, shape=size, L=self.L).to(device)
        return self._transforms[key]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, Z, H, W = x.shape
        J, L = self.J, self.L
        transform = self._transform((H, W), x.device)
        # Kymatio works through FFTs: always float32, even under mixed precision.
        with torch.autocast(device_type=x.device.type, enabled=False):
            # Kymatio order: order 0; order 1 by (j, theta); order 2 by (j1, theta1, j2, theta2).
            s = transform(x.float().reshape(batch * Z, H, W).contiguous())
        h, w = s.shape[-2:]
        order0 = s[:, :1]
        order1 = s[:, 1 : 1 + J * L].reshape(-1, J, L, h, w).mean(dim=2)
        outputs = [order0, order1]
        start = 1 + J * L
        for j1 in range(J):
            n = J - j1 - 1  # scales j2 > j1
            if n == 0:
                continue
            block = s[:, start : start + L * n * L].reshape(-1, L, n, L, h, w)  # theta1, j2, theta2
            start += L * n * L
            # Orientation difference: rotate theta2 by -theta1 before averaging over theta1.
            relative = torch.stack(
                [torch.roll(block[:, t1], shifts=-t1, dims=2) for t1 in range(L)], dim=1
            )
            outputs.append(relative.mean(dim=1).reshape(-1, n * L, h, w))
        out = torch.cat(outputs, dim=1)
        return out.reshape(batch, Z, self.channels, h, w).transpose(1, 2)  # [batch, C, Z, h, w]


def _block(inputs: int, outputs: int) -> nn.Sequential:
    """Two 3x3 convolutions on the pillar grid, each normalised then activated."""
    return nn.Sequential(
        nn.Conv2d(inputs, outputs, kernel_size=3, padding=1),
        nn.InstanceNorm2d(outputs, affine=True),
        nn.LeakyReLU(0.01),
        nn.Conv2d(outputs, outputs, kernel_size=3, padding=1),
        nn.InstanceNorm2d(outputs, affine=True),
        nn.LeakyReLU(0.01),
    )


class PillarUNet(nn.Module):
    """Three-level 2D U-Net over the pillar grid: base, 2 x base, 4 x base channels."""

    def __init__(self, inputs: int = 16, base: int = 16) -> None:
        super().__init__()
        self.down1 = _block(inputs, base)
        self.down2 = _block(base, 2 * base)
        self.bottom = _block(2 * base, 4 * base)
        self.up2 = nn.ConvTranspose2d(4 * base, 2 * base, kernel_size=2, stride=2)
        self.dec2 = _block(4 * base, 2 * base)
        self.up1 = nn.ConvTranspose2d(2 * base, base, kernel_size=2, stride=2)
        self.dec1 = _block(2 * base, base)
        self.head = nn.Conv2d(base, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        # Two halvings: the grid must be a multiple of 4; otherwise the border is replicated.
        dh, dw = (-h) % 4, (-w) % 4
        if dh or dw:
            x = F.pad(x, (0, dw, 0, dh), mode="replicate")
        a = self.down1(x)
        b = self.down2(F.max_pool2d(a, 2))
        c = self.bottom(F.max_pool2d(b, 2))
        b = self.dec2(torch.cat([self.up2(c), b], dim=1))
        a = self.dec1(torch.cat([self.up1(b), a], dim=1))
        return self.head(a)[..., :h, :w]


class PillarReader(nn.Module):
    """Fixed scattering, column encoder, one vector per pillar, 2D U-Net over the pillars."""

    def __init__(
        self,
        size: tuple[int, int] = (256, 256),
        J: int = 3,
        L: int = 8,
        width: int = 16,
        depth_kernel: int = 5,
        tau: float = 1.0,
        base: int = 16,
    ) -> None:
        super().__init__()
        self.scattering = InvariantScattering(size, J=J, L=L)
        c = self.scattering.channels
        self.norm = nn.InstanceNorm3d(c, affine=True)
        self.mix = nn.Conv3d(c, width, kernel_size=1)
        self.depth = nn.Conv3d(
            width, width, kernel_size=(depth_kernel, 1, 1), padding=(depth_kernel // 2, 0, 0)
        )
        self.tau = tau
        self.unet = PillarUNet(inputs=width, base=base)

    def pillars(self, x: torch.Tensor) -> torch.Tensor:
        """The pseudo-image: [batch, width, h, w], one vector per 2^J x 2^J px column."""
        s = torch.log1p(self.scattering(x).clamp_min(0))
        u = F.leaky_relu(self.mix(self.norm(s)), 0.01)
        u = F.leaky_relu(self.depth(u), 0.01)  # [batch, width, Z, h, w]
        # Soft maximum over depth, channel by channel: tau * log sum exp(u / tau).
        return self.tau * torch.logsumexp(u / self.tau, dim=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        logits = self.unet(self.pillars(x))  # [batch, 1, h, w]
        return F.interpolate(logits, size=size, mode="bilinear", align_corners=False)
