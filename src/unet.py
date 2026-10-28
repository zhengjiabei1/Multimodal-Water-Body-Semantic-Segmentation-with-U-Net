"""U-Net segmentation network for binary water body extraction."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """Two 3x3 convolutions, each followed by batch normalization and ReLU."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet(nn.Module):
    """Four-level U-Net with a sigmoid output for binary segmentation.

    Args:
        in_channels: Number of input channels (3-6 depending on the scheme).
        features: Channel widths of the four encoder levels.
    """

    def __init__(
        self,
        in_channels: int = 3,
        features: tuple[int, ...] = (64, 128, 256, 512),
    ) -> None:
        super().__init__()
        self.encoders = nn.ModuleList()
        channels = in_channels
        for feature in features:
            self.encoders.append(DoubleConv(channels, feature))
            channels = feature

        self.pool = nn.MaxPool2d(2, 2)
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for feature in reversed(features):
            self.upconvs.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.decoders.append(DoubleConv(feature * 2, feature))

        self.head = nn.Conv2d(features[0], 1, kernel_size=1)
        self.activation = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips: list[torch.Tensor] = []
        for encoder in self.encoders:
            x = encoder(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        for upconv, decoder, skip in zip(self.upconvs, self.decoders, reversed(skips)):
            x = upconv(x)
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(
                    x, size=skip.shape[-2:], mode="bilinear", align_corners=False
                )
            x = torch.cat([skip, x], dim=1)
            x = decoder(x)

        return self.activation(self.head(x))
