"""cnn_encoder.py
CNN Encoder：把输入图像 [B,C,H,W] 编码成沿宽度方向的序列特征 [B,T,D]。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _conv_bn_relu(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


class CNNEncoder(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 256):
        super().__init__()

        # Stage 1: /2 in (H,W)
        self.s1 = nn.Sequential(
            _conv_bn_relu(in_channels, 64),
            _conv_bn_relu(64, 64),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Stage 2: /2 in (H,W)  -> total width /4
        self.s2 = nn.Sequential(
            _conv_bn_relu(64, 128),
            _conv_bn_relu(128, 128),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Stage 3: /2 in H only (keep W)
        self.s3 = nn.Sequential(
            _conv_bn_relu(128, 256),
            _conv_bn_relu(256, 256),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
        )

        # Stage 4: project to out_channels
        self.s4 = nn.Sequential(
            _conv_bn_relu(256, out_channels),
            _conv_bn_relu(out_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args:
        x: [B,C,H,W]

        Returns:
        seq: [B,T,D] where T is along width.
        """
        feat = self.s1(x)
        feat = self.s2(feat)
        feat = self.s3(feat)
        feat = self.s4(feat)  # [B,D,H',W']

        # 压缩高度到 1，保持宽度 W'
        feat = F.adaptive_avg_pool2d(feat, (1, feat.size(3)))  # [B,D,1,W']
        feat = feat.squeeze(2).permute(0, 2, 1).contiguous()  # [B,W',D]
        return feat


