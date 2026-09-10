from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class TinyAMC(nn.Module):
    """Small 1D CNN chosen to fit a Zynq-7010 learning project."""

    def __init__(
        self,
        num_classes: int = 6,
        conv1_channels: int = 8,
        conv2_channels: int = 16,
        kernel1: int = 5,
        kernel2: int = 3,
    ) -> None:
        super().__init__()
        if kernel1 % 2 == 0 or kernel2 % 2 == 0:
            raise ValueError("Only odd kernels are supported by the HLS reference")
        self.conv1 = nn.Conv1d(
            2,
            conv1_channels,
            kernel_size=kernel1,
            padding=kernel1 // 2,
            bias=True,
        )
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(
            conv1_channels,
            conv2_channels,
            kernel_size=kernel2,
            padding=kernel2 // 2,
            bias=True,
        )
        self.relu2 = nn.ReLU()
        self.fc = nn.Linear(conv2_channels, num_classes)

    def forward_features(self, x: torch.Tensor):
        x = self.conv1(x)
        x = self.relu1(x)
        pooled = self.pool1(x)
        x = self.conv2(pooled)
        activated = self.relu2(x)
        gap = F.adaptive_avg_pool1d(activated, 1).squeeze(-1)
        return pooled, activated, gap

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, gap = self.forward_features(x)
        return self.fc(gap)


def model_from_config(cfg: dict) -> TinyAMC:
    model_cfg = cfg["model"]
    return TinyAMC(
        num_classes=len(cfg["data"]["classes"]),
        conv1_channels=int(model_cfg["conv1_channels"]),
        conv2_channels=int(model_cfg["conv2_channels"]),
        kernel1=int(model_cfg["kernel1"]),
        kernel2=int(model_cfg["kernel2"]),
    )
