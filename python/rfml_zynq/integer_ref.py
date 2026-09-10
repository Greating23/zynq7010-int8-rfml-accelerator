from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.nn import functional as F


@dataclass
class QuantizedParameters:
    w1: np.ndarray
    b1: np.ndarray
    w2: np.ndarray
    b2: np.ndarray
    w3: np.ndarray
    b3: np.ndarray
    m1_int: int
    m2_int: int
    requant_shift: int


def _requant_relu(x: torch.Tensor, multiplier: int, shift: int) -> torch.Tensor:
    values = x.to(torch.int64) * int(multiplier)
    rounding = 1 << (shift - 1)
    values = torch.where(values >= 0, values + rounding, values - rounding)
    values = torch.bitwise_right_shift(values, shift)
    return values.clamp(0, 127).to(torch.int8)


def integer_forward(input_iq: np.ndarray, params: QuantizedParameters) -> np.ndarray:
    """Bit-oriented reference matching hls/src/amc_accel.cpp."""
    if input_iq.shape != (2, 128):
        raise ValueError(f"Expected input shape (2, 128), got {input_iq.shape}")

    x = torch.from_numpy(input_iq.astype(np.int8)).unsqueeze(0).to(torch.float32)
    w1 = torch.from_numpy(params.w1.astype(np.int8)).to(torch.float32)
    b1 = torch.from_numpy(params.b1.astype(np.int32)).to(torch.float32)
    acc1 = F.conv1d(x, w1, b1, padding=params.w1.shape[-1] // 2)
    acc1 = torch.round(acc1).to(torch.int64)
    q1 = _requant_relu(acc1, params.m1_int, params.requant_shift)
    q1 = F.max_pool1d(q1.to(torch.float32), kernel_size=2, stride=2)
    q1 = torch.round(q1).to(torch.int8)

    w2 = torch.from_numpy(params.w2.astype(np.int8)).to(torch.float32)
    b2 = torch.from_numpy(params.b2.astype(np.int32)).to(torch.float32)
    acc2 = F.conv1d(
        q1.to(torch.float32), w2, b2, padding=params.w2.shape[-1] // 2
    )
    acc2 = torch.round(acc2).to(torch.int64)
    q2 = _requant_relu(acc2, params.m2_int, params.requant_shift)

    length = q2.shape[-1]
    gap = (q2.to(torch.int64).sum(dim=-1) + length // 2) // length
    gap_np = gap.squeeze(0).cpu().numpy().astype(np.int64)

    logits = params.w3.astype(np.int64) @ gap_np + params.b3.astype(np.int64)
    return logits.astype(np.int32)
