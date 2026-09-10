from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .signals import apply_channel, generate_clean_signal


@dataclass(frozen=True)
class DatasetConfig:
    classes: tuple[str, ...]
    frame_length: int
    samples_per_symbol: int
    snr_min_db: float
    snr_max_db: float
    max_cfo: float
    multipath_probability: float


class SyntheticAMCDataset(Dataset):
    """Deterministic, balanced synthetic automatic-modulation dataset."""

    def __init__(
        self,
        length: int,
        classes: Sequence[str],
        frame_length: int = 128,
        samples_per_symbol: int = 4,
        snr_min_db: float = -10.0,
        snr_max_db: float = 20.0,
        max_cfo: float = 0.015,
        multipath_probability: float = 0.7,
        seed: int = 0,
    ) -> None:
        if length <= 0:
            raise ValueError("length must be positive")
        if frame_length <= 0:
            raise ValueError("frame_length must be positive")
        self.length = int(length)
        self.cfg = DatasetConfig(
            classes=tuple(classes),
            frame_length=int(frame_length),
            samples_per_symbol=int(samples_per_symbol),
            snr_min_db=float(snr_min_db),
            snr_max_db=float(snr_max_db),
            max_cfo=float(max_cfo),
            multipath_probability=float(multipath_probability),
        )
        if not self.cfg.classes:
            raise ValueError("classes must not be empty")
        self.seed = int(seed)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int):
        if index < 0 or index >= self.length:
            raise IndexError(index)
        label = index % len(self.cfg.classes)
        rng = np.random.default_rng(self.seed + index * 1_000_003)
        modulation = self.cfg.classes[label]
        snr_db = rng.uniform(self.cfg.snr_min_db, self.cfg.snr_max_db)

        clean = generate_clean_signal(
            modulation=modulation,
            num_samples=self.cfg.frame_length,
            samples_per_symbol=self.cfg.samples_per_symbol,
            rng=rng,
        )
        impaired = apply_channel(
            clean,
            snr_db=snr_db,
            max_cfo=self.cfg.max_cfo,
            multipath_probability=self.cfg.multipath_probability,
            rng=rng,
        )

        iq = np.stack((impaired.real, impaired.imag), axis=0).astype(np.float32)
        return torch.from_numpy(iq), int(label), torch.tensor(snr_db, dtype=torch.float32)


def dataset_from_config(cfg: dict, split: str, seed_offset: int = 0) -> SyntheticAMCDataset:
    data_cfg = cfg["data"]
    length_key = f"{split}_samples"
    if length_key not in data_cfg:
        raise KeyError(f"Missing config key data.{length_key}")
    return SyntheticAMCDataset(
        length=int(data_cfg[length_key]),
        classes=data_cfg["classes"],
        frame_length=int(data_cfg["frame_length"]),
        samples_per_symbol=int(data_cfg["samples_per_symbol"]),
        snr_min_db=float(data_cfg["snr_min_db"]),
        snr_max_db=float(data_cfg["snr_max_db"]),
        max_cfo=float(data_cfg["max_cfo"]),
        multipath_probability=float(data_cfg["multipath_probability"]),
        seed=int(cfg["seed"]) + seed_offset,
    )
