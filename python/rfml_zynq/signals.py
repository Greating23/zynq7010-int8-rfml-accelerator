from __future__ import annotations

import math
from functools import lru_cache

import numpy as np


def _qam_constellation(order: int) -> np.ndarray:
    side = int(math.sqrt(order))
    if side * side != order:
        raise ValueError(f"QAM order must be square, got {order}")
    levels = np.arange(-(side - 1), side, 2, dtype=np.float32)
    constellation = np.array(
        [complex(i, q) for i in levels for q in levels], dtype=np.complex64
    )
    constellation /= np.sqrt(np.mean(np.abs(constellation) ** 2))
    return constellation


def _psk_constellation(order: int) -> np.ndarray:
    angles = 2.0 * np.pi * np.arange(order) / order
    return np.exp(1j * angles).astype(np.complex64)


@lru_cache(maxsize=16)
def root_raised_cosine(
    samples_per_symbol: int,
    span_symbols: int = 8,
    beta: float = 0.35,
) -> np.ndarray:
    """Return unit-energy RRC taps."""
    sps = samples_per_symbol
    n = span_symbols * sps
    t = np.arange(-n / 2, n / 2 + 1, dtype=np.float64) / sps
    taps = np.zeros_like(t)

    for idx, value in enumerate(t):
        if abs(value) < 1e-12:
            taps[idx] = 1.0 + beta * (4.0 / np.pi - 1.0)
        elif beta > 0 and abs(abs(value) - 1.0 / (4.0 * beta)) < 1e-10:
            taps[idx] = (
                beta
                / np.sqrt(2.0)
                * (
                    (1.0 + 2.0 / np.pi) * np.sin(np.pi / (4.0 * beta))
                    + (1.0 - 2.0 / np.pi) * np.cos(np.pi / (4.0 * beta))
                )
            )
        else:
            numerator = (
                np.sin(np.pi * value * (1.0 - beta))
                + 4.0 * beta * value * np.cos(np.pi * value * (1.0 + beta))
            )
            denominator = np.pi * value * (1.0 - (4.0 * beta * value) ** 2)
            taps[idx] = numerator / denominator

    taps /= np.sqrt(np.sum(taps**2))
    return taps.astype(np.float32)


def _linear_modulation(
    constellation: np.ndarray,
    num_samples: int,
    samples_per_symbol: int,
    rng: np.random.Generator,
) -> np.ndarray:
    taps = root_raised_cosine(samples_per_symbol)
    guard = len(taps)
    num_symbols = int(np.ceil((num_samples + guard) / samples_per_symbol)) + 4
    symbols = rng.choice(constellation, size=num_symbols)
    upsampled = np.zeros(num_symbols * samples_per_symbol, dtype=np.complex64)
    upsampled[::samples_per_symbol] = symbols
    shaped = np.convolve(upsampled, taps, mode="full")
    start = len(taps) // 2 + rng.integers(0, samples_per_symbol)
    return shaped[start : start + num_samples].astype(np.complex64)


def _fsk2(
    num_samples: int,
    samples_per_symbol: int,
    rng: np.random.Generator,
) -> np.ndarray:
    num_symbols = int(np.ceil(num_samples / samples_per_symbol)) + 2
    bits = rng.integers(0, 2, size=num_symbols)
    symbols = np.repeat(2 * bits - 1, samples_per_symbol)[:num_samples]
    deviation = 0.12 / samples_per_symbol
    phase = 2.0 * np.pi * np.cumsum(symbols * deviation)
    return np.exp(1j * phase).astype(np.complex64)


def generate_clean_signal(
    modulation: str,
    num_samples: int,
    samples_per_symbol: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if modulation == "BPSK":
        return _linear_modulation(
            _psk_constellation(2), num_samples, samples_per_symbol, rng
        )
    if modulation == "QPSK":
        return _linear_modulation(
            _psk_constellation(4), num_samples, samples_per_symbol, rng
        )
    if modulation == "8PSK":
        return _linear_modulation(
            _psk_constellation(8), num_samples, samples_per_symbol, rng
        )
    if modulation == "16QAM":
        return _linear_modulation(
            _qam_constellation(16), num_samples, samples_per_symbol, rng
        )
    if modulation == "64QAM":
        return _linear_modulation(
            _qam_constellation(64), num_samples, samples_per_symbol, rng
        )
    if modulation == "2FSK":
        return _fsk2(num_samples, samples_per_symbol, rng)
    raise KeyError(f"Unsupported modulation: {modulation}")


def apply_channel(
    signal: np.ndarray,
    snr_db: float,
    max_cfo: float,
    multipath_probability: float,
    rng: np.random.Generator,
) -> np.ndarray:
    x = np.asarray(signal, dtype=np.complex64).copy()

    phase = rng.uniform(0.0, 2.0 * np.pi)
    gain = rng.uniform(0.7, 1.3)
    cfo = rng.uniform(-max_cfo, max_cfo)
    n = np.arange(len(x), dtype=np.float32)
    x *= gain * np.exp(1j * (phase + 2.0 * np.pi * cfo * n))

    if rng.random() < multipath_probability:
        taps = np.zeros(5, dtype=np.complex64)
        taps[0] = 1.0 + 0j
        taps[2] = rng.uniform(0.05, 0.35) * np.exp(1j * rng.uniform(0, 2 * np.pi))
        taps[4] = rng.uniform(0.02, 0.20) * np.exp(1j * rng.uniform(0, 2 * np.pi))
        x = np.convolve(x, taps, mode="full")[: len(x)]

    power = float(np.mean(np.abs(x) ** 2)) + 1e-12
    noise_power = power / (10.0 ** (snr_db / 10.0))
    noise = np.sqrt(noise_power / 2.0) * (
        rng.standard_normal(len(x)) + 1j * rng.standard_normal(len(x))
    )
    x = x + noise.astype(np.complex64)

    peak = float(np.max(np.abs(x))) + 1e-9
    x = x / peak
    return x.astype(np.complex64)
