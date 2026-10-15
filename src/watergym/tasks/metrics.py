"""Benchmark metrics from per-step signals shaped [steps, envs].

Comfort is the ISO 2631-1:1997 frequency-weighted RMS acceleration a_w (clause 6.1), using
the vertical weighting Wk of Annex A. The weighting is a cascade of four analog filters:

    band limit   Hh (high-pass, f1) and Hl (low-pass, f2)
    a-v transition  Ht = (1 + s/w3) / (1 + s/(Q4 w4) + (s/w4)^2)
    upward step  Hs = (1 + s/(Q5 w5) + (s/w5)^2) / (1 + s/(Q6 w6) + (s/w6)^2) (w5/w6)^2

`wk_filter` applies the analog gain to the signal's spectrum, so the weighting is exact up to
Nyquist. A bilinear-transformed biquad cascade without prewarping reads 14-60 % low from 12.5
to 20 Hz at 50 Hz sampling, which flattered policies that chatter the flap.
"""

import math

import numpy as np
import torch
from torch import Tensor

# ISO 2631-1:1997 Table A.2, weighting Wk (vertical, seated or standing), frequencies in Hz.
WK = {
    "f1": 0.4,
    "f2": 100.0,
    "f3": 12.5,
    "f4": 12.5,
    "q4": 0.63,
    "f5": 2.37,
    "q5": 0.91,
    "f6": 3.35,
    "q6": 0.91,
}

Polynomial = list[float]  # coefficients of s^0, s^1, s^2


def wk_analog_sections() -> list[tuple[Polynomial, Polynomial]]:
    """Wk as (numerator, denominator) pairs in s, ascending powers."""
    w = {k: 2 * math.pi * v for k, v in WK.items() if k.startswith("f")}
    return [
        ([0.0, 0.0, 1 / w["f1"] ** 2], [1.0, math.sqrt(2) / w["f1"], 1 / w["f1"] ** 2]),
        ([1.0], [1.0, math.sqrt(2) / w["f2"], 1 / w["f2"] ** 2]),
        ([1.0, 1 / w["f3"]], [1.0, 1 / (WK["q4"] * w["f4"]), 1 / w["f4"] ** 2]),
        (
            [(w["f5"] / w["f6"]) ** 2, w["f5"] / WK["q5"] / w["f6"] ** 2, 1 / w["f6"] ** 2],
            [1.0, 1 / (WK["q6"] * w["f6"]), 1 / w["f6"] ** 2],
        ),
    ]


def analog_gain(sections: list[tuple[Polynomial, Polynomial]], frequency_hz: float) -> float:
    """|H(j 2 pi f)| of the cascade."""
    s = 2j * math.pi * frequency_hz
    gain = 1.0 + 0j
    for num, den in sections:
        gain *= np.polyval(num[::-1], s) / np.polyval(den[::-1], s)
    return abs(gain)


def wk_filter(acceleration: Tensor, sample_rate: float) -> Tensor:
    """Wk-weighted acceleration along dim 0 of [steps, envs].

    The signal is zero-padded to twice its length before the FFT, so the record is filtered
    as if the boat were at rest before and after it, the way a causal filter started from rest
    would see it, without wrapping the end of the record onto its start.
    """
    x = acceleration.detach().double().cpu().numpy()
    steps = x.shape[0]
    frequency = np.fft.rfftfreq(2 * steps, 1 / sample_rate)
    sections = wk_analog_sections()
    gain = np.array([analog_gain(sections, f) if f > 0 else 0.0 for f in frequency])
    spectrum = np.fft.rfft(x, n=2 * steps, axis=0) * gain.reshape(-1, *([1] * (x.ndim - 1)))
    weighted = np.fft.irfft(spectrum, n=2 * steps, axis=0)[:steps]
    return torch.from_numpy(weighted).to(acceleration.dtype).to(acceleration.device)


def rms(x: Tensor) -> float:
    return x.float().pow(2).mean().sqrt().item()


def per_minute(events: Tensor, dt: float) -> float:
    """Event count over [steps, envs] divided by the total sailed minutes."""
    minutes = events.numel() * dt / 60
    return events.float().sum().item() / minutes


def summarize(signals: dict[str, Tensor], dt: float) -> dict[str, float]:
    """The benchmark's metrics from stacked per-step signals."""
    return {
        "tracking_rms_m": rms(signals["height_error_m"]),
        "flap_rms_deg": rms(signals["flap_deg"]),
        "touchdowns_per_min": per_minute(signals["touchdown"], dt),
        "crashes_per_min": per_minute(signals["crashed"], dt),
        "ventilation_events_per_min": per_minute(signals["ventilation_onsets"], dt),
        "comfort_awz_m_s2": rms(wk_filter(signals["heave_accel_m_s2"], 1 / dt)),
    }
