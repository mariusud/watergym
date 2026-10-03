"""Benchmark metrics from per-step signals shaped [steps, envs].

Comfort is the ISO 2631-1:1997 frequency-weighted RMS acceleration a_w (clause 6.1), using
the vertical weighting Wk of Annex A. The weighting is a cascade of four analog filters:

    band limit   Hh (high-pass, f1) and Hl (low-pass, f2)
    a-v transition  Ht = (1 + s/w3) / (1 + s/(Q4 w4) + (s/w4)^2)
    upward step  Hs = (1 + s/(Q5 w5) + (s/w5)^2) / (1 + s/(Q6 w6) + (s/w6)^2) (w5/w6)^2

each turned into a digital biquad by the bilinear transform and run in series.
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
    s = 2j * math.pi * frequency_hz
    gain = 1.0 + 0j
    for num, den in sections:
        gain *= np.polyval(num[::-1], s) / np.polyval(den[::-1], s)
    return abs(gain)


def bilinear(num: Polynomial, den: Polynomial, sample_rate: float) -> tuple[np.ndarray, ...]:
    """Second-order analog section to digital (b, a), with s = 2 fs (1 - z^-1) / (1 + z^-1)."""
    k = 2 * sample_rate
    minus, plus = np.array([1.0, -1.0]), np.array([1.0, 1.0])
    powers = [np.polymul(np.polymul([1.0], plus), plus)]
    powers.append(np.polymul(minus, plus) * k)
    powers.append(np.polymul(minus, minus) * k**2)

    def digital(coefficients: Polynomial) -> np.ndarray:
        padded = list(coefficients) + [0.0] * (3 - len(coefficients))
        return sum(c * p for c, p in zip(padded, powers, strict=True))

    b, a = digital(num), digital(den)
    return b / a[0], a / a[0]


def wk_filter(acceleration: Tensor, sample_rate: float) -> Tensor:
    """Wk-weighted acceleration, filtering along dim 0 of [steps, envs]."""
    x = acceleration.double()
    for num, den in wk_analog_sections():
        b, a = bilinear(num, den, sample_rate)
        y = torch.zeros_like(x)
        x1 = x2 = y1 = y2 = torch.zeros_like(x[0])
        for n in range(len(x)):
            y[n] = b[0] * x[n] + b[1] * x1 + b[2] * x2 - a[1] * y1 - a[2] * y2
            x1, x2, y1, y2 = x[n], x1, y[n], y1
        x = y
    return x.to(acceleration.dtype)


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
