import math

import pytest
import torch

from watergym.foils import (
    Foil,
    FoilLoads,
    foil_loads,
    free_surface_factor,
    helmbold_lift_slope,
    update_ventilation,
)
from watergym.rigid_body import Pose
from watergym.waves import regular_wave

CALM = regular_wave(1, amplitude=0.0, period=5.0)
SPEED = 10.0


def lift_coefficient(foil: Foil, depth_m: float) -> float:
    pose = Pose.from_eta(torch.tensor([[0.0, 0.0, depth_m, 0.0, 0.0, 0.0]]))
    nu = torch.tensor([[SPEED, 0.0, 0.0, 0.0, 0.0, 0.0]])
    loads = foil_loads(foil, CALM, 0.0, pose, nu, torch.zeros(1), torch.zeros(1, dtype=torch.bool))
    lift = -loads.force[0, :, 2].sum().item()
    return lift / (0.5 * 1025.0 * SPEED**2 * foil.span_m * foil.chord_m)


def test_deep_thin_foil_lifts_two_pi_alpha() -> None:
    alpha = math.radians(2)
    foil = Foil("wing", (0.0, 0.0, 0.0), span_m=1.0, chord_m=0.1, incidence=alpha)
    assert lift_coefficient(foil, depth_m=1.0) == pytest.approx(2 * math.pi * alpha, rel=0.02)


def test_helmbold_slope_at_aspect_ratio_four() -> None:
    assert helmbold_lift_slope(4.0) == pytest.approx(3.883, rel=0.001)


@pytest.mark.parametrize(
    ("depth_over_chord", "ratio"),
    [(0.0, 0.5), (0.25, 0.667), (0.5, 0.833), (1.0, 0.944), (2.0, 0.985), (3.0, 0.993)],
)
def test_free_surface_factor_matches_image_vortex_table(
    depth_over_chord: float, ratio: float
) -> None:
    assert free_surface_factor(torch.tensor(depth_over_chord)).item() == pytest.approx(
        ratio, abs=0.003
    )


def test_lift_falls_as_foil_nears_surface() -> None:
    foil = Foil("wing", (0.0, 0.0, 0.0), span_m=1.0, chord_m=0.1, incidence=math.radians(2))
    lifts = [lift_coefficient(foil, depth) for depth in (1.0, 0.3, 0.15, 0.08)]
    assert lifts == sorted(lifts, reverse=True)


def ventilation_after(foil: Foil, alphas_deg: list[float], depth_m: float) -> list[bool]:
    ventilated = torch.zeros(1, dtype=torch.bool)
    history = []
    for alpha in alphas_deg:
        loads = FoilLoads(
            points=torch.zeros(1, 1, 3),
            force=torch.zeros(1, 1, 3),
            alpha=torch.tensor([[math.radians(alpha)]]),
            depth=torch.tensor([[depth_m]]),
            speed=torch.tensor([[SPEED]]),
        )
        ventilated = update_ventilation(foil, ventilated, loads)
        history.append(bool(ventilated.item()))
    return history


def test_ventilation_has_hysteresis_without_chatter() -> None:
    foil = Foil("strut", (0.0, 0.0, 0.0), span_m=0.5, chord_m=0.1)
    sweep_up = [0, 5, 9.9, 10.1, 10.1]
    hold_between = [9.95, 10.05] * 50
    sweep_down = [8, 6, 4.1, 3.9]
    history = ventilation_after(foil, sweep_up + hold_between + sweep_down, depth_m=0.05)
    assert history[:3] == [False, False, False]
    assert all(history[3 : len(sweep_up) + len(hold_between) + 3])
    assert history[-1] is False


def test_deep_foil_never_ventilates() -> None:
    foil = Foil("wing", (0.0, 0.0, 0.0), span_m=1.0, chord_m=0.1)
    assert not any(ventilation_after(foil, [0, 6, 11, 11.9], depth_m=0.5))
