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
    foil = Foil("wing", (0.0, 0.0, 0.0), 1.0, 0.1, incidence=alpha, lift_slope=2 * math.pi)
    assert lift_coefficient(foil, depth_m=1.0) == pytest.approx(2 * math.pi * alpha, rel=0.02)


def test_default_lift_slope_is_helmbold_for_foils_and_struts() -> None:
    foil = Foil("wing", (0.0, 0.0, 0.0), span_m=1.0, chord_m=0.25)
    strut = Foil("strut", (0.0, 0.0, 0.0), 1.0, 0.25, span_axis=(0.0, 0.0, 1.0))
    assert foil.lift_slope == strut.lift_slope == helmbold_lift_slope(4.0)


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


def test_horizontal_foil_near_surface_follows_image_vortex_factor() -> None:
    foil = Foil("wing", (0.0, 0.0, 0.0), span_m=1.0, chord_m=0.1, incidence=math.radians(2))
    ratio = lift_coefficient(foil, depth_m=0.05) / lift_coefficient(foil, depth_m=2.0)
    assert ratio == pytest.approx(0.833, abs=0.01)


def strut_loads(strut: Foil, center_depth_m: float, ventilated: bool = False) -> FoilLoads:
    pose = Pose.from_eta(torch.tensor([[0.0, 0.0, center_depth_m, 0.0, 0.0, 0.0]]))
    nu = torch.tensor([[SPEED, 0.0, 0.0, 0.0, 0.0, 0.0]])
    return foil_loads(strut, CALM, 0.0, pose, nu, torch.zeros(1), torch.tensor([ventilated]))


def test_surface_piercing_strut_lifts_from_immersed_span_only() -> None:
    alpha = math.radians(2)
    strut = Foil("strut", (0.0, 0.0, 0.0), 1.0, 0.1, span_axis=(0.0, 0.0, 1.0), incidence=alpha)
    loads = strut_loads(strut, center_depth_m=0.0)
    side_force = loads.force[0, :, 1]
    assert torch.all(side_force[loads.depth[0] < -strut.immersion_band / 2] == 0)
    immersed_area = 0.5 * strut.span_m * strut.chord_m
    expected = 0.5 * 1025.0 * SPEED**2 * immersed_area * strut.lift_slope * alpha
    assert side_force.sum().item() == pytest.approx(expected, rel=0.01)


def test_ventilation_raises_profile_drag() -> None:
    foil = Foil("wing", (0.0, 0.0, 0.0), span_m=1.0, chord_m=0.1)
    pose = Pose.from_eta(torch.tensor([[0.0, 0.0, 1.0, 0.0, 0.0, 0.0]]))
    nu = torch.tensor([[SPEED, 0.0, 0.0, 0.0, 0.0, 0.0]])

    def drag(ventilated: bool) -> float:
        loads = foil_loads(foil, CALM, 0.0, pose, nu, torch.zeros(1), torch.tensor([ventilated]))
        return -loads.force[0, :, 0].sum().item()

    assert drag(True) == pytest.approx(foil.ventilated_drag_ratio * drag(False), rel=1e-4)
    assert drag(True) != pytest.approx(drag(False))


DT = 0.125  # exact in binary, so washout_time_s / DT steps add up exactly


def ventilation_history(foil: Foil, steps: list[tuple[float, float]]) -> list[bool]:
    """Feed (alpha in degrees, depth in m) of a single strip, one step of DT each."""
    ventilated = torch.zeros(1, dtype=torch.bool)
    wetting_time = torch.zeros(1)
    history = []
    for alpha, depth in steps:
        loads = FoilLoads(
            points=torch.zeros(1, 1, 3),
            force=torch.zeros(1, 1, 3),
            alpha=torch.tensor([[math.radians(alpha)]]),
            depth=torch.tensor([[depth]]),
            speed=torch.tensor([[SPEED]]),
        )
        ventilated = update_ventilation(foil, ventilated, loads, wetting_time, DT)
        history.append(bool(ventilated.item()))
    return history


def at_depth(depth_m: float, alphas_deg: list[float]) -> list[tuple[float, float]]:
    return [(alpha, depth_m) for alpha in alphas_deg]


WASHOUT_STEPS = 4  # Foil.washout_time_s / DT


def test_strut_ventilation_has_hysteresis_without_chatter() -> None:
    strut = Foil("strut", (0.0, 0.0, 0.0), span_m=0.5, chord_m=0.1, span_axis=(0.0, 0.0, 1.0))
    sweep_up = [0, 5, 9.9, 10.1]
    hold_near_onset = [9.95, 10.05] * 50
    hold_near_washout = [3.95, 4.05] * 50
    sweep_down = [3.9] * WASHOUT_STEPS
    history = ventilation_history(
        strut, at_depth(0.05, sweep_up + hold_near_onset + hold_near_washout + sweep_down)
    )
    assert history[:3] == [False, False, False]
    assert all(history[3:-1])
    assert history[-1] is False


def test_ventilated_foil_rewets_only_after_washout_time_deep_and_slow() -> None:
    foil = Foil("wing", (0.0, 0.0, 0.0), span_m=1.0, chord_m=0.1)
    assert foil.washout_time_s == WASHOUT_STEPS * DT
    vent_at_surface = at_depth(0.02, [11.0])
    dive_still_loaded = at_depth(0.5, [6.0] * 20)
    deep_and_slow = at_depth(0.5, [2.0] * WASHOUT_STEPS)
    history = ventilation_history(foil, vent_at_surface + dive_still_loaded + deep_and_slow)
    assert all(history[:-1])
    assert history[-1] is False


def test_foil_at_surface_with_low_alpha_stays_ventilated() -> None:
    foil = Foil("wing", (0.0, 0.0, 0.0), span_m=1.0, chord_m=0.1)
    history = ventilation_history(foil, at_depth(0.02, [11.0] + [1.0] * 10 * WASHOUT_STEPS))
    assert all(history)


def test_surface_piercing_strut_ventilates_from_its_shallow_strips() -> None:
    strut = Foil(
        "strut", (0.0, 0.0, 0.0), 1.0, 0.1, span_axis=(0.0, 0.0, 1.0), incidence=math.radians(11)
    )
    loads = strut_loads(strut, center_depth_m=0.4)
    assert loads.depth.mean() > strut.chord_m
    assert update_ventilation(strut, torch.zeros(1, dtype=torch.bool), loads).item()


def test_deep_foil_never_ventilates() -> None:
    foil = Foil("wing", (0.0, 0.0, 0.0), span_m=1.0, chord_m=0.1)
    assert not any(ventilation_history(foil, at_depth(0.5, [0, 6, 11, 11.9])))
