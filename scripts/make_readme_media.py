"""Regenerate the README media in docs/images/readme/ (needs the viz extra, runs headless):

    OMP_NUM_THREADS=2 nice -n 19 uv run --extra viz python scripts/make_readme_media.py
    ... make_readme_media.py hero vessels      # only some of: hero vessels sea_states stack

Every image is a Newton viewer frame rendered at twice its final size and downscaled.
"""

import argparse
import math
import multiprocessing
import time
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import warp as wp
from PIL import Image, ImageDraw, ImageFont

from watergym import SeaState, WaterEnv
from watergym.vessel import Vessel
from watergym.vessels import bluerov2, box_barge, moth, otter
from watergym.vessels.moth_vessel import wand_action
from watergym.viewer import WaterViewer, make_viewer, vec3_array

OUT = Path(__file__).resolve().parents[1] / "docs" / "images" / "readme"
SEA_COLOR = (0.06, 0.40, 0.58)
FLOOR_COLOR = (0.04, 0.27, 0.42)
# Fog fades to HORIZON; the viewer tone-maps it to about HAZE, the flat sky colour.
HORIZON = (0.26, 0.43, 0.57)
HAZE = (75 / 255, 148 / 255, 192 / 255)
SUPERSAMPLE = 1  # the GL viewer renders at most 1280x720; MSAA does the smoothing
SUN = np.array([0.7, -0.3, 0.55])  # low in the north, so glints run toward a camera looking north
FONTS = ("/System/Library/Fonts/HelveticaNeue.ttc", "/System/Library/Fonts/Helvetica.ttc")
FONT_BOLD_INDEX = 1


OCEAN_CORNERS = [
    [-600.0, -600.0, -1.6],
    [600.0, -600.0, -1.6],
    [600.0, 600.0, -1.6],
    [-600.0, 600.0, -1.6],
]


def make_scene(
    num_envs: int,
    patch_m: float,
    width: int,
    height: int,
    spacing_m: float | None = None,
    edge_fade: float = 0.25,
    floor_color: tuple[float, float, float] = FLOOR_COLOR,
) -> tuple[WaterViewer, object]:
    """A headless viewer styled for the README: one ocean from the foreground to the horizon,
    patches that taper to flat at their borders so they join into a single sea."""
    viewer = make_viewer(
        "gl", headless=True, width=width * SUPERSAMPLE, height=height * SUPERSAMPLE
    )
    renderer = viewer.renderer
    renderer.draw_sky = False
    renderer.sky_lower, renderer.sky_upper = HORIZON, HAZE
    renderer.ambient_sky = (0.75, 0.82, 0.9)
    renderer.ambient_ground = (0.2, 0.3, 0.35)
    scene = WaterViewer(
        viewer,
        num_envs,
        patch_size_m=patch_m,
        patch_resolution=min(int(patch_m * 8) + 1, 161),
        spacing_m=spacing_m or patch_m,
        edge_fade=edge_fade,
        sea_color=SEA_COLOR,
        sea_opacity=0.8,
        sea_roughness=0.45,
    )
    renderer.spotlight_enabled = False
    renderer.specular_scale = 0.4
    renderer._sun_direction = SUN / np.linalg.norm(SUN)
    renderer.exposure = 1.4
    viewer.log_mesh(
        "ocean_floor",
        vec3_array(torch.tensor(OCEAN_CORNERS)),
        wp.array(np.array([0, 1, 2, 0, 2, 3], np.int32), dtype=wp.int32),
        color=floor_color,
        roughness=1.0,
        backface_culling=False,
    )
    return scene, viewer


def snapshot(viewer, size: tuple[int, int]) -> Image.Image:
    frame = Image.fromarray(viewer.get_frame().numpy())
    return frame.resize(size, Image.Resampling.LANCZOS)


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    for path in FONTS:
        try:
            return ImageFont.truetype(path, size, index=FONT_BOLD_INDEX if bold else 0)
        except OSError:
            continue
    return ImageFont.load_default(size)


def add_label(image: Image.Image, title: str, subtitle: str = "") -> None:
    """A dark rounded pill at the bottom left, readable on any sea."""
    draw = ImageDraw.Draw(image, "RGBA")
    scale = image.width / 800
    big, small, pad = font(round(26 * scale), True), font(round(17 * scale)), round(14 * scale)
    title_w = draw.textlength(title, big)
    subtitle_w = draw.textlength(subtitle, small) if subtitle else 0
    height = big.size + (small.size + 4 * scale if subtitle else 0) + 2 * pad
    x0, y0 = round(24 * scale), image.height - round(24 * scale) - height
    box = (x0, y0, x0 + max(title_w, subtitle_w) + 2 * pad, y0 + height)
    draw.rounded_rectangle(box, radius=round(14 * scale), fill=(8, 22, 34, 190))
    draw.text((x0 + pad, y0 + pad - 3 * scale), title, font=big, fill=(255, 255, 255, 255))
    if subtitle:
        draw.text(
            (x0 + pad, y0 + pad + big.size + 2 * scale),
            subtitle,
            font=small,
            fill=(190, 215, 230, 255),
        )


def moth_fleet(
    num_envs: int, hs: float, tp: float = 3.2, seed: int = 0, warmup_s: float = 4.0
) -> WaterEnv:
    sea = SeaState(hs=hs, tp=tp, heading_rad=math.pi, spreading=10.0, num_components=48)
    env = WaterEnv(moth(), num_envs, sea)
    env.reset(seed=seed)
    advance(env, warmup_s)
    return env


def advance(env: WaterEnv, seconds: float) -> None:
    for _ in range(round(seconds / env.dt)):
        env.step(wand_action(env.sea, env.t, env.eta))


def draw_moths(scene: WaterViewer, env: WaterEnv) -> None:
    scene.draw(env.t, env.sea, env.vessel, env.eta)


def hero_camera(scene: WaterViewer, phase: float) -> None:
    """A slow drift around the fleet that returns to its start at phase 1."""
    angle = 2 * math.pi * phase
    scene.look_at_grid(
        distance=0.40 + 0.03 * math.sin(angle),
        pitch_deg=-12 - 1.5 * math.sin(angle + 1),
        yaw_deg=-33 + 12 * math.cos(angle),
    )


def hero_frames(width: int, seconds: float, fps: float, fade_s: float = 0.9) -> list[Image.Image]:
    height = width * 9 // 16
    num_envs = 12
    env = moth_fleet(num_envs, hs=0.6, tp=2.5)
    scene, viewer = make_scene(num_envs, 8.0, 1280, 720)
    steps_per_frame = round(1 / fps / env.dt)
    loop_frames = round(seconds * fps)
    fade_frames = round(fade_s * fps)

    frames = []
    for i in range(loop_frames + fade_frames):
        hero_camera(scene, (i % loop_frames) / loop_frames)
        draw_moths(scene, env)
        frames.append(snapshot(viewer, (width, height)))
        if i == loop_frames // 4:
            snapshot(viewer, (1280, 720)).save(OUT / "hero.png", optimize=True)
        advance(env, steps_per_frame * env.dt)
        time.sleep(0.05)

    # The sim does not repeat, so the first frames dissolve in from the frames that come
    # after the end of the loop, which makes the last frame lead straight into the first.
    looped = frames[fade_frames:loop_frames]
    for i in range(fade_frames):
        blended = Image.blend(frames[loop_frames + i], frames[i], (i + 1) / (fade_frames + 1))
        looped.insert(i, blended)
    viewer.close()
    return looped


def make_hero(width: int = 800, seconds: float = 6.2, fps: float = 12.5) -> None:
    save_gif(hero_frames(width, seconds, fps), OUT / "hero.gif", round(100 / fps))


def save_gif(frames: list[Image.Image], path: Path, delay_cs: int) -> None:
    sample = Image.new("RGB", (frames[0].width, frames[0].height * 8))
    for k, frame in enumerate(frames[:: max(1, len(frames) // 8)][:8]):
        sample.paste(frame, (0, k * frames[0].height))
    palette = sample.quantize(colors=96, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    quantized = [f.quantize(palette=palette, dither=Image.Dither.NONE) for f in frames]
    quantized[0].save(
        path,
        save_all=True,
        append_images=quantized[1:],
        duration=delay_cs * 10,
        loop=0,
        optimize=True,
        disposal=1,
    )


@dataclass
class Panel:
    title: str
    subtitle: str
    vessel: Vessel
    sea: SeaState
    patch_m: float
    camera: tuple[float, float, float]  # distance, pitch_deg, yaw_deg for look_at_grid
    seconds: float = 6.0
    action: Callable[[WaterEnv], torch.Tensor | None] = lambda env: None
    dt: float = 0.02
    sea_opacity: float = 0.8


def render_panel(panel: Panel, size: tuple[int, int], seed: int = 0) -> Image.Image:
    env = WaterEnv(panel.vessel, 1, panel.sea, dt=panel.dt)
    env.reset(seed=seed)
    for _ in range(round(panel.seconds / panel.dt)):
        env.step(panel.action(env))
    scene, viewer = make_scene(1, panel.patch_m, *size, edge_fade=0.1)
    scene.sea_opacity = panel.sea_opacity
    scene.look_at_grid(*panel.camera)
    scene.draw(env.t, env.sea, env.vessel, env.eta)
    image = snapshot(viewer, size)
    viewer.close()
    return image


def hold_depth(depth_m: float) -> Callable[[WaterEnv], torch.Tensor]:
    def action(env: WaterEnv) -> torch.Tensor:
        vertical = (2.0 * (depth_m - env.eta[:, 2]) - env.nu[:, 2]).clamp(-1, 1)
        return torch.cat((torch.zeros(1, 4), vertical[:, None].expand(1, 4)), dim=-1)

    return action


def vessel_panels() -> list[Panel]:
    return [
        Panel(
            "Moth hydrofoil",
            "Flies on a wand-controlled foil flap",
            moth(),
            SeaState(hs=0.5, tp=2.6, heading_rad=math.pi, spreading=10.0),
            30.0,
            (0.46, -16, -40),
            action=lambda env: wand_action(env.sea, env.t, env.eta),
        ),
        Panel(
            "Otter USV",
            "Twin-pontoon catamaran, two thrusters",
            otter(),
            SeaState(hs=0.6, tp=3.5, heading_rad=math.pi, spreading=10.0),
            30.0,
            (0.17, -15, -40),
            action=lambda env: torch.full((1, 2), 0.5),
        ),
        Panel(
            "BlueROV2",
            "Six thrusters, holding 0.5 m depth",
            bluerov2(),
            SeaState(hs=0.5, tp=3.0, spreading=10.0),
            30.0,
            (0.09, -14, -40),
            action=hold_depth(0.5),
            sea_opacity=0.45,
        ),
        Panel(
            "Box barge",
            "10 m hull, beam seas",
            box_barge(),
            SeaState(hs=1.2, tp=5.0, heading_rad=math.pi / 2, spreading=20.0),
            60.0,
            (0.4, -15, -40),
            seconds=8.0,
            dt=0.05,
        ),
    ]


def sea_state_panels() -> list[Panel]:
    names = {0.0: "calm", 0.3: "moderate", 0.6: "rough"}
    return [
        Panel(
            f"Hs {hs:g} m",
            names[hs],
            moth(),
            SeaState(hs=hs, tp=2.8, heading_rad=math.pi, spreading=10.0),
            30.0,
            (0.42, -20, -50),
            action=lambda env: wand_action(env.sea, env.t, env.eta),
        )
        for hs in names
    ]


PANELS = {"vessels": vessel_panels, "sea_states": sea_state_panels}


def render_labelled(kind: str, index: int, size: tuple[int, int]) -> Image.Image:
    """Runs in a fresh process: Newton's GL viewer renders black once a second one is made."""
    panel = PANELS[kind]()[index]
    image = render_panel(panel, size)
    add_label(image, panel.title, panel.subtitle)
    return image


def make_sheet(kind: str, size: tuple[int, int], columns: int) -> None:
    count = len(PANELS[kind]())
    context = multiprocessing.get_context("spawn")
    time.sleep(0.05)
    with ProcessPoolExecutor(1, mp_context=context, max_tasks_per_child=1) as pool:
        tiles = list(pool.map(render_labelled, [kind] * count, range(count), [size] * count))
    rows = math.ceil(count / columns)
    sheet = Image.new("RGB", (size[0] * columns, size[1] * rows))
    for k, tile in enumerate(tiles):
        sheet.paste(tile, ((k % columns) * size[0], (k // columns) * size[1]))
    sheet.save(OUT / f"{kind}.png", optimize=True)


def make_vessels() -> None:
    make_sheet("vessels", (800, 500), columns=2)


def make_sea_states() -> None:
    make_sheet("sea_states", (640, 720), columns=3)


STACK_LAYERS = [
    ("Waves", "JONSWAP spectrum", "random phases per env"),
    ("Hydrostatics + foils", "buoyancy, lift and drag", "foil ventilation"),
    ("6-DOF rigid body", "added mass, damping", "RK4, NED frame"),
    ("Batched WaterEnv", "reset, step, reward", "[envs, ...] tensors"),
    ("rsl_rl / your trainer", "PPO or any policy", "obs in, actions out"),
]
STACK_STYLE = """
  :root { --ink: #1f2328; --muted: #57606a; --fill: #0969da; --line: #0969da;
          --accent: #bf5700; }
  @media (prefers-color-scheme: dark) {
    :root { --ink: #e6edf3; --muted: #9da7b3; --fill: #58a6ff; --line: #58a6ff; --accent: #f0883e; }
  }
  .box { fill: var(--fill); fill-opacity: 0.12; stroke: var(--line); stroke-width: 1.5; }
  .viewer { fill: var(--accent); fill-opacity: 0.1; stroke: var(--accent); stroke-width: 1.5;
            stroke-dasharray: 6 4; }
  .title { fill: var(--ink); font: 600 17px -apple-system, "Segoe UI", Helvetica, sans-serif; }
  .note { fill: var(--muted); font: 13px -apple-system, "Segoe UI", Helvetica, Arial, sans-serif; }
  .arrow { stroke: var(--line); stroke-width: 1.8; fill: none; marker-end: url(#head); }
  .dashed { stroke: var(--accent); stroke-width: 1.8; fill: none; stroke-dasharray: 6 4;
            marker-end: url(#head-accent); }
  #head path { fill: var(--line); }
  #head-accent path { fill: var(--accent); }
"""


def svg_text(css: str, x: float, y: float, text: str) -> str:
    return f'<text class="{css}" x="{x}" y="{y}" text-anchor="middle">{text}</text>'


def svg_box(css: str, x: float, y: float, w: float, h: float) -> str:
    return f'<rect class="{css}" x="{x}" y="{y}" width="{w}" height="{h}" rx="10"/>'


def make_stack() -> None:
    """Hand-laid-out diagram: five layers left to right, the viewer hanging off the env."""
    width, box_w, box_h, gap, top = 1060, 180, 92, 30, 40
    parts = []
    for i, (title, line1, line2) in enumerate(STACK_LAYERS):
        x = 20 + i * (box_w + gap)
        centre = x + box_w / 2
        parts.append(svg_box("box", x, top, box_w, box_h))
        parts.append(svg_text("title", centre, top + 32, title))
        parts.append(svg_text("note", centre, top + 56, line1))
        parts.append(svg_text("note", centre, top + 74, line2))
        if i:
            parts.append(f'<path class="arrow" d="M{x - gap + 4} {top + box_h / 2} H{x - 4}"/>')
    env_x = 20 + 3 * (box_w + gap) + box_w / 2
    viewer_y, viewer_h = top + box_h + 56, box_h - 20
    parts.append(f'<path class="dashed" d="M{env_x} {top + box_h + 4} V{viewer_y - 4}"/>')
    parts.append(svg_box("viewer", env_x - box_w / 2, viewer_y, box_w, viewer_h))
    parts.append(svg_text("title", env_x, viewer_y + 32, "Newton viewer"))
    parts.append(svg_text("note", env_x, viewer_y + 52, "optional, headless ok"))
    marker = (
        '<marker id="{}" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7"'
        ' orient="auto-start-reverse"><path d="M0 0L10 5L0 10z"/></marker>'
    )
    label = "WaterGym layers from waves to trainer, with a viewer attached to the env"
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" role="img" aria-label="{label}" '
        f'viewBox="0 0 {width} {viewer_y + viewer_h + 20}">'
        f"<style>{STACK_STYLE}</style>"
        f"<defs>{marker.format('head')}{marker.format('head-accent')}</defs>"
        f"{''.join(parts)}</svg>\n"
    )
    (OUT / "stack.svg").write_text(svg)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "targets", nargs="*", choices=["hero", "vessels", "sea_states", "stack"], default=[]
    )
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    for target in args.targets or ["hero", "vessels", "sea_states", "stack"]:
        {
            "hero": make_hero,
            "vessels": make_vessels,
            "sea_states": make_sea_states,
            "stack": make_stack,
        }[target]()


if __name__ == "__main__":
    main()
