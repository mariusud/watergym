"""Regenerate the README media in docs/images/readme/ (needs the viz extra, runs headless):

    OMP_NUM_THREADS=2 nice -n 19 uv run --extra viz python scripts/make_readme_media.py
    ... make_readme_media.py hero vessels      # only some of: hero vessels sea_states stack

Every image is a Newton viewer frame rendered at twice its final size and downscaled.
"""

import argparse
import math
import multiprocessing
import shutil
import subprocess
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
from watergym.rigid_body import Pose
from watergym.vessel import Vessel
from watergym.vessels import bluerov2, box_barge, moth, otter
from watergym.vessels.moth_vessel import FOIL_Z, HULL_BOTTOM_Z, RUDDER_X, wand_action
from watergym.viewer import WaterViewer, make_viewer, vec3_array
from watergym.waves import elevation

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
# Clearer water around the Moth, so its foils show below the surface.
MOTH_SEA_OPACITY = 0.6


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
    sea_opacity: float = 0.8,
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
        sea_opacity=sea_opacity,
        sea_roughness=0.45,
    )
    renderer.spotlight_enabled = False
    renderer.specular_scale = 0.4
    renderer._sun_direction = SUN / np.linalg.norm(SUN)
    renderer.exposure = 1.4
    log_outer_sea(viewer, scene)
    viewer.log_mesh(
        "ocean_floor",
        vec3_array(torch.tensor(OCEAN_CORNERS)),
        wp.array(np.array([0, 1, 2, 0, 2, 3], np.int32), dtype=wp.int32),
        color=floor_color,
        roughness=1.0,
        backface_culling=False,
    )
    return scene, viewer


def log_outer_sea(viewer, scene: WaterViewer, reach_m: float = 600.0) -> None:
    """Flat water from the edge of the env grid out to the horizon. The patches taper to
    flat at their borders, so the two join without a seam and no patch edge shows."""
    half = scene.patch[:, 0].max().item()
    x0, y0 = (scene.offsets[:, :2].min(0).values - half).tolist()
    x1, y1 = (scene.offsets[:, :2].max(0).values + half).tolist()
    inner = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
    outer = [[x0 - reach_m, y0 - reach_m], [x1 + reach_m, y0 - reach_m]]
    outer += [[x1 + reach_m, y1 + reach_m], [x0 - reach_m, y1 + reach_m]]
    corners = torch.tensor(inner + outer)
    corners = torch.cat((corners, torch.zeros(8, 1)), -1)
    frame = [[k, (k + 1) % 4, 4 + (k + 1) % 4, 4 + k] for k in range(4)]
    # Wound counter-clockwise seen from above, or the shader lights it as facing down.
    quads = np.array([[a, c, b, a, d, c] for a, b, c, d in frame], np.int32).reshape(-1)
    viewer.log_mesh(
        "outer_sea",
        vec3_array(corners),
        wp.array(quads, dtype=wp.int32),
        normals=vec3_array(torch.tensor([[0.0, 0.0, 1.0]]).expand(8, 3)),
        color=scene.sea_color,
        roughness=scene.sea_roughness,
        opacity=scene.sea_opacity,
        backface_culling=False,
    )


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


HERO_SIDE = 8  # an 8 x 8 grid of Moths, each in its own random sea
HERO_PATCH_M = 8.0
HERO_STAR = HERO_SIDE - 1  # the close-up Moth sits on the near corner, so the camera
# pulls back over open water instead of through the fleet
HERO_FPS = 20.0
CLOSE_S, ZOOM_S, HOLD_S, FADE_S = 2.6, 3.6, 1.4, 0.6


HERO_SEA = SeaState(hs=0.6, tp=2.8, heading_rad=math.pi, spreading=10.0, num_components=48)


def foils_stay_wet(env: WaterEnv) -> bool:
    """Every Moth has both foils under the surface and its hull bottom clear of it."""
    points = torch.tensor([[0.0, 0.0, FOIL_Z], [RUDDER_X, 0.0, FOIL_Z], [0.0, 0.0, HULL_BOTTOM_Z]])
    world = Pose.from_eta(env.eta).to_world(points)
    depth = world[..., 2] + elevation(env.sea, world, env.t)
    return bool((depth[:, :2] > 0.05).all() and (depth[:, 2] < 0.05).all())


def hero_states(seconds: float) -> tuple[WaterEnv, list[tuple[torch.Tensor, torch.Tensor]]]:
    """(t, eta) for every frame of a run in which no Moth crashes or launches, trying
    seeds until one does."""
    for seed in range(20):
        env = WaterEnv(moth(), HERO_SIDE**2, HERO_SEA)
        env.reset(seed=seed)
        advance(env, 4.0)
        states, steps_per_frame = [], round(1 / HERO_FPS / env.dt)
        for _ in range(round(seconds * HERO_FPS)):
            if not foils_stay_wet(env):
                break
            states.append((env.t.clone(), env.eta.clone()))
            advance(env, steps_per_frame * env.dt)
        else:
            print(f"hero: seed {seed}")
            return env, states
    raise RuntimeError("no seed keeps every Moth flying")


def advance(env: WaterEnv, seconds: float) -> None:
    for _ in range(round(seconds / env.dt)):
        env.step(wand_action(env.sea, env.t, env.eta))


def smoothstep(x: float) -> float:
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def aim(viewer, target: torch.Tensor, distance: float, pitch: float, yaw: float) -> None:
    """Camera `distance` m from `target` (viewer frame, z up), looking along yaw, pitch."""
    p, y = math.radians(pitch), math.radians(yaw)
    back = torch.tensor([math.cos(p) * math.cos(y), math.cos(p) * math.sin(y), math.sin(p)])
    viewer.set_camera(wp.vec3(*(target - distance * back).tolist()), pitch, yaw)


def hero_camera(scene: WaterViewer, seconds: float) -> None:
    """Low and ahead of one Moth, then an eased pull back and up to the whole fleet."""
    star = scene.offsets[HERO_STAR] + torch.tensor([0.0, 0.0, 0.35])
    fleet = scene.offsets.mean(0)
    s = smoothstep((seconds - CLOSE_S) / ZOOM_S)
    drift = min(seconds / CLOSE_S, 1.0)
    target = star + s * (fleet - star)
    distance = 5.5 * (85.0 / 5.5) ** s
    pitch = -6 - 22 * smoothstep(2 * s)
    aim(scene.viewer, target, distance, pitch, -160 + 10 * drift + 15 * s)


def hero_frames(width: int) -> tuple[list[Image.Image], list[Image.Image]]:
    """GIF-size frames and full 1280 x 720 frames. The clip ends by dissolving from the
    wide shot into its own opening frames, so it loops without a jump."""
    loop_s = CLOSE_S + ZOOM_S + HOLD_S
    env, states = hero_states(loop_s + FADE_S)
    scene, viewer = make_scene(HERO_SIDE**2, HERO_PATCH_M, 1280, 720, sea_opacity=MOTH_SEA_OPACITY)
    small, full = [], []
    for i, (t, eta) in enumerate(states):
        hero_camera(scene, i / HERO_FPS)
        scene.draw(t, env.sea, env.vessel, eta)
        frame = snapshot(viewer, (1280, 720))
        full.append(frame)
        small.append(frame.resize((width, width * 9 // 16), Image.Resampling.LANCZOS))
        if i == round(1.2 * HERO_FPS):
            frame.save(OUT / "hero.png", optimize=True)
        time.sleep(0.05)
    viewer.close()
    loop, fade = round(loop_s * HERO_FPS), round(FADE_S * HERO_FPS)
    for frames in (small, full):
        dissolve = [
            Image.blend(frames[loop + i], frames[i], (i + 1) / (fade + 1)) for i in range(fade)
        ]
        frames[:] = frames[fade:loop] + dissolve
    return small, full


def make_hero(width: int = 800) -> None:
    small, full = hero_frames(width)
    every = 2  # the GIF runs at half the video frame rate to stay small
    save_gif(small[::every], OUT / "hero.gif", round(100 * every / HERO_FPS))
    if shutil.which("ffmpeg"):
        save_mp4(full, OUT / "hero.mp4", HERO_FPS)


def save_mp4(frames: list[Image.Image], path: Path, fps: float) -> None:
    command = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24"]
    command += ["-s", f"{frames[0].width}x{frames[0].height}", "-r", str(fps), "-i", "-"]
    command += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", "-movflags"]
    command += ["+faststart", str(path)]
    subprocess.run(command, input=b"".join(f.tobytes() for f in frames), check=True)


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
    scene, viewer = make_scene(
        1, panel.patch_m, *size, edge_fade=0.1, sea_opacity=panel.sea_opacity
    )
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
            (0.36, -10, -40),
            action=lambda env: wand_action(env.sea, env.t, env.eta),
            sea_opacity=MOTH_SEA_OPACITY,
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
            "Eight thrusters, holding 0.5 m depth",
            bluerov2(),
            SeaState(hs=0.5, tp=3.0, spreading=10.0),
            30.0,
            (0.075, -13, -40),
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
            sea_opacity=MOTH_SEA_OPACITY,
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
