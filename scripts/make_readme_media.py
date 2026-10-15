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
    scene_type: type[WaterViewer] = WaterViewer,
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
    scene = scene_type(
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


def log_outer_sea(
    viewer,
    scene: WaterViewer,
    bounds: tuple[float, float, float, float] | None = None,
    reach_m: float = 600.0,
) -> None:
    """Flat water from `bounds` (x0, y0, x1, y1; by default the env grid) out to the
    horizon. What lies inside fades to flat at its border, so no edge shows."""
    half = scene.patch[:, 0].max().item()
    x0, y0 = (scene.offsets[:, :2].min(0).values - half).tolist()
    x1, y1 = (scene.offsets[:, :2].max(0).values + half).tolist()
    if bounds is not None:
        x0, y0, x1, y1 = bounds
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


HERO_SIDE = 8  # an 8 x 8 fleet of Moths sailing through one shared sea
HERO_SPACING_M = 9.0
HERO_STANDOFF_M = 10.0
HERO_FPS = 30.0
ISOLATE_S, REVEAL_S, HOLD_S = 3.0, 4.4, 2.0
HERO_SEA = SeaState(hs=0.6, tp=2.8, heading_rad=math.pi, spreading=10.0, num_components=48)
# The ocean mesh, in metres around the hero: 0.2 m cells within 15 m of it, 0.5 m beyond,
# fading to flat over the last 10 m before the flat far ocean.
OCEAN_BOUNDS = (-35.0, -100.0, 100.0, 35.0)
OCEAN_FINE_M, OCEAN_FINE_STEP, OCEAN_STEP, OCEAN_FADE_M = 15.0, 0.2, 0.5, 10.0


def fleet_homes() -> torch.Tensor:
    """Start positions [N, 3] in the viewer frame (z up), hero (env 0) at the origin and
    the rest on a scattered grid beyond it toward +x and -y."""
    cells = torch.arange(HERO_SIDE**2)
    grid = torch.stack(
        (cells % HERO_SIDE, -(cells // HERO_SIDE), torch.zeros_like(cells)), -1
    ).float()
    scatter = torch.rand(len(cells), 3, generator=torch.Generator().manual_seed(1)) - 0.5
    scatter *= torch.tensor([3.0, 3.0, 0.0])
    scatter[0] = 0.0
    homes = grid * HERO_SPACING_M + scatter
    # The hero stands off the fleet, so not even a neighbour's shadow reaches the close-up.
    homes[1:] += torch.tensor([HERO_STANDOFF_M, -HERO_STANDOFF_M, 0.0])
    return homes


def graded_axis(lo: float, hi: float) -> torch.Tensor:
    return torch.cat(
        (
            torch.arange(lo, -OCEAN_FINE_M, OCEAN_STEP),
            torch.arange(-OCEAN_FINE_M, OCEAN_FINE_M, OCEAN_FINE_STEP),
            torch.arange(OCEAN_FINE_M, hi + 1e-3, OCEAN_STEP),
        )
    )


class OceanScene(WaterViewer):
    """The fleet on one continuous ocean. Every Moth sails at its own world position in
    the same sea, and the ocean is a single mesh of that sea around the hero, so there
    are no per-env patches and no seams. The view follows the hero."""

    def setup(self, homes: torch.Tensor) -> None:
        self.homes = homes
        self.offsets = torch.zeros(self.num_envs, 3)
        x0, y0, x1, y1 = OCEAN_BOUNDS
        xs, ys = graded_axis(x0, x1), graded_axis(y0, y1)
        self.axes = (xs, ys)
        gx, gy = torch.meshgrid(xs, ys, indexing="ij")
        self.ocean = torch.stack((gx, gy, torch.zeros_like(gx)), -1).reshape(-1, 3)
        inside = torch.stack((gx - x0, x1 - gx, gy - y0, y1 - gy)).amin(0).reshape(-1)
        fade = (inside / OCEAN_FADE_M).clamp(0, 1)
        self.ocean_weight = fade * fade * (3 - 2 * fade)
        nx, ny = gx.shape
        i, j = torch.meshgrid(torch.arange(nx - 1), torch.arange(ny - 1), indexing="ij")
        corner = (i * ny + j).reshape(-1)
        quads = torch.stack((corner, corner + ny, corner + ny + 1, corner + 1), -1)
        triangles = torch.cat((quads[:, [0, 1, 2]], quads[:, [0, 2, 3]])).reshape(-1)
        self.ocean_triangles = wp.array(triangles.numpy().astype(np.int32), dtype=wp.int32)
        log_outer_sea(self.viewer, self, OCEAN_BOUNDS)

    def draw(self, t, sea, vessel=None, eta=None, foil_loads=None, newton_per_m=1500.0):
        hero = eta[0, :2].detach().cpu()
        followed = torch.zeros(self.num_envs, 3)
        followed[:, :2] = hero
        self.viewer.begin_frame(float(t.reshape(-1)[0]))
        self._draw_ocean(t, sea, hero)
        self._draw_vessels(vessel, eta.detach().cpu(), followed)
        self.viewer.end_frame()

    def _draw_ocean(self, t, sea, hero: torch.Tensor) -> None:
        world = self.ocean * torch.tensor([1.0, -1.0, 0.0])
        world[:, :2] += hero
        surface = elevation(sea.subset([0]), world[None].to(sea.amplitude.device), t[:1])[0]
        vertices = self.ocean.clone()
        vertices[:, 2] = surface.cpu() * self.ocean_weight
        grid = vertices.reshape(len(self.axes[0]), len(self.axes[1]), 3)
        along_x, along_y = torch.gradient(grid, spacing=self.axes, dim=(0, 1))
        normals = torch.nn.functional.normalize(torch.linalg.cross(along_x, along_y), dim=-1)
        self.viewer.log_mesh(
            "ocean",
            vec3_array(vertices),
            self.ocean_triangles,
            normals=vec3_array(normals.reshape(-1, 3)),
            color=self.sea_color,
            roughness=self.sea_roughness,
            opacity=self.sea_opacity,
            dynamic=True,
            backface_culling=False,
        )


def foils_stay_wet(env: WaterEnv) -> bool:
    """Every Moth has both foils under the surface and its hull bottom clear of it."""
    points = torch.tensor([[0.0, 0.0, FOIL_Z], [RUDDER_X, 0.0, FOIL_Z], [0.0, 0.0, HULL_BOTTOM_Z]])
    world = Pose.from_eta(env.eta).to_world(points)
    depth = world[..., 2] + elevation(env.sea, world, env.t)
    return bool((depth[:, :2] > 0.05).all() and (depth[:, 2] < 0.05).all())


def hero_states(seconds: float) -> tuple[WaterEnv, list[tuple[torch.Tensor, torch.Tensor]]]:
    """(t, eta) for every frame of a run in which every Moth starts at its home position
    in one shared sea and none crashes, launches or resets, trying seeds until one works."""
    homes = fleet_homes()
    for seed in range(20):
        env = WaterEnv(moth(), HERO_SIDE**2, HERO_SEA)
        env.reset(seed=seed)
        env.sea = env.sea.subset([0] * env.num_envs)
        env.eta[:, 0] += homes[:, 0]
        env.eta[:, 1] -= homes[:, 1]
        advance(env, 4.0)
        states, steps_per_frame = [], round(1 / HERO_FPS / env.dt)
        for frame in range(round(seconds * HERO_FPS)):
            expected_t = 4.0 + frame * steps_per_frame * env.dt
            if not foils_stay_wet(env) or ((env.t - expected_t).abs() > 1e-3).any():
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


def smootherstep(x: float) -> float:
    """0 to 1 with zero speed and zero acceleration at both ends."""
    x = min(max(x, 0.0), 1.0)
    return x**3 * (x * (6 * x - 15) + 10)


def look_at(viewer, eye: torch.Tensor, target: torch.Tensor) -> None:
    d = target - eye
    yaw = math.degrees(math.atan2(d[1], d[0]))
    pitch = math.degrees(math.atan2(d[2], d[:2].norm()))
    viewer.set_camera(wp.vec3(*eye.tolist()), pitch, yaw)


def hero_camera(scene: OceanScene, seconds: float) -> None:
    """One continuous drone move, in polar coordinates around the hero Moth.

    Isolate: low (1.4 m) and 7 m out, ahead of the beam on the fleet side, looking back
    past the boat at open sea while orbiting slowly at 4 deg/s.
    Reveal: an S-curve (smootherstep, no linear segment) swings the camera about 180 deg
    round the bow, out to 16 m and up to 24 m, while the aim slides from the hero to the
    middle of the fleet, so the other Moths enter from the frame edge as the camera turns.
    Hold: the same slow orbit carries on over the wide shot.
    """
    hero = scene.homes[0] + torch.tensor([0.0, 0.0, 1.7])
    fleet = scene.homes.mean(0)
    to_fleet = math.degrees(math.atan2(fleet[1] - hero[1], fleet[0] - hero[0]))
    start, end = -45.0, to_fleet + 180.0
    s = smootherstep((seconds - ISOLATE_S) / REVEAL_S)
    angle = math.radians(start + 4.0 * seconds + s * (end - start - 4.0 * ISOLATE_S))
    radius = 7.0 * (16.0 / 7.0) ** s
    height = 1.4 * (24.0 / 1.4) ** s
    target = hero + smootherstep((seconds - ISOLATE_S - 0.4) / REVEAL_S) * (fleet - hero)
    eye = hero + torch.tensor([radius * math.cos(angle), radius * math.sin(angle), 0.0])
    eye[2] = height
    look_at(scene.viewer, eye, target)


def hero_frames() -> list[Image.Image]:
    env, states = hero_states(ISOLATE_S + REVEAL_S + HOLD_S)
    scene, viewer = make_scene(
        HERO_SIDE**2, 1.0, 1280, 720, sea_opacity=MOTH_SEA_OPACITY, scene_type=OceanScene
    )
    scene.setup(fleet_homes())
    frames = []
    for i, (t, eta) in enumerate(states):
        hero_camera(scene, i / HERO_FPS)
        scene.draw(t, env.sea, env.vessel, eta)
        frames.append(snapshot(viewer, (1280, 720)))
        if i == round(1.5 * HERO_FPS):
            frames[-1].save(OUT / "hero.png", optimize=True)
        time.sleep(0.05)
    viewer.close()
    return frames


def make_hero(width: int = 600) -> None:
    """hero.mp4 at 30 fps and 1280 x 720, hero.gif at 15 fps and `width`. Both end on the
    wide shot and cut straight back to the close-up."""
    frames = hero_frames()
    small = [f.resize((width, width * 9 // 16), Image.Resampling.LANCZOS) for f in frames[::2]]
    save_gif(small, OUT / "hero.gif", round(200 / HERO_FPS))
    if shutil.which("ffmpeg"):
        save_mp4(frames, OUT / "hero.mp4", HERO_FPS)


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
