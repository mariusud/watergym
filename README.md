# WaterGym

A batched PyTorch gym for vessels in waves: hydrofoils, boats, USVs and underwater vehicles, on CPU, Apple MPS or CUDA.

![A fleet of Moths foiling through head seas](docs/images/03_moth_on_foils.png)

## Quickstart

```bash
git clone https://github.com/YOUR-ORG/watergym && cd watergym  # placeholder URL
uv sync --extra viz
uv run --extra viz examples/03_moth_on_foils.py
```

No window? Add `--headless --screenshot out.png`. No GPU or viewer? `uv sync` alone installs the physics, and `--viewer null` runs without drawing.

[Try it in Colab](https://colab.research.google.com/github/YOUR-ORG/watergym/blob/main/notebooks/quickstart.ipynb) (placeholder link). The notebook is [notebooks/quickstart.ipynb](notebooks/quickstart.ipynb).

## Examples

| | |
|---|---|
| ![sea](docs/images/01_sea_state.png) `01_sea_state.py`: four JONSWAP seas with Hs 1.5 m and Tp 4.5 s, each with its own random phases | ![barge](docs/images/02_floating_box.png) `02_floating_box.py`: 10 m box barges heaving and rolling in beam seas |
| ![moth](docs/images/03_moth_on_foils.png) `03_moth_on_foils.py`: International Moths foiling through head seas, flap set by the mechanical wand, lift and drag arrows per foil strip | ![rov](docs/images/04_underwater_vehicle.png) `04_underwater_vehicle.py`: BlueROV2s holding 0.5, 1, 2 and 4 m depth under waves |

Every example takes `--headless --screenshot out.png` to render offscreen, `--viewer null` to run without drawing, and `--device mps` or `--device cuda` where it has a `--device` flag.

## Using it

```python
import torch
from watergym import SeaState, WaterEnv
from watergym.vessels import otter

env = WaterEnv(otter(), num_envs=64, sea_state=SeaState(hs=1.0, tp=5.0), device="cpu")
obs, info = env.reset(seed=0)
for _ in range(500):
    action = torch.full((64, 2), 0.5)  # both thrusters at half power
    obs, reward, terminated, truncated, info = env.step(action)
```

`obs` is `[eta, nu]`, the NED pose and body velocity. Each env draws its own sea. Envs that capsize or time out reset inside `step()`, and their last observation is in `info["final_obs"]`. Pass `reward_fn=lambda env: ...` to score a task.

A vessel is a dataclass: a `RigidBody` (mass, inertia, added mass, damping), hull volume samples for buoyancy, a mesh for drawing, and optional `Foil`s and `Thruster`s. The four in `watergym/vessels/` are written out in full, so copy one to start a new vessel.

## Tutorials

Five short hands-on pages in [docs/tutorials](docs/tutorials/README.md), from drawing a first sea to defining your own vessel.

## Concepts

Frames, waves, rigid-body equations, buoyancy, foils and batching, in [docs/concepts](docs/concepts/README.md).

## Training

Train a policy with rsl_rl: see [docs/training.md](docs/training.md). Install the extra with `uv sync --extra rl`.

## Benchmark

Tasks, metrics and baselines are in [docs/benchmark.md](docs/benchmark.md).

## Layout

```
src/watergym/
  waves.py          JONSWAP spectrum, directional seas, surface elevation, orbital velocity and acceleration
  rigid_body.py     Fossen 6-DOF equations: kinematics, Coriolis, damping, RK4
  geometry.py       boxes as volume samples (buoyancy) and as meshes (drawing)
  hydrostatics.py   buoyancy plus Froude-Krylov force from samples against the instantaneous surface
  foils.py          strip-theory lift and drag, free-surface lift loss, ventilation with hysteresis
  vessel.py         Vessel and Thruster, and the sum of forces that drives the equations of motion
  vessels/          box_barge, moth, otter, bluerov2
  env.py            WaterEnv: batched reset/step in the Gymnasium vector style
  viewer.py         draws seas, vessels and force arrows with Newton's viewer
  rl/               rsl_rl adapter (extra: rl)
examples/           numbered scripts, each runs on its own
notebooks/          Colab quickstart
docs/               tutorials, concepts, training, benchmark
tests/              physics checks, one file per module
research/           the design plan and background research
```

## How to cite

Not yet published. Placeholder:

```bibtex
@software{watergym,
  title = {WaterGym: a batched PyTorch gym for vessels in waves},
  author = {TODO},
  year = {2026},
  url = {https://github.com/YOUR-ORG/watergym}
}
```

## Physics in one paragraph

Fossen's equation (M_RB + M_A) nu_dot + C(nu) nu + D(nu) nu = tau, integrated with RK4, with eta and nu in NED as in Fossen's handbook. The sea is a sum of Airy components drawn from a JONSWAP spectrum. Each submerged volume sample feels rho V (a_water - g), which gives buoyancy, restoring moments and Froude-Krylov wave forces in one term. Foils are cut into spanwise strips that see the water velocity relative to the strip, waves included. Section 2 of research/00-plan.md explains every term for readers new to marine hydrodynamics.

## What the tests check

- Hs from the generated components and from a 30 min surface record; Airy orbital velocity; surface rise rate equals vertical orbital velocity.
- Free fall, RK4 fourth-order convergence, Coriolis forces doing no work.
- Box barge: floats at the analytic draft, heave stiffness rho g L B, roll moment from GM, heave natural period within 1 %, follows long waves.
- Foils: 2 pi alpha in deep water, Helmbold slope, the free-surface table, lift falling toward the surface, ventilation hysteresis.
- Env API shapes, seeding, auto-reset, and the wand keeping a Moth flying.

Reference values come from research/11-vessel-parameters.md and research/12-validation-references.md.

## Borrowed from

- [mjlab](https://github.com/mujocolab/mjlab): `src/` layout, uv for everything, a README that leads with commands you can run.
- [MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground): one file per model, grouped in one folder, and a picture per example.
- [CleanRL](https://github.com/vwxyzjn/cleanrl): examples are single scripts you read top to bottom, with argparse flags and no framework.
- [Gymnasium](https://github.com/Farama-Foundation/Gymnasium): `reset(seed) -> (obs, info)` and the five-value `step`, vector-env style.
- [PythonVehicleSimulator](https://github.com/cybergalactic/PythonVehicleSimulator): NED frames, `eta`/`nu` notation, `Rzyx`, `Tzyx` and `m2c`, and the Otter coefficients.

## Simplifications to know about

- Euler angles, singular at 90 degrees pitch. Vessels never get there.
- Constant added mass and linear plus quadratic damping per vessel. No radiation memory, diffraction or second-order drift yet (plan WP7/WP8).
- Deep-water waves only. Above the mean surface the orbital velocity is held at its mean-level value.
- Damping acts on the body velocity, not the velocity relative to the waves.
- Hulls are boxes cut into vertical cells. Fewer cells across the beam lowers the waterplane inertia and with it GM, so the barge uses 16.
- The Moth flies in the vertical plane only (surge, heave, pitch): the sailor's roll balance is not modelled, the sail is a constant forward force at the CG, and the wand reads the water height under its pivot.
- Foils use thin-airfoil lift with a stall clamp and a flap-effectiveness factor, not a measured polar. Ventilation thresholds are placeholders; the hysteresis is real, the numbers are not.
- The BlueROV2 has four horizontal and four vertical thrusters in a simplified layout.

## Research

The design plan and the background notes behind it are in [research/](research/00-plan.md). They cover simulators, hydrofoil control, sim-to-real methodology and vessel parameters. The repo implements the `watergym-lite` core of that plan (WP11).
