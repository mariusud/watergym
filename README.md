# WaterGym

[![CI](https://github.com/mariusud/watergym/actions/workflows/ci.yml/badge.svg)](https://github.com/mariusud/watergym/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

A batched PyTorch gym for vessels in waves: hydrofoils, boats, USVs (uncrewed surface vessels) and ROVs (remotely operated underwater vehicles), on CPU, Apple MPS or CUDA. It is legged_gym for vessels: thousands of vessels stepped in parallel, each in its own random sea, with a task, a reward and PPO on top.

You need no marine background. The terms in the code are defined below, and [docs/concepts](docs/concepts/README.md) explains the rest.

<p align="center">
  <img src="docs/images/readme/hero.gif" alt="A grid of International Moths foiling through head seas, each in its own random sea" width="100%">
</p>

## Quickstart

```bash
git clone https://github.com/mariusud/watergym && cd watergym
uv run --extra viz examples/03_moth_on_foils.py
```

The first run downloads torch, warp and newton, a few hundred MB. Then a window opens with four International Moths (foiling dinghies that ride on underwater wings) flying into head seas, waves that come at the bow, with a significant wave height Hs of 0.3 m. Without a display, add `--viewer null` to run the physics without drawing. It prints a one-line summary when it ends, for example `5.0 sim s, mean hull bottom above mean water 0.53 m, 545 env-steps/s` with `--seconds 5`. `--headless --screenshot out.png` saves a frame without a window but still uses OpenGL. Without the `viz` extra, `uv sync` installs only numpy and torch.

[Open in Colab](https://colab.research.google.com/github/mariusud/watergym/blob/main/notebooks/quickstart.ipynb) with no install. The notebook is [notebooks/quickstart.ipynb](notebooks/quickstart.ipynb).

## Using it

The code below is [examples/00_hello.py](examples/00_hello.py). It needs no window and no extras:

```python
import math

import torch

from watergym import SeaState, WaterEnv
from watergym.vessels import moth
from watergym.vessels.moth_vessel import wand_action


def ride_height(env: WaterEnv) -> torch.Tensor:
    """1 at the starting ride height, 0.37 at 10 cm off. NED z points down."""
    error = env.eta[:, 2] - env.vessel.initial_eta[2]
    return torch.exp(-((error / 0.1) ** 2))


head_seas = SeaState(hs=0.3, tp=3.0, heading_rad=math.pi)  # 0.3 m waves on the bow
env = WaterEnv(moth(), num_envs=64, sea_state=head_seas, reward_fn=ride_height)
obs, info = env.reset(seed=0)

for _ in range(250):  # 5 s at dt = 0.02 s
    action = wand_action(env.sea, env.t, env.eta)  # [64, 2]: sail, flap. Your policy goes here.
    obs, reward, terminated, truncated, info = env.step(action)

print(obs.shape, f"mean reward {reward.mean():.2f}")
```

```bash
uv run examples/00_hello.py
```

Expected output:

```
torch.Size([64, 12]) mean reward 0.61
```

Every tensor is shaped `[num_envs, ...]`, and each env draws its own sea. Envs that capsize or time out reset inside `step()`, and their last observation is in `info["final_obs"]`. Terms in the code:

| Term | Meaning |
|---|---|
| Hs, Tp | Significant wave height, the mean of the highest third of waves, and peak period, the period of the most energetic waves ([waves](docs/concepts/02-waves.md)). |
| Head and beam seas | Head seas come at the bow, and `heading_rad=math.pi` sets that. Beam seas hit the side (`math.pi / 2`) ([waves](docs/concepts/02-waves.md)). |
| NED, `eta`, `nu` | North-east-down axes, so z points down and a hull 0.6 m above the water has z = -0.6. `eta` is position plus roll, pitch and yaw in the world frame. `nu` is velocity in the body frame. `obs` is `[eta, nu]` ([frames](docs/concepts/01-frames-and-state.md)). |
| Wand | A rod that trails on the water ahead of the Moth's bow. A low hull pushes it back, which raises the flap and the lift. `wand_action` is that mechanical ride-height controller, standing in for a policy here ([foils](docs/concepts/05-foils.md)). |
| Ventilation | Air drawn down onto a foil. Its lift collapses until the angle of attack falls back and the flow reattaches ([foils](docs/concepts/05-foils.md)). |

A vessel is a dataclass: a `RigidBody` (mass, inertia, added mass, damping), hull volume samples for buoyancy, a mesh for drawing, and optional `Foil`s and `Thruster`s. The four in `watergym/vessels/` are written out in full, so copy one to start a new vessel.

## Why it exists

Vessels in waves have no equivalent. [MarineGym](https://github.com/Marine-RL/MarineGym) ([arXiv:2503.09203](https://arxiv.org/abs/2503.09203)) batches underwater vehicles on the GPU but has no free surface. [VRX](https://github.com/osrf/vrx) and [Stonefish](https://github.com/patrykcieslak/stonefish) model waves but are built around one CPU simulation at a time.

In WaterGym each vessel gets its own irregular sea, the physics is plain PyTorch tensors, the API follows Gymnasium's vector envs, and an adapter plugs it into rsl_rl. `OMP_NUM_THREADS=2 uv run benchmarks/speed.py` times 1024 Moths flown by the wand: on 2 CPU threads of an Apple M4 Pro it measured 7,200 to 7,820 env-steps/s (four runs, with `nice -n 19` and other work on the machine). With dt = 0.02 s the real-time factor, env-steps/s × dt, is 144 to 156.

It is for RL researchers who want a disturbance-rejection benchmark beyond terrain, and for marine control engineers who want to test learned controllers against PID and the wand on the same seas. Version 0.1 has one benchmark task and the simplifications listed below.

## Vessels

<p align="center">
  <img src="docs/images/readme/vessels.png" alt="The four WaterGym vessels in waves: box barge, Otter USV, International Moth and BlueROV2" width="100%">
</p>

| Vessel | Function | Actions | Where the numbers come from |
|---|---|---|---|
| 10 x 4 x 2 m box barge | `box_barge()` | none | analytic hydrostatics (Newman, *Marine Hydrodynamics*), the reference case |
| Otter USV, 2 m catamaran | `otter()` | 2 thrusters | Fossen's [PythonVehicleSimulator](https://github.com/cybergalactic/PythonVehicleSimulator) |
| International Moth | `moth()` | sail, main-foil flap | Moth class rules and Day et al. 2019 T-foil tank tests; sailor mass and wand geometry are estimates |
| BlueROV2 Heavy | `bluerov2()` | 8 thrusters | von Benzon et al. 2022, *JMSE* 10:1898, Table A1 |

All four import from `watergym.vessels`, one file each in `src/watergym/vessels/`.

## Sea state is the benchmark axis

<p align="center">
  <img src="docs/images/readme/sea_states.png" alt="The same Moth at significant wave heights of 0, 0.3 and 0.6 m" width="100%">
</p>

Sea state sets the difficulty, as terrain does in legged RL. Every task runs over a grid of Hs, Tp and heading, and a score is a set of curves against sea state.

The first task is `RideControl-Moth-v0`. A policy moves the main-foil flap to hold the hull 0.6 m above the mean water level. It sees only what a real foiler can measure: bow height, pitch, pitch rate, heave acceleration, speed and its last command. The quick sweep runs in about 30 s on a laptop CPU. It prints one line of metrics per sea state and policy, and writes [benchmarks/results/ride_control.csv](benchmarks/results/ride_control.csv) and a plot, `docs/images/benchmark_ride_control.png`:

```bash
OMP_NUM_THREADS=2 uv run --with matplotlib python benchmarks/ride_control_sweep.py --quick
```

Baselines from that sweep, head seas, Tp 3 s:

| Hs [m] | wand tracking RMS [m] | wand a_w [m/s²] | zero-flap tracking RMS [m] | zero-flap crashes / min |
|---|---|---|---|---|
| 0.0 | 0.069 | 0.04 | 0.27 | 18 |
| 0.2 | 0.073 | 0.52 | 0.26 | 15 |
| 0.4 | 0.084 | 1.07 | 0.25 | 12 |
| 0.6 | 0.103 | 1.70 | 0.21 | 12 |

The wand never crashes on this grid, but it follows the waves, so the ISO 2631 comfort figure a_w climbs past 1 m/s² by Hs 0.4 m. Pass your own policy as `--policy wand mypkg.policies:actor`. The task is frozen: any change to its reward, observations or seeds makes a new id. Observations, reward, metrics and the full grid are in [docs/benchmark.md](docs/benchmark.md).

## Training

Train a policy with [rsl_rl](https://github.com/leggedrobotics/rsl_rl)'s PPO, then watch it:

```bash
uv run --extra rl examples/train_rsl_rl.py --num-envs 1024 --iterations 300
uv run --extra rl --extra viz examples/train_rsl_rl.py --play logs/rsl_rl/moth/<timestamp>/model_299.pt
```

The example trains the Moth to hold its starting ride height and writes checkpoints and TensorBoard logs to `logs/rsl_rl/moth/<timestamp>/`. `model_299.pt` exists only after the 300 iterations finish; the script also saves a checkpoint every 50 iterations, so `model_50.pt` appears earlier. Add `--device mps` or `--device cuda` to run more envs.

To check a checkpoint without a window, add `--viewer null`: it runs `--seconds 20` (the default) of sim time and prints the mean episode reward, the ride-height RMS in metres and the number of crashes. `RslRlVecEnv(env)` wraps any `WaterEnv`; [docs/training.md](docs/training.md) shows how `terminated` and `truncated` map to rsl_rl's `dones` and `time_outs`.

## Examples and the viewer

| | |
|---|---|
| ![sea](docs/images/01_sea_state.png) `01_sea_state.py`: four JONSWAP seas with Hs 1.5 m and Tp 4.5 s, each with its own random phases | ![barge](docs/images/02_floating_box.png) `02_floating_box.py`: 10 m box barges heaving and rolling in beam seas |
| ![moth](docs/images/03_moth_on_foils.png) `03_moth_on_foils.py`: International Moths foiling through head seas, flap set by the mechanical wand, lift and drag arrows per foil strip | ![rov](docs/images/04_underwater_vehicle.png) `04_underwater_vehicle.py`: BlueROV2s holding 0.5, 1, 2 and 4 m depth under waves |

Drawing uses [Newton](https://github.com/newton-physics/newton)'s viewer from the `viz` extra: one grid cell per env, the sea patch follows its vessel, and orange arrows show the force on every foil strip. `--viewer gl` opens a window, `viser` serves a browser view, `usd` records a file and `null` draws nothing. Examples 01 to 04 take `--headless --screenshot out.png`, which saves a frame without a window but still renders with OpenGL, so it needs a working GPU driver. `--device mps` or `--device cuda` works on 02, 03 and `train_rsl_rl.py`, the scripts with a `--device` flag.

## Learning path

[docs/tutorials](docs/tutorials/README.md) has five short hands-on pages, each building on the last:

1. [First sea](docs/tutorials/01-first-sea.md): make a sea, sample the surface, draw it.
2. [Make it float](docs/tutorials/02-make-it-float.md): drop a barge and check its draft against Archimedes.
3. [Foiling Moth](docs/tutorials/03-foiling-moth.md): fly with the wand, read the foil loads, then break the wand.
4. [Your own vessel](docs/tutorials/04-your-own-vessel.md): define a box ROV from scratch.
5. [Batching and devices](docs/tutorials/05-batching-and-devices.md): `num_envs`, seeds, CPU, MPS and CUDA.

[docs/concepts](docs/concepts/README.md) covers frames, waves, rigid-body equations, buoyancy, foils and batching. Each page starts with an example you can check by hand.

## How it fits together

<p align="center">
  <img src="docs/images/readme/stack.svg" alt="Layers: waves feed hydrostatics and foils, which feed the rigid body, which WaterEnv batches, which rsl_rl trains" width="720">
</p>

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
  tasks/            versioned benchmark tasks and metrics, starting with RideControl-Moth-v0
  viewer.py         draws seas, vessels and force arrows with Newton's viewer
  rl/               rsl_rl adapter (extra: rl)
examples/           numbered scripts, each runs on its own
benchmarks/         the sea-state sweep and its results
notebooks/          Colab quickstart
docs/               tutorials, concepts, training, benchmark
tests/              physics checks, one file per module
```

## Physics in one paragraph

Fossen's equation (M_RB + M_A) nu_dot + C(nu) nu + D(nu) nu = tau, integrated with RK4, with eta and nu in NED as in Fossen's handbook. The sea is a sum of Airy components drawn from a JONSWAP spectrum. Each submerged volume sample feels rho V (a_water - g), which gives buoyancy, restoring moments and Froude-Krylov wave forces in one term. Foils are cut into spanwise strips that see the water velocity relative to the strip, waves included.

## What the tests check

- Hs from the generated components and from a 30 min surface record; Airy orbital velocity; surface rise rate equals vertical orbital velocity.
- Free fall, RK4 fourth-order convergence, Coriolis forces doing no work.
- Box barge: floats at the analytic draft, heave stiffness rho g L B, roll moment from GM, heave natural period within 1 %, follows long waves.
- Foils: 2 pi alpha in deep water, Helmbold slope, the free-surface table, lift falling toward the surface, ventilation hysteresis.
- The ISO 2631 Wk weighting against the standard's table, and the v0 task definition staying frozen.
- Env API shapes, seeding, auto-reset, and the wand keeping a Moth flying.

Reference values come from DNV-RP-C205 (JONSWAP), linear wave theory as in Faltinsen's *Sea Loads on Ships and Offshore Structures*, textbook box hydrostatics (Newman, *Marine Hydrodynamics*), thin-airfoil and Helmbold lift (Abbott and von Doenhoff) and ISO 2631-1:1997. Run them with `uv run pytest`.

## Simplifications to know about

- Euler angles, singular at 90 degrees pitch, which no vessel here reaches.
- Constant added mass and linear plus quadratic damping per vessel. No radiation memory, diffraction or second-order drift yet.
- Deep-water waves only. Above the mean surface the orbital velocity is held at its mean-level value.
- Damping acts on the body velocity, not the velocity relative to the waves.
- Hulls are boxes cut into vertical cells. Fewer cells across the beam lowers the waterplane inertia and with it GM, so the barge uses 16 cells.
- The Moth flies in the vertical plane only (surge, heave, pitch): the sailor's roll balance is not modelled, the sail is a constant forward force at the CG, and the wand reads the water height under its pivot.
- Foils use thin-airfoil lift with a stall clamp and a flap-effectiveness factor, not a measured polar. Ventilation thresholds are placeholders; the hysteresis is modelled.
- The BlueROV2 has four horizontal and four vertical thrusters in a simplified layout.
- No actuator lag, sensor noise or wind yet, so no policy trained here has been tested on a real vessel.
- Ventilation washes out once every wet strip has stayed below the washout angle (4 degrees by default) for 0.5 s; a horizontal foil must also be more than 1 chord deep. A surface-piercing strut washes out on angle alone and has no free-surface lift factor, so it loses lift near the surface only through its dry strips. Any change to these rules comes with a test that pins the new behaviour.

## Roadmap

The next goal is one policy, trained in WaterGym, controlling a real vessel in real waves, with logs and a protocol others can repeat. In order:

1. Actuator lag, rate limits and per-episode latency randomization, then IMU, compass and GNSS noise models.
2. An ArduPilot SITL bridge, so WaterGym can be the physics behind the autopilot that BlueBoat and BlueROV2 already run.
3. System identification from logged runs, writing a model card per vessel.
4. A BlueBoat trial across a sea-state ladder from calm to Hs 0.3 m, against tuned ArduRover, PID and PPO. An RC-scale foiler is the stretch goal.

More tasks follow the same sea-state grid: StationKeep and PathFollow for USVs, DepthHold under waves for the BlueROV2, and roll damping with active fins. On the physics side: radiation memory, diffraction and drift forces, mesh hulls in place of boxes, and a mesh to hydrodynamic-coefficients compiler.

## Borrowed from

- [legged_gym](https://github.com/leggedrobotics/legged_gym) and [rsl_rl](https://github.com/leggedrobotics/rsl_rl): the pattern of a batched env, a task with a reward and PPO on top, and the idea of a graded disturbance as the difficulty axis.
- [mjlab](https://github.com/mujocolab/mjlab): `src/` layout, uv for everything, a README that leads with commands you can run.
- [MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground): one file per model, grouped in one folder, and a picture per example.
- [CleanRL](https://github.com/vwxyzjn/cleanrl): examples are single scripts you read top to bottom, with argparse flags and no framework.
- [Gymnasium](https://github.com/Farama-Foundation/Gymnasium): `reset(seed) -> (obs, info)`, the five-value `step`, vector-env style, and versioned task ids.
- [PythonVehicleSimulator](https://github.com/cybergalactic/PythonVehicleSimulator): NED frames, `eta`/`nu` notation, `Rzyx`, `Tzyx` and `m2c`, and the Otter coefficients.

## How to cite

Not yet published. Placeholder:

```bibtex
@software{watergym,
  title = {WaterGym: a batched PyTorch gym for vessels in waves},
  author = {TODO},
  year = {2026},
  url = {https://github.com/mariusud/watergym}
}
```

## License

MIT, see [LICENSE](LICENSE).
