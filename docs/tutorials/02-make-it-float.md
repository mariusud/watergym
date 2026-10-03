# 2. Make it float

You will drop a box barge into flat water, watch it settle, and check the draft against Archimedes.

## Build the environment

```python
import torch

from watergym import SeaState, WaterEnv
from watergym.vessels import box_barge

env = WaterEnv(box_barge(heave_damping_ratio=0.5), num_envs=2, sea_state=SeaState(hs=0.0), dt=0.05)
obs, info = env.reset(seed=0)
print(obs.shape)  # [2, 12]
print(env.eta[0])  # [x, y, z, roll, pitch, yaw]
```

`SeaState(hs=0.0)` is flat water. `reset(seed=0)` returns the observation, which is `eta` and `nu` glued together: 6 pose numbers, then 6 velocities. `env.eta` and `env.nu` hold the same data without the copy. [Frames and state](../concepts/01-frames-and-state.md) explains the axes. The one to remember: NED z points down, so a negative `eta[:, 2]` is above the mean water level.

## Drop it and step

The barge starts at its equilibrium. Lift it clear of the water, then let go:

```python
env.eta[:, 2] = -1.0  # the barge's centre 1 m above the water: bottom at the surface

heave = []
for _ in range(400):  # 400 steps x 0.05 s = 20 s
    env.step(None)  # the barge has no thrusters or flaps
    heave.append(env.eta[0, 2].item())

print([round(z, 3) for z in heave[::40]])  # one sample every 2 s
```

`step` takes an action tensor shaped `[num_envs, num_actions]`. This vessel has no actions, so pass `None`. It returns `(obs, reward, terminated, truncated, info)`, as in Gymnasium; here we ignore them. Each `dt` of 0.05 s runs two RK4 sub-steps ([rigid body](../concepts/03-rigid-body.md)).

The numbers swing around a value and converge. That value is where weight equals buoyancy.

## Compare with the analytic draft

A floating box displaces its own weight in water: `draft = mass / (rho * length * beam)`.

```python
from watergym.vessels.box_barge_vessel import BEAM_M, HEIGHT_M, LENGTH_M, MASS_KG

analytic = MASS_KG / (1025.0 * LENGTH_M * BEAM_M)
simulated = env.eta[0, 2].item() + HEIGHT_M / 2  # centre depth + half height = bottom depth
print(f"analytic draft {analytic:.3f} m, simulated {simulated:.3f} m")
```

The bottom of the box sits `HEIGHT_M / 2` below the centre of gravity, which is the body origin. The two numbers agree to the millimetre once the motion has died out. The hull is cut into cells ([buoyancy](../concepts/04-buoyancy.md)), so a coarse grid can shift the result slightly.

## Read eta and nu

```python
print("eta", [round(v, 3) for v in env.eta[0].tolist()])
print("nu ", [round(v, 3) for v in env.nu[0].tolist()])
```

`eta[2]` is heave position, `nu[2]` heave speed. Everything else is zero because nothing pushes sideways in flat water.

## Add waves and draw

```python
from watergym.viewer import WaterViewer, make_viewer

waves = SeaState(hs=1.0, tp=5.0, heading_rad=1.57)  # heading_rad is in radians: 1.57 is a beam sea
env = WaterEnv(box_barge(), num_envs=2, sea_state=waves, dt=0.05)
env.reset(seed=0)

viewer = make_viewer("gl")
scene = WaterViewer(viewer, num_envs=2, patch_size_m=30.0, patch_resolution=64)
scene.look_at_grid(distance=0.8, pitch_deg=-20.0, yaw_deg=30.0)

while viewer.is_running() and env.t[0] < 3.0:
    env.step(None)
    scene.draw(env.t, env.sea, env.vessel, env.eta)
    viewer.log_scalar("heave env 0 [m]", -env.eta[0, 2])
viewer.close()
```

`scene.draw` gets the clock, the sea, the vessel (for its mesh) and the poses. Now the barge heaves and rolls, because each hull cell feels the passing wave. The two envs have different seas, so their motions differ.

Next: [3. Foiling Moth](03-foiling-moth.md).
