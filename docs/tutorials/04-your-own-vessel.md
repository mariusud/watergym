# 4. Your own vessel

A vessel is a plain dataclass, so a new one needs no registry or base class. This page builds a 20 kg box ROV with two thrusters and holds it at 2 m depth.

## The pieces

A `Vessel` needs three things: a `RigidBody` (mass and damping), `VolumeSamples` for the hull (buoyancy) and a `Mesh` (drawing). Thrusters and foils are optional.

```python
import torch

from watergym import RigidBody, Thruster, Vessel, box_mesh, box_samples
from watergym.geometry import VolumeSamples

SIZE = (0.5, 0.4, 0.3)  # length, beam, height in metres
MASS_KG = 20.0
DISPLACED_M3 = 0.0205  # a little more than 20 kg of water: slightly buoyant


def box_rov() -> Vessel:
    body = RigidBody.from_diagonals(
        MASS_KG,
        inertia=[0.4, 0.5, 0.6],  # Ixx, Iyy, Izz in kg m^2
        added_mass=[4.0, 8.0, 14.0, 0.2, 0.3, 0.3],
        linear_damping=[10.0, 10.0, 20.0, 1.0, 1.0, 1.0],
        quadratic_damping=[60.0, 100.0, 120.0, 1.0, 1.0, 1.0],
    )
    cells = box_samples((0.0, 0.0, -0.04), SIZE, cells=(4, 4, 4))
    hull = VolumeSamples(
        cells.centers, cells.volumes * DISPLACED_M3 / cells.volumes.sum(), cells.heights
    )
    return Vessel(
        name="box_rov",
        body=body,
        hull=hull,
        mesh=box_mesh((0.0, 0.0, 0.0), SIZE),
        thrusters=[
            Thruster(position=(0.0, 0.0, 0.0), direction=(1.0, 0.0, 0.0), max_force_n=40.0),
            Thruster(position=(0.0, 0.0, 0.0), direction=(0.0, 0.0, 1.0), max_force_n=40.0),
        ],
        density=1000.0,
        initial_eta=(0.0, 0.0, 2.0, 0.0, 0.0, 0.0),  # 2 m below the surface
    )
```

What each part means:

- `RigidBody.from_diagonals` builds the diagonal mass, inertia and added-mass matrices. Vectors are ordered surge, sway, heave, roll, pitch, yaw ([rigid body](../concepts/03-rigid-body.md)). Added mass is the water the body drags along when it accelerates.
- `box_samples(center, size, cells)` cuts a box into cells. Each cell is a buoyancy sample. Here the cell volumes are rescaled so they add up to the real displaced volume, because a solid 0.5 x 0.4 x 0.3 m box would displace 60 litres, and an open-frame ROV displaces far less ([buoyancy](../concepts/04-buoyancy.md)).
- The centre of buoyancy sits 4 cm above the centre of gravity (`z = -0.04`, up is negative).
- A body with more added mass in heave than in surge pitches up as it moves forward (the Munk moment). With the centre of buoyancy at the centre of gravity, this ROV tumbled at 0.5 m/s. Raising the centre of buoyancy gives a righting moment.
- `box_mesh` makes the triangles that the viewer draws. It plays no part in the physics.
- A `Thruster` pushes along `direction` (body axes, x forward, z down) with `command * max_force_n`. `position` is relative to the centre of gravity and produces torque.
- Thruster order sets the action order: here `action[:, 0]` is surge and `action[:, 1]` is heave, positive down.
- `initial_eta` is where a reset puts the vessel. Negative z is above the surface.

## Step it

```python
from watergym import SeaState, WaterEnv

env = WaterEnv(box_rov(), num_envs=4, sea_state=SeaState(hs=0.3, tp=4.0), dt=0.05)
env.reset(seed=0)
print(env.vessel.num_actions)  # 2

action = torch.zeros(4, 2)
action[:, 0] = 0.5  # half surge thrust
for step in range(200):  # 10 s
    env.step(action)
    if step % 40 == 39:
        print(
            f"t={env.t[0]:.0f}s x {env.eta[0, 0]:5.2f} m, depth {env.eta[0, 2]:.2f} m, surge {env.nu[0, 0]:.2f} m/s"
        )
```

With a slightly buoyant hull and no heave thrust, the ROV drifts upward (depth falls from 2 m to 1.14 m in 10 s) while it moves forward. Thrust balances quadratic drag, so surge speed levels off near 0.5 m/s. Add a depth hold as in `examples/04_underwater_vehicle.py`:

```python
env.reset(seed=0)
target_depth = 2.0
for step in range(200):
    error = target_depth - env.eta[:, 2]
    heave = (2.0 * error - 1.0 * env.nu[:, 2]).clamp(-1, 1)
    env.step(torch.stack((torch.full((4,), 0.5), heave), dim=-1))
print("depth after 10 s", env.eta[:, 2].tolist())
```

The four envs end between 1.94 and 1.95 m, 5 to 6 cm short of 2 m. The net buoyancy needs a steady downward thrust, and this PD controller has no integral term to supply it without an error. The controller reads `env.eta` and `env.nu` directly. A sign check: depth too small means `error > 0`, so `heave > 0`, which pushes down.

## Draw it

```python
from watergym.viewer import WaterViewer, make_viewer

env.reset(seed=0)
viewer = make_viewer("gl")
scene = WaterViewer(viewer, num_envs=4, patch_size_m=6.0, patch_resolution=48)
scene.look_at_grid(distance=1.0, pitch_deg=-28.0, yaw_deg=45.0)
while viewer.is_running() and env.t[0] < 3.0:
    error = target_depth - env.eta[:, 2]
    heave = (2.0 * error - 1.0 * env.nu[:, 2]).clamp(-1, 1)
    env.step(torch.stack((torch.full((4,), 0.5), heave), dim=-1))
    scene.draw(env.t, env.sea, env.vessel, env.eta)
viewer.close()
```

The mesh follows `vessel.name`, so give each vessel type a unique name.

## Try this

- Make a vertical thruster pair at `y = ±0.15` and watch roll appear when you drive them unevenly.
- Add a `Foil` to `foils=[...]` (see `src/watergym/vessels/moth_vessel.py`) for a vessel that flies. Foil flaps add entries after the thrusters in the action vector.
- Set `free_dofs=(True, False, True, False, True, False)` to lock sway, roll and yaw to zero, as the International Moth does.

Next: [5. Batching and devices](05-batching-and-devices.md).
