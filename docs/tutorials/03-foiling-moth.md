# 3. Foiling Moth

## Fly it

```python
import math

import torch

from watergym import SeaState, WaterEnv
from watergym.vessels import moth
from watergym.vessels.moth_vessel import HULL_BOTTOM_Z, WAND_GEARING, wand_action

head_seas = SeaState(hs=0.3, tp=3.0, heading_rad=math.pi, spreading=10.0, num_components=48)
env = WaterEnv(moth(), num_envs=4, sea_state=head_seas)
env.reset(seed=0)

for step in range(500):  # 500 x 0.02 s = 10 s
    action = wand_action(env.sea, env.t, env.eta)
    env.step(action)
    if step % 100 == 99:
        ride = -env.eta[0, 2] - HULL_BOTTOM_Z  # hull bottom above the mean water level
        print(f"t={env.t[0]:.0f}s ride height {ride:.2f} m, speed {env.nu[0, 0]:.1f} m/s")
```

`heading_rad=math.pi` means the waves travel toward the boat, so it flies into them. Without a controller the International Moth cannot stay up, so `wand_action` plays the part of the sailor's wand. It reads the water height under the wand pivot and returns a full action.

## What the action means

```python
print(action.shape)  # [4, 2]
print(action[0])
print(
    env.vessel.num_actions,
    [t for t in env.vessel.thrusters],
    [f.name for f in env.vessel.flapped_foils],
)
```

The action vector is `[thruster commands..., flap commands...]`, one entry per thruster, then one per foil that has a flap. Every entry is in `[-1, 1]`, and `step` clamps it.

- `action[:, 0]` is the sail. The one `Thruster` stands in for the sail: a forward force of `max_force_n` (200 N) times the command. `wand_action` sends 0.65.
- `action[:, 1]` is the main foil flap. It scales `max_flap` (6 degrees), so 1.0 is full trailing-edge down.

The rudder and the struts have no flap, so they take no action.

## Read the foil loads

```python
loads = env.foil_loads()  # one FoilLoads per foil, in vessel.foils order
for foil, load in zip(env.vessel.foils, loads):
    force = load.force[0].sum(0)  # strips summed, env 0, body frame in newtons
    print(
        f"{foil.name:13s} lift up {-force[2]:7.1f} N, forward {force[0]:7.1f} N, "
        f"alpha {math.degrees(load.alpha[0].mean()):5.2f} deg"
    )

print("weight", env.vessel.body.mass * 9.81, "N, ventilated", env.ventilated[0].tolist())
```

`load.force` is `[envs, strips, 3]` in the body frame (x forward, z down), so lift up is minus z. The main foil carries the whole weight (it read 1410 N against a weight of 1128 N here, because the boat was accelerating upward), the rudder foil trims pitch with 43 N of downforce, and drag shows up as negative x. The flow angle `alpha` and the `ventilated` flags come from the [foils](../concepts/05-foils.md) model. A ventilated foil has lost most of its lift.

## Change the sea

Run the same boat in flat water, then in bigger head seas. Env 0 reports.

```python
def fly(hs, seconds=10.0, gain=1.0):
    sea_state = SeaState(hs=hs, tp=3.0, heading_rad=math.pi, spreading=10.0, num_components=48)
    env = WaterEnv(moth(), num_envs=4, sea_state=sea_state)
    env.reset(seed=0)
    heights = []
    while env.t[0] < seconds:
        action = wand_action(env.sea, env.t, env.eta)
        action[:, 1] *= gain
        env.step(action)
        heights.append((-env.eta[:, 2] - HULL_BOTTOM_Z))
    return torch.stack(heights)  # [steps, envs]


for hs in (0.0, 0.3, 0.6):
    h = fly(hs)
    print(f"hs {hs}: ride height min {h.min():.2f} max {h.max():.2f} m over all envs")
```

In flat water the height moves only while the boat settles from its start. Waves make it swing, and the spread grows with `hs`. When `min` reaches 0 the hull touches the water.

## Change the wand gain

The wand turns water height into a flap angle: `WAND_GEARING` radians of flap per radian of wand swing. The `gain` argument above scales that command, so `gain=0` is a dead wand and `gain=0.3` is a weak one:

```python
print(f"flap per wand swing at gain 1: {WAND_GEARING} rad/rad")
for gain in (0.0, 0.3, 1.0, 2.0):
    h = fly(hs=0.0, gain=gain)
    print(f"gain {gain}: ride height after 10 s {h[-1, 0]:.2f} m")

for gain in (1.0, 2.0):
    h = fly(hs=0.3, gain=gain)
    print(f"gain {gain} in hs 0.3: ride height min {h.min():.2f} max {h.max():.2f} m")
```

With no gain the flap stays at zero and the boat sinks onto its hull (-0.02 m). A weak wand holds a lower height, 0.33 m. Gain 1.0 holds 0.54 m and gain 2.0 holds 0.56 m in flat water. In the 0.3 m head seas the stronger wand swings more: gain 2.0 ranges from 0.40 to 0.69 m over the four envs, against 0.42 to 0.64 m at gain 1.0.

## Draw it

```python
from watergym.viewer import WaterViewer, make_viewer

env = WaterEnv(moth(), num_envs=4, sea_state=head_seas)
env.reset(seed=0)
viewer = make_viewer("gl")
scene = WaterViewer(viewer, num_envs=4, patch_size_m=8.0, patch_resolution=48)
scene.look_at_grid(distance=0.9, pitch_deg=-12.0, yaw_deg=70.0)

while viewer.is_running() and env.t[0] < 3.0:
    env.step(wand_action(env.sea, env.t, env.eta))
    scene.draw(env.t, env.sea, env.vessel, env.eta, env.foil_loads(), newton_per_m=300.0)
viewer.close()
```

Passing `env.foil_loads()` to `draw` adds the orange lift and drag arrows. `newton_per_m` sets how many newtons one metre of arrow stands for.

Next: [4. Your own vessel](04-your-own-vessel.md).
