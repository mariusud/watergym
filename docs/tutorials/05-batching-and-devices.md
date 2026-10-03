# 5. Batching and devices

The same `WaterEnv` runs 1 env or 64; only the leading dimension of each tensor changes.

## One tensor per quantity

```python
import math
import time

import torch

from watergym import SeaState, WaterEnv
from watergym.vessels import box_barge

waves = SeaState(hs=1.0, tp=5.0, heading_rad=math.pi / 2)


def run(num_envs, steps=20, device="cpu"):
    env = WaterEnv(box_barge(), num_envs, waves, dt=0.05, device=device)
    env.reset(seed=0)
    start = time.perf_counter()
    for _ in range(steps):
        env.step(None)
    return env, time.perf_counter() - start


for n in (1, 8, 64):
    env, seconds = run(n)
    print(
        f"{n:5d} envs: {seconds:5.2f} s for 20 steps, eta {tuple(env.eta.shape)}, sea {tuple(env.sea.phase.shape)}"
    )
```

Every state has a leading `num_envs` dimension: `eta` and `nu` are `[num_envs, 6]`, and the sea tensors are `[num_envs, num_components]`. The loop is over time only. On CPU the time grows with the batch, but slower than the env count: in one run on a laptop CPU, 64 envs took 3.74 s against 0.17 s for one env, 22 times as long. The batch layout pays off on a GPU, where one kernel covers all envs ([batching](../concepts/06-batching.md)).

## Seeds

```python
env_a, _ = run(4)
env_b, _ = run(4)
print("same seed, same result:", torch.equal(env_a.eta, env_b.eta))

env_c = WaterEnv(box_barge(), 4, waves, dt=0.05)
env_c.reset(seed=1)
env_c.step(None)
print("other seed differs:", not torch.equal(env_a.eta[:, 2], env_c.eta[:, 2]))

print("envs differ from each other:", env_a.eta[:, 2].tolist())
```

`reset(seed=n)` seeds the generator that draws every sea. The same seed rebuilds the same seas, env by env, so a run repeats exactly. Within one reset the envs get different phases, so their motions differ. `reset()` with no seed keeps drawing from the running generator. Envs that finish an episode inside `step()` also draw a fresh sea from it.

## Devices

```python
def best_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


device = best_device()
env, seconds = run(64, steps=20, device=device)
print(device, f"{seconds:.2f} s", env.eta.device)

cpu_env, _ = run(64, steps=20, device="cpu")
print("max difference to cpu [m]:", (env.eta.cpu() - cpu_env.eta).abs().max().item())
```

Pass `device="cpu"`, `"mps"` or `"cuda"` to `WaterEnv`. The vessel, the sea and all state move to that device, and so must your actions: build them with `device=env.device`. The same code gives the same physics on every device up to float32 rounding, but chaotic motion can amplify the rounding over long runs, so compare short runs. [Batching](../concepts/06-batching.md#devices) covers when CPU beats GPU and how to pick `num_envs`.

`examples/02_floating_box.py`, `examples/03_moth_on_foils.py` and `examples/train_rsl_rl.py` take `--device mps` or `--device cuda`.

## Draw a few envs out of many

```python
from watergym.viewer import WaterViewer, make_viewer

env = WaterEnv(box_barge(), 64, waves, dt=0.05)
env.reset(seed=0)

shown = 4
viewer = make_viewer("gl")
scene = WaterViewer(viewer, num_envs=shown, patch_size_m=30.0, patch_resolution=64)
scene.look_at_grid(distance=0.8, pitch_deg=-20.0, yaw_deg=30.0)

while viewer.is_running() and env.t[0] < 3.0:
    env.step(None)
    scene.draw(env.t[:shown], env.sea.subset(slice(0, shown)), env.vessel, env.eta[:shown])
viewer.close()
```

`WaterViewer` builds a mesh patch per env and copies the data to the host every frame, so its cost grows with the number of envs drawn. With the `"null"` viewer on a laptop CPU, one frame took 3.7 ms for 4 drawn envs and 48 ms for 64, against 183 ms for one physics step of 64 envs. A real window adds rendering on top, and with the physics on a GPU, drawing can cost more than the physics. The viewer's `num_envs` must equal the length of what you pass to `draw`, so slice the time, the sea and the poses to the first `shown` envs. `Sea.subset` takes a slice, a list of env ids or a tensor of ids.

Back to the [index](README.md).
