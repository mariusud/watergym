# 1. First sea

This page builds four random seas with the same 1.5 m significant wave height, samples the surface, and draws the four side by side.

## Make a sea

```python
import torch

from watergym.waves import elevation, make_sea, significant_wave_height

sea = make_sea(num_envs=4, hs=1.5, tp=4.5, spreading=8.0, num_components=96)
print(significant_wave_height(sea))  # tensor([1.4995, 1.4995, 1.4995, 1.4995])
```

`make_sea` returns a `Sea`: a bundle of tensors, each shaped `[num_envs, num_components]` (amplitude, frequency, wavenumber, direction, phase). Env 0 and env 1 share `hs` and `tp` but get different random phases, so their waves differ. The realised Hs is the same in every env, 1.4995 m, because the spectrum is cut into 96 bins. Only the phases, frequencies inside each bin and directions are random. More on this in [waves](../concepts/02-waves.md).

## Sample the surface

```python
points = torch.tensor([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [0.0, 5.0, 0.0]])  # x, y, z in NED
points = points.expand(4, -1, -1)  # the same 3 points in every env: [4, 3, 3]
t = torch.zeros(4)  # one clock per env

height = elevation(sea, points, t)  # [4, 3], metres, positive up
print(height)
```

Points are `[num_envs, points, 3]` in the NED world frame ([frames](../concepts/01-frames-and-state.md)). Only x and y matter for `elevation`. `t` is one time per env. The result is the surface height, positive up, so a crest reads positive even though NED z points down.

Step time and the surface moves:

```python
for step in range(3):
    print(f"t={step * 0.5:.1f}s", elevation(sea, points, t + step * 0.5)[0].tolist())
```

## Draw it

```python
from watergym.viewer import WaterViewer, make_viewer

viewer = make_viewer(
    "gl"
)  # a window. "null" draws nothing, "gl" with headless=True renders offscreen
scene = WaterViewer(viewer, num_envs=4, patch_size_m=40.0, patch_resolution=128)
scene.look_at_grid(distance=0.7, pitch_deg=-22.0)

dt = 1 / 30
while viewer.is_running() and t[0] < 3.0:  # raise 3.0 to watch longer
    scene.draw(t, sea)
    viewer.log_scalar("elevation at origin, env 0 [m]", elevation(sea, points, t)[0, 0])
    t += dt
viewer.close()
```

`WaterViewer` lays the envs out on a grid, one square sea patch of `patch_size_m` metres per env, sampled on a `patch_resolution` x `patch_resolution` mesh. `scene.draw(t, sea)` takes the clock and the sea and pushes one frame. `log_scalar` adds a live plot to the viewer window.

To save a picture, create the viewer with `make_viewer("gl", headless=True)` and call `scene.save_png("sea.png")` after the loop.

Next: [2. Make it float](02-make-it-float.md).
