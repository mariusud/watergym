# Batching

**Example.** Wave height at 3 hull points for 4096 environments, no loop.

    num_envs, n_points = 4096, 3
    pos = torch.zeros(num_envs, n_points, 2)      # world x, y
    t = torch.zeros(num_envs, 1)
    eta = wave_height(pos, t)                      # -> [4096, 3]

One call, one tensor. Inside, the wave sum has shape `[num_envs, n_points, n_components]`: with 4096 envs, 3 points and 128 components that is 1.57 million cosines per call, which a GPU does in a single kernel.

## The rule

Every state, force and parameter has a leading `num_envs` dimension.

    eta, nu:       [num_envs, 6]
    wave phases:   [num_envs, n_components]
    hull points:   [num_envs, n_points, 3]
    foil force:    [num_envs, n_strips, 3]

Code never loops over environments. A step computes forces for all envs, then runs RK4 for all envs, then returns observations `[num_envs, obs_dim]` and rewards `[num_envs]`. Per-env differences (wave seed, sea state, mass) live in tensor values, not in Python branches.

## Branches become masks

Ventilation is a per-env, per-foil switch. Without an `if`, use a mask:

    lift = torch.where(is_ventilated, lift * ventilated_factor, lift)

Resets work the same way. When env 17 falls over, only row 17 is overwritten (`state[done] = initial[done]`); the other 4095 rows continue.

## Devices

The same code runs on CPU, MPS (Apple GPU) and CUDA. The device is a choice at construction time and every tensor lives there. Nothing in the physics calls `.item()` or copies to the host mid-step, because that would stall the GPU.

Check on the laptop: with a small `num_envs` (say 64) the CPU is competitive, since launching GPU kernels has a fixed cost. Throughput grows with `num_envs` until the device is full. Pick `num_envs` by measuring steps per second at a few sizes, not by guessing.

## Practical consequences

- Keep tensors contiguous and `float32`. MPS does not support `float64`.
- Seed per env on reset so runs can be reproduced.
- A shape bug usually shows up as a silent broadcast. Test with `num_envs` = 1 and 2 and compare to a loop over single envs.

The step loop is in `src/watergym/env.py`. Back to the [index](README.md).
