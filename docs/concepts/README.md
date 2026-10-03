# WaterGym concepts

Short pages. Each starts with an example you can check by hand, then explains the idea and names the module that implements it.

Suggested reading order:

1. [Frames and state](01-frames-and-state.md): NED and body frames, the pose vector η and velocity vector ν.
2. [Waves](02-waves.md): from a spectrum to surface height and orbital velocity.
3. [Rigid body](03-rigid-body.md): the Fossen equation term by term, and why the integrator is RK4.
4. [Buoyancy](04-buoyancy.md): sampling the hull against the instantaneous surface.
5. [Foils](05-foils.md): strip theory, relative flow, free-surface decay, ventilation, the Moth wand.
6. [Batching](06-batching.md): why every quantity is a `[num_envs, ...]` tensor.

Background: `research/00-plan.md` section 2 (physics primer) and its glossary.
