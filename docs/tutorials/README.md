# Tutorials

Each page builds on the one before and has code you can paste into a script and run. They only step and draw; nothing trains.

1. [First sea](01-first-sea.md): make a sea, sample the surface height, draw it.
2. [Make it float](02-make-it-float.md): drop a box barge, watch it settle, check the draft against Archimedes.
3. [Foiling Moth](03-foiling-moth.md): fly an International Moth with its wand, read the action vector and foil loads, change the sea and the wand gain.
4. [Your own vessel](04-your-own-vessel.md): define a box ROV from scratch, step it, draw it.
5. [Batching and devices](05-batching-and-devices.md): `num_envs`, seeds, CPU, MPS and CUDA, drawing a few envs out of many.

Run the code with `uv run --extra viz script.py`, the same way the README runs the files in `examples/`. The pages use `make_viewer("gl")`, which opens a window. Replace `"gl"` with `"null"` to run without drawing.

The physics behind each page is in [concepts](../concepts/README.md). Read those when a term is new.
