# Foils

**Example.** A foil with 0.1 m^2 area moves at 6 m/s in seawater (rho = 1025 kg/m^3) at 3 degrees angle of attack.

    alpha = 3 deg = 0.0524 rad
    C_L   = 2*pi*alpha = 0.329
    lift  = 0.5 * rho * v^2 * A * C_L
          = 0.5 * 1025 * 36 * 0.1 * 0.329 = 607 N

That is 62 kg of lift from a hand-sized wing. Now add a wave. The water below a passing wave moves up at 0.5 m/s. The foil sees the flow tilted by `atan(0.5 / 6)` = 4.8 degrees, which is more than the 3 degrees it was set to.

## Strip theory

Cut the foil span into strips. Treat each strip as a 2D wing, compute its force from the flow at that strip, and add the forces and moments up (with the arms to the body centre). Different strips can sit at different depths and see different water motion. Heeling the boat then changes what each strip experiences, which a single-point model cannot show.

## Relative flow

The flow a strip sees is the water velocity minus the strip velocity, in the strip's frame:

    v_rel = v_water(x, y, z, t)  -  (v_body + omega x r)

`v_water` is the current plus the wave orbital velocity from the [wave field](02-waves.md). Angle of attack comes from the direction of `v_rel` against the chord line. This is how waves disturb a foil: they change alpha, and lift follows `~ 2*pi*alpha`. The thin-airfoil slope 2 pi is a textbook result (see Faltinsen, Hydrodynamics of High-Speed Marine Vehicles, Cambridge University Press, 2005).

## Free-surface decay

Near the surface, lift drops. At high Froude number, the free surface acts like an image vortex mirrored above it, which induces flow that reduces the foil's effective angle of attack. For a 2D foil of chord c at depth h:

    lift ratio = (1 + 16 (h/c)^2) / (2 + 16 (h/c)^2)
    h/c = 0 -> 0.50,  0.25 -> 0.67,  0.5 -> 0.83,  1 -> 0.94,  2 -> 0.99

At the surface lift halves, and one chord down the loss is only about 6 %. This is the high-Froude limit, recalled from Faltinsen's *Hydrodynamics of High-Speed Marine Vehicles* (2005) and checked against a two-vortex derivation in `research/12`. At lower Froude numbers, wave-making changes the result. `foils.py` implements this formula.

## Ventilation as a regime switch

Ventilation is air drawn down to the suction side of a foil or strut. Lift collapses at once, and the foil does not recover when the angle of attack falls back. It has hysteresis: the same (speed, depth, angle) can be ventilated or not, depending on history. The recorded onset for tail ventilation is depth Froude number `Fr_h > AR^(-1/2)` (`research/06`, section 5, citing [arXiv:2503.18015](https://arxiv.org/abs/2503.18015)).

A sample Froude number: `Fr_h = U / sqrt(g*h)`. At 6 m/s and 0.3 m depth: 3.5. At 1.0 m depth: 1.9.

So each foil carries a small state machine, attached or ventilated, with entry and exit conditions that differ. Lift gets multiplied by a reduction factor in the ventilated state. A policy that only sees smooth dynamics will walk into this edge, which is why it makes a good benchmark.

## The Moth wand

A Moth sailing dinghy holds ride height with a wand, a rod that trails on the water ahead of the bow. A low hull pushes the rod back, which through a linkage raises the flap angle, which raises lift. A high hull does the reverse. It is a mechanical proportional controller (`research/06`, section 1). In chop, the wand skips, so it follows small waves it should ignore. In WaterGym the wand is the baseline controller: sample the surface at the wand tip, convert the angle to flap deflection, and compare a learned policy against it.

Implemented in `src/watergym/foils.py`. Next: [batching](06-batching.md).
