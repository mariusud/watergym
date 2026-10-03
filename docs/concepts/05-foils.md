# Foils

**Example.** A foil with 0.1 m² area moves at 6 m/s in seawater (rho = 1025 kg/m³) at 3 degrees angle of attack.

    alpha = 3 deg = 0.0524 rad
    C_L   = 2*pi*alpha = 0.329
    lift  = 0.5 * rho * v^2 * A * C_L
          = 0.5 * 1025 * 36 * 0.1 * 0.329 = 607 N

That is 62 kg of lift from a 0.1 m² wing. Now add a wave. The water below a passing wave moves up at 0.5 m/s. The foil sees the flow tilted by `atan(0.5 / 6)` = 4.8 degrees, which is more than the 3 degrees it was set to.

## Strip theory

Cut the foil span into strips. Treat each strip as a 2D wing, compute its force from the flow at that strip, and add the forces and moments up (with the arms to the body centre). Different strips can sit at different depths and see different water motion. Heeling the boat then changes what each strip experiences, which a single-point model cannot show.

## Relative flow

The flow a strip sees is the water velocity minus the strip velocity, in the strip's frame:

    v_rel = v_water(x, y, z, t)  -  (v_body + omega x r)

`v_water` is the current plus the wave orbital velocity from the [wave field](02-waves.md). Angle of attack comes from the direction of `v_rel` against the chord line. This is how waves disturb a foil: they change alpha, and lift follows `~ 2*pi*alpha`. The thin-airfoil slope 2π is given in Faltinsen, *Hydrodynamics of High-Speed Marine Vehicles* (Cambridge University Press, 2005).

## Free-surface decay

Near the surface, lift drops. At high Froude number, the free surface acts like an image vortex mirrored above it, which induces flow that reduces the foil's effective angle of attack. For a 2D foil of chord c at depth h:

    lift ratio = (1 + 16 (h/c)^2) / (2 + 16 (h/c)^2)
    h/c = 0 -> 0.50,  0.25 -> 0.67,  0.5 -> 0.83,  1 -> 0.94,  2 -> 0.99

At the surface lift halves, and one chord down the loss is only about 6 %. This is the high-Froude limit, given in Faltinsen (2005). At lower Froude numbers, wave-making changes the result.

`foils.py` applies this formula to horizontal foils only. A vertical strut that pierces the surface gets no image-vortex factor: it lifts with its immersed span, and strips above the water carry nothing. A strip fades from dry to wet over its own height plus `WETTING_RAMP_CHORDS` (0.1) chords, so a horizontal foil does not switch its lift on in a single step.

## Ventilation as a regime switch

Ventilation is air drawn down to the suction side of a foil or strut. Lift collapses at once, and the foil does not recover when the angle of attack falls back. It has hysteresis: the same (speed, depth, angle) can be ventilated or not, depending on history. Harwood et al. record the onset for tail ventilation at depth Froude number `Fr_h > AR^(-1/2)` ([*On the ventilation of surface-piercing hydrofoils under steady-state conditions*](https://www.cambridge.org/core/journals/journal-of-fluid-mechanics/article/on-the-ventilation-of-surfacepiercing-hydrofoils-under-steadystate-conditions/3935077A7B4DE189B3C14A6CBBAD0A4C), *Journal of Fluid Mechanics*; preprint [arXiv:2503.18015](https://arxiv.org/abs/2503.18015)).

A sample Froude number: `Fr_h = U / sqrt(g*h)`. At 6 m/s and 0.3 m depth: 3.5. At 1.0 m depth: 1.9.

So each foil carries a small state machine, attached or ventilated, with entry and exit conditions that differ:

- **Onset.** Any wet strip within one chord of the surface has `Fr_h > AR^(-1/2)` and `|alpha|` above 10 degrees. Air comes down from the surface, so the shallowest strips decide. A strut can ventilate from its top strips even when most of it is deep.
- **Washout.** Every wet strip is below 4 degrees, the foil is deeper than one chord (horizontal foils only, since a surface-piercing strut always has an air path), and both have held for `washout_time_s` (0.5 s).

While ventilated, lift is multiplied by 0.25 and profile drag by 1.5. All of these numbers are placeholders: Harwood et al. map the regimes but give no thresholds, so each one is a `Foil` field meant to be domain-randomised. Onset is deterministic for now; a probability ramp with `alpha_crit(Fr_h)` comes later.

A policy trained on smooth dynamics has not seen a switch like this, which makes ventilation a useful benchmark.

## The Moth wand

An International Moth, a foiling dinghy, holds ride height with a wand, a rod that trails on the water ahead of the bow. A low hull pushes the rod back, which through a linkage raises the flap angle, which raises lift. A high hull does the reverse. It is a mechanical proportional controller. In short, steep waves the wand skips, so it follows small waves it should ignore. In WaterGym the wand is the baseline controller: sample the surface at the wand tip, convert the angle to flap deflection, and compare a learned policy against it.

Implemented in `src/watergym/foils.py`. Next: [batching](06-batching.md).
