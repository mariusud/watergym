# Buoyancy

**Example.** Treat a hull as 4 sample points, each owning 0.025 m³ of displaced volume if fully submerged. The water surface is at height 0.

    point z (up):   +0.05   -0.02   -0.10   -0.30
    depth below surface:  0      0.02    0.10    0.30     (clamped at 0)
    submerged fraction:   0      0.2     1.0     1.0      (fraction ramps over a 0.1 m band)
    force up per point = rho * g * V * fraction
                       = 1025 * 9.81 * 0.025 * fraction  = 251 N * fraction

Total: 251 * (0 + 0.2 + 1 + 1) = 553 N. The boat sinks until the sum reaches its weight. Lift one end of the hull and only that end loses force, so the sum also produces a torque that rights the boat.

(The 0.1 m ramp is a modelling choice made for this example. The real band is set in `hydrostatics.py`.)

## Sample against the instantaneous surface

The usual textbook approach linearises the surface around the mean waterline and uses a fixed restoring stiffness, `rho*g*A_waterplane` for heave. For a 0.5 m² waterplane and a 200 kg body that gives a heave period of `2*pi*sqrt(200 / (1025*9.81*0.5))` = 1.25 s.

WaterGym instead asks, at every step, for each hull point: where is the water surface above this point right now? The answer is the wave height from the [wave field](02-waves.md) at the point's world (x, y) at time t. The point's depth below that height decides how much buoyancy it gives.

## What the point sum gives

Heave, roll and pitch stiffness fall out of the point sum, so nobody enters a `GM` or a waterplane area by hand. The same sum gives the nonlinear Froude-Krylov force, which is the pressure of the undisturbed wave integrated over the hull. Because each point samples the true surface, the hull feels the wave's height and slope where it is. A linear tool assumes small motion and evaluates the wave at the mean position, so it misses the extra buoyancy and wave force when a wave rises over the deck, and the lost force when the bow lifts clear of the water. The hull also moves through the wave field, so the encounter-frequency shift ([waves](02-waves.md)) is included.

## What it leaves out

The hull disturbs the waves and makes its own waves (diffraction and radiation). Modelling that needs precomputed coefficients and a state-space filter, and WaterGym does not have them yet.

The force is also only as smooth as the point count allows. With too few points it jumps each time a point crosses the surface, so use enough points that the ramp smooths it.

Implemented in `src/watergym/hydrostatics.py`. Next: [foils](05-foils.md).
