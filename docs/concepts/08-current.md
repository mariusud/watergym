# Current

**Example.** An Otter USV points into a 0.5 m/s current. Its linear surge damping is 77.55 N s/m and its quadratic surge damping is zero. To stay in one place it must push against

    drag = d_u * U = 77.55 * 0.5 = 38.8 N

which is 16 % of what its two 119.7 N thrusters give. With the thrusters off it drifts with the water and reaches 0.5 m/s in a few seconds.

## Relative velocity

Damping comes from water moving past the hull, so it acts on the velocity relative to the water. With the current velocity `nu_c` expressed in the body frame (its angular part is zero):

    nu_r = nu - nu_c

Fossen's equation with a current (*Handbook of Marine Craft Hydrodynamics and Motion Control*, 2nd ed., 2021, sec. 10.3):

    M_RB * nu_dot + C_RB(nu)*nu + M_A * nu_r_dot + C_A(nu_r)*nu_r + D(nu_r)*nu_r = tau

The rigid-body terms keep `nu`: the mass of the vessel does not care what the water does. Added mass, its Coriolis part and damping see `nu_r`. Solving for `nu_dot`:

    nu_dot = M^-1 (tau - C_RB(nu)*nu - C_A(nu_r)*nu_r - D(nu_r)*nu_r + M_A * nu_c_dot)

The last term matters even for a current that never changes. A current fixed in NED turns in the body frame when the vessel yaws:

    v_c_dot = -omega x v_c

Leave it out and a body that spins while drifting with the water would build up a speed relative to the water from nothing. `tests/test_current.py` checks that it does not.

Hydrostatics and Froude-Krylov forces do not change: they come from pressure, and a uniform current does not change the pressure.

## Depth profile

The current flows horizontally at `U * f(d)` toward the direction `beta`, measured from north toward east, with `d` the depth below the mean surface:

    v_c = U * f(d) * (cos(beta), sin(beta), 0)

`f` is a table of factors at depths, linear in between, constant beyond the ends, and 1 when there is no table. A wind-driven current that halves by 2 m deep is `profile_depths = [0, 2]`, `profile_factors = [1, 0.5]`. The hull uses the current at its centre of gravity. As the CG moves up or down through the profile, `nu_c` changes too, so `nu_c_dot` gets one more term, `R^T (d v_c / dz) z_dot`. Fossen derives the equation for a uniform current, so this term is our extension. It is zero without a profile.

Each foil strip adds the current at its own depth to the water velocity it sees, as it does with the wave orbital velocity (see [foils](05-foils.md)). A deep foil under a surface current feels less of it than the hull does.

## Using it

The current belongs to the sea, with one speed and direction per env:

```python
env.sea.current = Current.uniform(env.num_envs, speed=torch.rand(env.num_envs), direction=math.pi)
```

Envs that reset keep their current. Change `env.sea.current.speed[ids]` to draw a new one. With no current, `sea.current` is `None` and every result is bit-identical to a run without this feature.

Implemented in `src/watergym/current.py`, used by `rigid_body.acceleration` and `foils.foil_loads`.
