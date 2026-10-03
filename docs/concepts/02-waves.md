# Waves

**Example.** One 1 m amplitude (2 m height) wave with an 8 s period, in deep water.

    g = 9.81
    omega = 2*pi/8        # 0.785 rad/s
    k = omega**2 / g      # 0.0629 rad/m      (dispersion)
    wavelength = 2*pi/k   # 99.9 m
    speed = omega/k       # 12.5 m/s
    orbital speed at surface = a*omega = 0.785 m/s
    orbital speed at depth 2 m = a*omega*exp(-k*2) = 0.693 m/s
    orbital speed at depth 10 m = 0.419 m/s

Water particles under the wave travel in circles of radius 1 m at the surface, once per 8 s. A foil 2 m down still sees almost 0.7 m/s of that motion.

## Surface as a sum of cosines

A real sea has no single wave. The surface elevation ζ is a sum of many, each with its own amplitude, frequency, direction and random phase:

    zeta(x, y, t) = sum_i a_i * cos(k_i*(x*cos(d_i) + y*sin(d_i)) - w_i*t + phase_i)

Each component obeys the **deep-water dispersion relation** `w^2 = g*k`. Long waves (small ω) travel faster. WaterGym draws N components once per environment reset (64 by default; the tutorials use 48 and 96), then evaluates the sum at any (x, y, t) with plain trig. No fluid solver runs.

## Where the amplitudes come from

A **spectrum** S(ω) says how much wave energy sits at each frequency. Standard choices are Pierson-Moskowitz and JONSWAP, set by two numbers:

- **Hs**, significant wave height: the mean of the highest third of waves, about 4*sqrt(m0), where m0 is the area under S(ω).
- **Tp**, peak period: the period where S(ω) is largest.

Spectrum formulas are in [DNV-RP-C205, *Environmental conditions and environmental loads*](https://www.dnv.com/energy/standards-guidelines/dnv-rp-c205-environmental-conditions-and-environmental-loads/). Slice the frequency axis into bins of width Δω and set `a_i = sqrt(2*S(w_i)*dw_i)`. Check by hand: one component with a = 1 m has variance a²/2 = 0.5, so m0 = 0.5 and `4*sqrt(0.5)` = 2.83 m. A sea with Hs = 1.9 m has total standard deviation 0.475 m.

## Orbital velocity

Below the surface each component moves water in circles whose size decays as `exp(k*z)` (z negative below the surface). The horizontal and vertical velocity of the water at a point is the sum over components. Two consumers use it: hull excitation, and foils, which add it to their relative flow (see [foils](05-foils.md)).

## Encounter frequency

A boat moving at speed U meets a wave at a different frequency than the wave has. In head seas with the example wave and U = 5 m/s:

    w_e = w + k*U = 0.785 + 0.0629*5 = 1.10 rad/s    (period 5.7 s instead of 8 s)
    following seas: w_e = w - k*U = 0.471 rad/s       (period 13.3 s)

You don't compute this. The wave field is evaluated at the vessel's current position at time t, so the shift appears on its own. This is also why wave forces must not be precomputed in a vessel-fixed frame.

Implemented in `src/watergym/waves.py`. Next: [rigid body](03-rigid-body.md).
