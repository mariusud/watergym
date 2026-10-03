# Rigid body

**Example.** A 100 kg body in water. Push it with 50 N along its length. Ignoring the water, it accelerates at 0.5 m/s^2. With an added mass of 20 kg surge:

    a = F / (m + m_a) = 50 / (100 + 20) = 0.417 m/s^2

Moving sideways, the hull pushes more water, so say m_a = 80 kg: `50 / 180 = 0.278 m/s^2`. Same body, different acceleration per direction. Ignoring this makes heave and sway wrong by 50 to 100 percent or more (`research/00-plan.md`, section 2).

## The Fossen equation

    (M_RB + M_A) * nu_dot + C(nu)*nu + D(nu)*nu + g(eta) = tau

Solve for `nu_dot`, integrate, repeat. Term by term:

- **M_RB**: mass and inertia of the body. 6x6.
- **M_A, added mass**: accelerating the hull also accelerates nearby water, which acts like extra mass. It differs per direction and is a property of the hull shape.
- **C(nu)**: Coriolis and centripetal terms. They appear because ν is in a rotating frame. A body spinning about one axis while moving along another feels a force from this.
- **D(nu)**: damping. A linear part (skin friction, wave radiation) plus a quadratic part, `-d * |u| * u`. At 2 m/s with d = 30 N s^2/m^2 the quadratic drag is `30 * 2 * 2` = 120 N.
- **g(eta)**: gravity and buoyancy restoring force. See [buoyancy](04-buoyancy.md).
- **tau**: everything we apply: propulsion, foil forces, wave excitation, wind.

The equation form is from [Fossen's handbook](https://fossen.biz/html/marineCraftModel.html).

## Why RK4

Take an oscillator with a 1 s period (a heave bounce on a stiff foil is about this fast) and a 10 ms step. After 10 s of simulated time, starting at amplitude 1:

    explicit Euler: amplitude 7.17   (energy grows, the sim blows up)
    RK4:            amplitude 1.0000

Euler adds energy to oscillating systems. RK4 evaluates the derivative four times per step and keeps the amplitude to about 1e-6 here. The cost is four force evaluations per step, which batching makes cheap (see [batching](06-batching.md)). Run times are short, so accuracy matters more than the extra evaluations.

## Notes

- Forces on the hull come from waves, buoyancy and foils, all evaluated at each RK4 stage with the stage's state. Skipping this and holding forces fixed over the step reduces RK4 to first order.
- The mass matrix is constant in the body frame, so `(M_RB + M_A)^-1` is computed once per robot.

Implemented in `src/watergym/rigid_body.py`. Next: [buoyancy](04-buoyancy.md).
