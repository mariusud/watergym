# Frames and state

**Example.** A boat sits at 10 m north, 5 m east of the origin, 0.3 m above the mean waterline, heading east (yaw 90 degrees), moving forward at 2 m/s.

    eta = [10, 5, -0.3, 0, 0, pi/2]   # x, y, z, roll, pitch, yaw  (world, NED)
    nu  = [2, 0, 0, 0, 0, 0]          # u, v, w, p, q, r              (body)

The z entry is -0.3 because NED has z pointing down. The boat's world-frame velocity is not in these vectors. Rotate the body velocity into the world: with yaw 90 degrees, forward 2 m/s becomes 0 m/s north and 2 m/s east. In code that is `R(phi, theta, psi) @ nu[:3]`.

## Two frames

- **World frame (NED).** x north, y east, z down. Fixed to the earth. Waves, current and wind are defined here.
- **Body frame.** Fixed to the vessel. x forward (surge), y starboard (sway), z down through the keel (heave). Thrusters, foils and sensors are defined here.

## Two vectors

- **η = (x, y, z, φ, θ, ψ)**: position and roll, pitch, yaw, in the world frame.
- **ν = (u, v, w, p, q, r)**: linear velocity (surge, sway, heave) and angular velocity (roll rate, pitch rate, yaw rate), in the body frame.

They are related by the kinematic equation `eta_dot = J(eta) @ nu`, where J holds the rotation matrix for position and a Euler-rate matrix for the angles. This is the notation of Fossen, [*Handbook of Marine Craft Hydrodynamics and Motion Control*, 2nd ed. (Wiley, 2021)](https://fossen.biz/html/marineCraftModel.html). Most marine control papers use it.

## Why body-frame velocity

Drag, added mass and foil forces depend on how the water moves past the hull. That is a body-frame question: "how fast am I going forward". Writing the dynamics in ν keeps the mass matrix constant and the hydrodynamic coefficients fixed.

Waves live in the world frame, so wave code takes world positions. The step is: rotate a hull point to the world with η, ask the wave field for the surface height there, then rotate any force back to the body.

## Watch for

- Positive pitch θ is bow up. Because z points down, a point that moves up has a more negative z.
- Euler angles have a singularity at pitch ±90 degrees. Boats and foilers stay far from it, so the sim uses Euler angles. A vehicle that can flip would need quaternions.

Implemented in `src/watergym/rigid_body.py`. Next: [waves](02-waves.md).
