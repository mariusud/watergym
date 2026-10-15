# Wind

**Example.** The 10 x 4 m box barge floats with 1 m of freeboard, so its side above the water is 10 m², centred 0.5 m above the water. A 15 m/s wind, measured at 10 m, blows straight onto that side.

    speed at 0.5 m = 15 * (0.5/10)**0.12     = 10.5 m/s      (power-law profile)
    Y = 0.5 * rho_air * V**2 * c_y * A_L
      = 0.5 * 1.225 * 10.5**2 * 0.85 * 10    = 571 N         (sideways force)
    K = 0.5 m * 571 N                        = 285 N m       (heel away from the wind)
    sway acceleration = 571 / (41000 + 41000) = 0.007 m/s²   (mass plus added mass)

Air is 800 times lighter than water, so wind barely moves a barge in a few seconds. Over minutes it sets the drift, and on a light boat with a tall rig it matters at once.

## Mean wind, height and gusts

Each env draws a wind: a mean speed U10 at 10 m above the water, a direction it blows toward (radians from north toward east, as for waves), and gusts. The speed at height z follows the power law of DNV-RP-C205 sec. 2.3.2, with α = 0.12 for open sea:

    U(z, t) = (U10 + u_gust(t)) * (z/10)**0.12

The gusts are a sum of cosines, built like the waves, from the Frøya spectrum (DNV-RP-C205 sec. 2.3.5, NORSOK N-003) at 10 m:

    S(f)  = 320 * (U10/10)**2 / (1 + f~**n)**(5/(3n)),   n = 0.468
    f~    = 172 * f * (U10/10)**-0.75
    u_gust(t) = sum_i a_i * cos(2*pi*f_i*t + phase_i),   a_i = sqrt(2*S(f_i)*df_i)

The 64 components sit in log-spaced bins from 1/3600 Hz to 5 Hz. One hour is the averaging time behind U10, so slower changes belong to the mean. At U10 = 15 m/s the band holds 2.99 m²/s² of the spectrum's 3.22, a gust standard deviation of 1.73 m/s. The gusts change only the speed, along the mean direction. The vessel reads the wind at one height, the centroid of its side area above the water. `WindState(gusts=False)` gives a constant wind.

## Load on the hull

The load follows Fossen's wind model (*Handbook of Marine Craft Hydrodynamics and Motion Control*, 2nd ed., 2021, sec. 10.1). The vessel feels the apparent wind, its own velocity minus the wind's, both in the body frame:

    u_r = u - u_w,   v_r = v - v_w,   V_r**2 = u_r**2 + v_r**2,   gamma = -atan2(v_r, u_r)

    X = 0.5 * rho_air * V_r**2 * (-c_x * cos(gamma))  * A_F
    Y = 0.5 * rho_air * V_r**2 * ( c_y * sin(gamma))  * A_L
    N = 0.5 * rho_air * V_r**2 * ( c_n * sin(2*gamma)) * A_L * L

A_F and A_L are the frontal and side areas above the water, and L the length of the side silhouette. The coefficients are Fossen's simple cosine and sine forms with c_x = 0.7, c_y = 0.85 and c_n = 0.1, inside his ranges of 0.50 to 0.90, 0.70 to 0.95 and 0.05 to 0.20. They are a rough fit for a generic hull, not measured for these vessels. X and Y act at the centroids of their areas, which adds roll K = -z_L Y and pitch M = z_F X. Vertical force is zero. A moving vessel in still air feels a head wind of its own speed, so X also gives air drag.

`windage_from_mesh` finds the areas: it draws the mesh's outline above the waterline at the initial pose on a fine grid, from the front and from the side, and counts the covered cells. For the barge it gives A_F = 5.1 m² and A_L = 12.7 m², a little above the plain box because of the bulwarks and bollards. Set `Vessel.windage` to your own `Windage` to override it. A submerged ROV gets zero area and no wind load.

## Zero wind

`WaterEnv(..., wind_state=WindState(speed=15.0, direction_rad=math.pi / 2))` turns wind on. The default, zero speed, adds no term at all, so results match a run without wind bit for bit, and the wind draws its phases from its own generator, so turning it on leaves the seas unchanged.

## What it leaves out

The gusts are uniform in space, with no lateral or vertical gust and no change of direction. The wind sees the hull at one height and ignores the vessel's rotation rate. The Moth's sail is still a fixed forward thrust, so a Moth with wind also gets the sail's side area as plain windage.

Implemented in `src/watergym/wind.py`. Back to the [index](README.md).
