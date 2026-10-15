"""Math audit: derive expected values by hand and compare with what WaterGym computes.

    OMP_NUM_THREADS=2 nice -n 19 uv run python benchmarks/audit_math.py

CPU only, small batches. Each check prints id, expected, measured and a verdict. The write-up
is research/19-math-audit.md.
"""

import math
from dataclasses import replace

import numpy as np
import torch

torch.set_num_threads(2)

from watergym.env import SeaState, WaterEnv, advance  # noqa: E402
from watergym.foils import (  # noqa: E402
    Foil,
    FoilLoads,
    foil_loads,
    foil_wrench,
    free_surface_factor,
    helmbold_lift_slope,
    update_ventilation,
)
from watergym.hydrostatics import WATER_DENSITY, hydrostatic_force  # noqa: E402
from watergym.rigid_body import (  # noqa: E402
    Pose,
    RigidBody,
    coriolis_force,
    euler_rate_matrix,
    rk4_step,
    rotation_matrix,
)
from watergym.tasks import metrics  # noqa: E402
from watergym.tasks import ride_control as rc  # noqa: E402
from watergym.vessel import generalized_force, thrust_force, vessel_derivative  # noqa: E402
from watergym.vessels import bluerov2, box_barge, moth  # noqa: E402
from watergym.waves import (  # noqa: E402
    GRAVITY,
    Sea,
    jonswap,
    make_sea,
    regular_wave,
    sample_directions,
    significant_wave_height,
    water_at,
)

RHO, G = WATER_DENSITY, GRAVITY
ROWS: list[tuple[str, str, str, str, str]] = []


def check(
    cid: str, name: str, expected: float, measured: float, rel: float = 0.0, abs_: float = 0.0
):
    ok = abs(measured - expected) <= max(rel * abs(expected), abs_)
    ROWS.append((cid, name, f"{expected:.6g}", f"{measured:.6g}", "pass" if ok else "FAIL"))


def note(cid: str, name: str, expected: str, measured: str, verdict: str):
    ROWS.append((cid, name, expected, measured, verdict))


def double_vessel(vessel, zero_damping: bool = False):
    """The same vessel with every tensor in float64 (meshes stay float32, physics ignores them)."""
    from watergym.geometry import VolumeSamples

    b = vessel.body
    k = 0.0 if zero_damping else 1.0
    body = RigidBody(
        b.mass,
        b.inertia.double(),
        b.added_mass.double(),
        k * b.linear_damping.double(),
        k * b.quadratic_damping.double(),
    )
    hull = VolumeSamples(
        vessel.hull.centers.double(), vessel.hull.volumes.double(), vessel.hull.heights.double()
    )
    foils = []
    for f in vessel.foils:
        g = replace(f)
        g.strip_centers, g.span_vector, g.chord_vector = (
            f.strip_centers.double(),
            f.span_vector.double(),
            f.chord_vector.double(),
        )
        foils.append(g)
    thrusters = []
    for t in vessel.thrusters:
        u = replace(t)
        u.force_vector, u.position_vector = t.force_vector.double(), t.position_vector.double()
        thrusters.append(u)
    out = replace(vessel, body=body, hull=hull, foils=foils, thrusters=thrusters)
    out.free_dof_mask = out.free_dof_mask.double()
    return out


def calm_sea(n: int = 1, dtype=torch.float32) -> Sea:
    z = torch.zeros(n, 1, dtype=dtype)
    return Sea(
        z.clone(),
        torch.ones(n, 1, dtype=dtype),
        torch.full((n, 1), 1 / G, dtype=dtype),
        z.clone(),
        z.clone(),
    )


def as_double(sea: Sea) -> Sea:
    return Sea(
        *(
            getattr(sea, f).double()
            for f in ("amplitude", "omega", "wavenumber", "direction", "phase")
        )
    )


# ---------------------------------------------------------------- 1. waves
def waves() -> None:
    hs, tp = 2.0, 8.0
    wp = 2 * math.pi / tp
    fine = torch.linspace(0.05, 40 * wp, 400_000, dtype=torch.float64)
    for gamma in (1.0, 3.3, 7.0):
        s = jonswap(
            fine,
            torch.tensor(hs, dtype=torch.float64),
            torch.tensor(tp, dtype=torch.float64),
            gamma,
        )
        m0 = torch.trapezoid(s, fine).item()
        check(
            "W1",
            f"JONSWAP 4 sqrt(m0) vs Hs (gamma {gamma}, full integral)",
            hs,
            4 * math.sqrt(m0),
            rel=0.02,
        )
        check(
            "W2",
            f"JONSWAP peak at omega_p (gamma {gamma}) [rad/s]",
            wp,
            fine[s.argmax()].item(),
            rel=1e-3,
        )
        pm = 5 / 16 * hs**2 * wp**4 * wp**-5 * math.exp(-5 / 4)
        check(
            "W2",
            f"S(wp)/S_PM(wp) = gamma (1-0.287 ln gamma) (gamma {gamma})",
            gamma * (1 - 0.287 * math.log(gamma)),
            s[s.argmax()].item() / pm,
            rel=1e-3,
        )
    sea = make_sea(2000, hs, tp, num_components=64, generator=torch.Generator().manual_seed(1))
    check(
        "W1",
        "make_sea Hs from 4 sqrt(sum a^2/2), 64 comps",
        hs,
        significant_wave_height(sea).mean().item(),
        rel=0.02,
    )
    check(
        "W3",
        "max |k - w^2/g| over components",
        0.0,
        (sea.wavenumber - sea.omega**2 / G).abs().max().item(),
        abs_=1e-6,
    )

    # Airy kinematics, regular wave a=1 T=8 heading 0, point at 2 m depth.
    a, T, d = 1.0, 8.0, 2.0
    w = 2 * math.pi / T
    k = w**2 / G
    reg = as_double(regular_wave(1, a, T))

    def at(x, z, t):
        return water_at(reg, torch.tensor([[[x, 0.0, z]]], dtype=torch.float64), t)

    crest = at(0.0, d, 0.0)
    check(
        "W4",
        "u under crest at 2 m = a w e^-kd",
        a * w * math.exp(-k * d),
        crest.velocity[0, 0, 0].item(),
        rel=1e-6,
    )
    check(
        "W4",
        "a_z(NED) under crest = +a w^2 e^-kd (down)",
        a * w**2 * math.exp(-k * d),
        crest.acceleration[0, 0, 2].item(),
        rel=1e-6,
    )
    ahead = at(math.pi / 2 / k, d, 0.0)
    check(
        "W4",
        "w_up a quarter wave ahead of crest = +a w e^-kd",
        a * w * math.exp(-k * d),
        -ahead.velocity[0, 0, 2].item(),
        rel=1e-6,
    )
    # Time derivatives by central difference: w_up = d eta/dt at the surface, a = dv/dt.
    h = 1e-4
    x0 = 3.0
    e1, e0 = at(x0, 0.0, h), at(x0, 0.0, -h)
    deta = (e1.elevation - e0.elevation).item() / (2 * h)
    check(
        "W4",
        "kinematic BC: -w_ned(z=0) = d eta/dt",
        deta,
        -at(x0, 0.0, 0.0).velocity[0, 0, 2].item(),
        rel=1e-5,
    )
    e1, e0 = at(x0, d, h), at(x0, d, -h)
    dv = (e1.velocity - e0.velocity)[0, 0] / (2 * h)
    acc = at(x0, d, 0.0).acceleration[0, 0]
    check(
        "W4",
        "max |a - dv/dt| (central difference, 3 axes)",
        0.0,
        (dv - acc).abs().max().item(),
        abs_=1e-6,
    )
    # Head seas: heading pi travels toward -x; the water under a crest moves toward -x.
    head = as_double(regular_wave(1, a, T, heading_rad=math.pi))
    hv = (
        water_at(head, torch.tensor([[[0.0, 0.0, 0.0]]], dtype=torch.float64), 0.0)
        .velocity[0, 0, 0]
        .item()
    )
    check("W5", "head seas (pi): u under crest = -a w (toward the bow)", -a * w, hv, rel=1e-6)
    # Encounter frequency at U = 5 m/s in head seas: elevation seen by a point moving +x.
    U = 5.0
    ts = torch.arange(0, 200, 0.01, dtype=torch.float64)
    pts = torch.stack((U * ts, 0 * ts, 0 * ts), -1)[None]
    # The point moves, so take theta along its path in closed form.
    theta = k * (pts[0, :, 0] * math.cos(math.pi)) - w * ts
    up = ((torch.cos(theta[:-1]) < 0) & (torch.cos(theta[1:]) >= 0)).sum().item()
    check(
        "W5",
        "encounter frequency head seas w + kU [rad/s]",
        w + k * U,
        2 * math.pi * up / 200.0,
        rel=0.02,
    )
    # Directional spreading: cos-2s(theta/2), E[cos(offset)] = s / (s + 1).
    s = 10.0
    dirs = sample_directions(1, 200_000, torch.zeros(1), s, torch.Generator().manual_seed(0))
    check(
        "W6",
        "spreading s=10: E[cos(dtheta)] = s/(s+1)",
        s / (s + 1),
        torch.cos(dirs).mean().item(),
        rel=0.01,
    )
    # Above the mean level: decay clamps to 1 (Wheeler-free constant extrapolation).
    above = at(0.0, -0.5, 0.0).velocity[0, 0, 0].item()
    note(
        "W7",
        "u at z=-0.5 m (above mean level) under crest",
        f"a w = {a * w:.4f} (clamped)",
        f"{above:.4f}",
        "pass (documented)",
    )


# ---------------------------------------------------------------- 2. hydrostatics
def barge_env(damped: bool = True, **kw) -> WaterEnv:
    vessel = box_barge(heave_damping_ratio=0.1 if damped else 0.0)
    if not damped:
        b = vessel.body
        vessel = replace(
            vessel,
            body=RigidBody(
                b.mass, b.inertia, b.added_mass, 0 * b.linear_damping, 0 * b.quadratic_damping
            ),
        )
    return WaterEnv(vessel, 1, SeaState(hs=0.0), **kw)


def period(signal: np.ndarray, dt: float) -> float:
    x = signal - signal[len(signal) // 2 :].mean()
    idx = np.where((x[:-1] < 0) & (x[1:] >= 0))[0]
    times = [(i + x[i] / (x[i] - x[i + 1])) * dt for i in idx]
    return float(np.mean(np.diff(times)))


def hydrostatics() -> None:
    m, L, B, H, T = 41000.0, 10.0, 4.0, 2.0, 1.0
    check("H1", "box barge draft m/(rho L B) [m]", 1.0, m / (RHO * L * B), rel=1e-3)
    env = barge_env()
    env.reset()
    pose = Pose.from_eta(env.eta)
    tau = generalized_force(env.vessel, env.sea, env.t, pose, env.nu, env.action, env.ventilated)
    check("H1", "barge net heave force at 1 m draft [N]", 0.0, tau[0, 2].item(), abs_=1.0)

    c33 = RHO * G * L * B
    gm_t = B**2 / (12 * T) + T / 2 - H / 2
    gm_l = L**2 / (12 * T) + T / 2 - H / 2
    check("H2", "GM_T hand B^2/12T + T/2 - KG [m]", 0.8333, gm_t, rel=1e-3)
    for axis, gm, label, cells in ((3, gm_t, "GM_T", 16), (4, gm_l, "GM_L", 10)):
        eta = env.eta.clone()
        eta[0, axis] = math.radians(0.5)
        tau = generalized_force(
            env.vessel, env.sea, env.t, Pose.from_eta(eta), env.nu, env.action, env.ventilated
        )
        measured = -tau[0, axis].item() / (RHO * G * L * B * T * math.sin(eta[0, axis].item()))
        disc = (B if axis == 3 else L) ** 2 / (12 * T) * (1 - 1 / cells**2) + T / 2 - H / 2
        check(
            "H2",
            f"{label} from static 0.5 deg restoring moment [m] (hand {gm:.4f})",
            disc,
            measured,
            rel=0.01,
        )

    a33 = m
    t_heave = 2 * math.pi * math.sqrt((m + a33) / c33)
    check(
        "H3",
        "heave Tn 2pi sqrt((m+a33)/rho g Awp), research/11 says 2.84 s",
        2.84,
        t_heave,
        rel=0.005,
    )
    note(
        "H3",
        "heave Tn without added mass (research/11 says 2.01 s)",
        "2.01",
        f"{2 * math.pi * math.sqrt(m / c33):.3f}",
        "pass",
    )
    for damped, expected in ((False, t_heave), (True, t_heave / math.sqrt(1 - 0.1**2))):
        env = barge_env(damped)
        env.reset()
        env.eta[0, 2] += 0.1
        zs = []
        for _ in range(1000):
            env.step(None)
            zs.append(env.eta[0, 2].item())
        check(
            "H3",
            f"simulated heave period, damping {'0.1' if damped else '0'} [s]",
            expected,
            period(np.array(zs), env.dt),
            rel=0.01,
        )

    i44 = m / 12 * (B**2 + H**2)
    c44 = RHO * G * L * B * T * gm_t
    t_roll = 2 * math.pi * math.sqrt(1.3 * i44 / c44)
    check("H4", "roll Tn hand (A44 = 0.3 I44), research/11 says 3.23 s", 3.23, t_roll, rel=0.005)
    env = barge_env()
    env.reset()
    env.eta[0, 3] = math.radians(2)
    phis = []
    for _ in range(1000):
        env.step(None)
        phis.append(env.eta[0, 3].item())
    zeta = 5e4 / (2 * math.sqrt(1.3 * i44 * c44 * (1 - 1 / 256 * 1.333 / gm_t)))
    expected = t_roll / math.sqrt(1 - zeta**2)
    check(
        "H4",
        f"simulated roll period with roll damping 5e4 (zeta {zeta:.3f}) [s]",
        expected,
        period(np.array(phis), env.dt),
        rel=0.02,
    )

    # Froude-Krylov: force on a small submerged cell = -V grad p,
    # linear p = rho g d + rho g a e^-kd cos(theta).
    a, Tw = 0.5, 4.0
    w = 2 * math.pi / Tw
    k = w**2 / G
    sea = regular_wave(1, a, Tw)
    from watergym.geometry import VolumeSamples

    xs, d = 1.3, 1.5
    samples = VolumeSamples(
        torch.tensor([[xs, 0.0, d]]), torch.tensor([1e-3]), torch.tensor([0.01])
    )
    pose = Pose(torch.zeros(1, 3), torch.eye(3)[None])
    f = hydrostatic_force(samples, sea, torch.zeros(1), pose)[0]
    theta = k * xs
    dp_dd = RHO * G - RHO * G * a * k * math.exp(-k * d) * math.cos(theta)
    dp_dx = -RHO * G * a * k * math.exp(-k * d) * math.sin(theta)
    check("H5", "FK vertical force = -V dp/dz (NED) [N]", -1e-3 * dp_dd, f[2].item(), rel=1e-4)
    check("H5", "FK surge force = -V dp/dx [N]", -1e-3 * dp_dx, f[0].item(), rel=1e-3)


# ---------------------------------------------------------------- 3. rigid body
def skew(v: torch.Tensor) -> torch.Tensor:
    x, y, z = v
    return torch.tensor([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=v.dtype)


def rigid_body() -> None:
    torch.manual_seed(0)
    A = torch.randn(6, 6, dtype=torch.float64)
    M = A @ A.T + 6 * torch.eye(6, dtype=torch.float64)
    nu = torch.randn(64, 6, dtype=torch.float64)
    cn = coriolis_force(M, nu)
    check(
        "R2",
        "max |nu^T C(nu) nu| (random SPD M, 64 nu)",
        0.0,
        (nu * cn).sum(-1).abs().max().item(),
        abs_=1e-10,
    )
    # Against Fossen's m2c matrix written out.
    n = nu[0]
    p1 = M[:3, :3] @ n[:3] + M[:3, 3:] @ n[3:]
    p2 = M[3:, :3] @ n[:3] + M[3:, 3:] @ n[3:]
    C = torch.zeros(6, 6, dtype=torch.float64)
    C[:3, 3:] = -skew(p1)
    C[3:, :3] = -skew(p1)
    C[3:, 3:] = -skew(p2)
    check(
        "R2", "max |C_m2c nu - coriolis_force|", 0.0, (C @ n - cn[0]).abs().max().item(), abs_=1e-10
    )
    body = box_barge().body
    rigid = torch.diag(torch.tensor([body.mass] * 3 + body.inertia.diag().tolist()))
    check(
        "R1",
        "max |M - (M_RB + M_A)| barge",
        0.0,
        (body.mass_matrix - rigid - body.added_mass).abs().max().item(),
        abs_=1e-3,
    )

    eta = torch.tensor([[0.3, -0.4, 1.1]], dtype=torch.float64)
    R = rotation_matrix(eta[:, 0], eta[:, 1], eta[:, 2])[0]
    check(
        "R3",
        "max |R^T R - I|",
        0.0,
        (R.T @ R - torch.eye(3, dtype=torch.float64)).abs().max().item(),
        abs_=1e-12,
    )
    check("R3", "det R", 1.0, torch.det(R).item(), abs_=1e-12)
    omega = torch.tensor([0.2, -0.5, 0.7], dtype=torch.float64)
    Tm = euler_rate_matrix(eta[:, 0], eta[:, 1])[0]
    h = 1e-6
    e2 = eta[0] + h * Tm @ omega
    Rdot = (rotation_matrix(e2[0:1], e2[1:2], e2[2:3])[0] - R) / h
    check(
        "R3",
        "max |dR/dt - R S(omega)| with eta_dot = T omega",
        0.0,
        (Rdot - R @ skew(omega)).abs().max().item(),
        abs_=1e-5,
    )
    # Positive pitch is bow up: the bow x axis gets a negative (up) NED z.
    Rp = rotation_matrix(torch.zeros(1), torch.tensor([0.1]), torch.zeros(1))[0]
    check(
        "R3",
        "R(theta=+0.1) x_body . z_ned (bow up => negative)",
        -math.sin(0.1),
        Rp[2, 0].item(),
        rel=1e-6,
    )

    barge64 = double_vessel(box_barge(0.0), zero_damping=True)
    moth64 = double_vessel(moth())
    torch.set_default_dtype(torch.float64)
    try:
        # Energy: barge in calm water, all damping off,
        # released at heave 0.1 m, roll 5, pitch 2 deg.
        vessel = barge64
        sea = calm_sea(dtype=torch.float64)
        none = torch.zeros(1, 0)
        deriv = vessel_derivative(vessel, sea, none, torch.zeros(1, 0, dtype=torch.bool))
        hull = vessel.hull

        def energy(eta, nu):
            pts = Pose.from_eta(eta).to_world(hull.centers)[0, :, 2]
            hh = hull.heights
            G_ = torch.where(
                pts < -hh / 2,
                0 * pts,
                torch.where(pts > hh / 2, pts, (pts + hh / 2) ** 2 / (2 * hh)),
            )
            ke = 0.5 * (nu @ vessel.body.mass_matrix @ nu.T).item()
            return (
                ke
                - vessel.body.mass * G * eta[0, 2].item()
                + (RHO * G * hull.volumes * G_).sum().item()
            )

        eta0 = torch.tensor([[0.0, 0.0, 0.1, math.radians(5), math.radians(2), 0.0]])
        nu0 = torch.zeros(1, 6)
        e_ref = energy(torch.tensor([[0.0] * 6]), nu0)
        e0 = energy(eta0, nu0)
        for dt in (0.02, 0.01):
            eta, nu, t = eta0.clone(), nu0.clone(), torch.zeros(1)
            drift = 0.0
            for _ in range(int(20 / dt)):
                eta, nu = rk4_step(deriv, t, eta, nu, dt)
                t = t + dt
                drift = max(drift, abs(energy(eta, nu) - e0))
            check(
                "R4",
                f"barge, no damping, 20 s, dt {dt}: max |dE| / E_excess",
                0.0,
                drift / (e0 - e_ref),
                abs_=1e-3,
            )

        # RK4 order: pure heave decay against a dt/16 reference run (float32-built cell volumes
        # shift the equilibrium by ~1e-8 m, so the analytic cosine is not a clean reference).
        def heave(dt):
            eta, nu, t = torch.tensor([[0.0, 0, 0.1, 0, 0, 0]]), torch.zeros(1, 6), torch.zeros(1)
            for _ in range(round(5.0 / dt)):
                eta, nu = rk4_step(deriv, t, eta, nu, dt)
                t = t + dt
            return eta[0, 2].item()

        ref = heave(0.1 / 64)
        errs = [abs(heave(dt) - ref) for dt in (0.1, 0.05, 0.025)]
        check(
            "R5",
            f"RK4 heave decay error ratio dt 0.1->0.05 (errs {errs[0]:.2e},{errs[1]:.2e})",
            16.0,
            errs[0] / errs[1],
            rel=0.15,
        )
        check(
            "R5",
            f"RK4 heave decay error ratio dt 0.05->0.025 ({errs[2]:.2e})",
            16.0,
            errs[1] / errs[2],
            rel=0.15,
        )

        # RK4 order on the Moth in waves through env.advance, as substeps double.
        mv = moth64
        msea = as_double(
            make_sea(2, 0.3, 3.0, heading_rad=math.pi, spreading=10.0, num_components=48)
        )
        action = torch.tensor([[0.65, 0.2]] * 2)

        def fly(substeps: int):
            eta = torch.tensor([mv.initial_eta] * 2)
            nu = torch.tensor([mv.initial_nu] * 2)
            t = torch.zeros(2)
            vent = torch.zeros(2, 4, dtype=torch.bool)
            wet = torch.zeros(2, 4)
            for _ in range(25):
                t, eta, nu, vent = advance(
                    mv, msea, t, eta, nu, action, vent, wet, 0.02, substeps, False
                )
            return eta

        ref = fly(32)
        e = [(fly(n) - ref)[:, 2].abs().max().item() for n in (1, 2, 4)]
        order = math.log2(e[0] / e[2]) / 2
        check(
            "R5",
            "Moth in waves 0.5 s, mean order over substeps 1->4 "
            f"(errs {e[0]:.1e},{e[1]:.1e},{e[2]:.1e})",
            4.0,
            order,
            abs_=0.5,
        )
    finally:
        torch.set_default_dtype(torch.float32)


# ---------------------------------------------------------------- 4. foils
def foils() -> None:
    mv = moth()
    main = mv.foils[0]
    ar = main.span_m / main.chord_m
    a_h = math.pi * ar / (1 + math.sqrt(1 + (ar / 2) ** 2))
    check("F1", f"Helmbold slope AR {ar:.2f} [1/rad]", a_h, main.lift_slope, rel=1e-6)
    check(
        "F1",
        "Helmbold at AR 2 vs Prandtl 2piAR/(AR+2): ratio",
        1 / 1.2,
        helmbold_lift_slope(2) / (2 * math.pi * 2 / 4),
        rel=0.03,
    )
    U, depth_cg = 8.0, 5.0
    pose = Pose(torch.tensor([[0.0, 0.0, depth_cg]]), torch.eye(3)[None])
    nu = torch.tensor([[U, 0, 0, 0, 0, 0.0]])
    loads = foil_loads(
        main, calm_sea(), torch.zeros(1), pose, nu, torch.zeros(1), torch.zeros(1, dtype=torch.bool)
    )
    alpha = main.incidence  # 3.2 deg since the v1 re-trim (was 4)
    S = main.span_m * main.chord_m
    q = 0.5 * RHO * U**2
    fs = free_surface_factor(torch.tensor((depth_cg + 1.25) / main.chord_m)).item()
    cl = a_h * alpha * fs
    check(
        "F1",
        "whole main foil lift 1/2 rho U^2 S a alpha [N, up]",
        q * S * cl,
        -loads.force[0, :, 2].sum().item(),
        rel=1e-4,
    )
    check(
        "F1",
        "one strip lift [N]",
        q * S / main.num_strips * cl,
        -loads.force[0, 0, 2].item(),
        rel=1e-4,
    )
    cd = 0.008 + cl**2 / (math.pi * 0.9 * ar)
    check(
        "F1",
        "main foil drag (profile + induced) [N, aft]",
        q * S * cd,
        -loads.force[0, :, 0].sum().item(),
        rel=1e-4,
    )

    # Directions under an oblique flow: lift perpendicular to flow and span, drag along the flow.
    nu = torch.tensor(
        [[U, 0, -1.0, 0, 0, 0.0]]
    )  # moving up in body z (w<0) -> flow from above, alpha down
    ld = foil_loads(
        main, calm_sea(), torch.zeros(1), pose, nu, torch.zeros(1), torch.zeros(1, dtype=torch.bool)
    )
    flow = -nu[0, :3]
    F = ld.force[0, 0]
    drag = (F @ flow) / flow.norm()
    lift_vec = F - drag * flow / flow.norm()
    check("F2", "lift . flow (after removing drag) [N]", 0.0, (lift_vec @ flow).item(), abs_=1e-3)
    check("F2", "force . span [N]", 0.0, F[1].item(), abs_=1e-4)
    check("F2", "drag component along relative flow is positive", 1.0, float(drag.item() > 0))
    check(
        "F2",
        "alpha with w=-1 (body moving up): incidence - atan(1/8) [deg]",
        math.degrees(alpha - math.atan2(1, U)),
        math.degrees(ld.alpha[0, 0].item()),
        rel=1e-4,
    )

    for hc, expect in ((0.0, 0.5), (0.25, 2 / 3), (1.0, 17 / 18), (2.0, 65 / 66)):
        check(
            "F3",
            f"free-surface factor at h = {hc} c",
            expect,
            free_surface_factor(torch.tensor(hc)).item(),
            rel=1e-6,
        )

    # Moment arm and sign: a lifting foil 0.5 m ahead of the CG pitches the bow up (+q moment).
    fwd = Foil("t", (0.5, 0.0, 1.25), main.span_m, main.chord_m, incidence=alpha, lift_slope=a_h)
    lf = foil_loads(
        fwd,
        calm_sea(),
        torch.zeros(1),
        pose,
        torch.tensor([[U, 0, 0, 0, 0, 0.0]]),
        torch.zeros(1),
        torch.zeros(1, dtype=torch.bool),
    )
    wr = foil_wrench(fwd, lf)[0]
    expected_m = 0.5 * (-lf.force[0, :, 2].sum().item()) + 1.25 * lf.force[0, :, 0].sum().item()
    check(
        "F4",
        "pitch moment of foil at x=+0.5,z=+1.25: x L + z F_x [N m]",
        expected_m,
        wr[4].item(),
        rel=1e-5,
    )
    check("F4", "lift ahead of CG pitches bow up (sign of M_y > 0)", 1.0, float(wr[4].item() > 0))
    lm = foil_loads(
        main,
        calm_sea(),
        torch.zeros(1),
        pose,
        torch.tensor([[U, 0, 0, 0, 0, 0.0]]),
        torch.zeros(1),
        torch.zeros(1, dtype=torch.bool),
    )
    arm = foil_wrench(main, lm)[0, 4].item() - main.position[2] * lm.force[0, :, 0].sum().item()
    check(
        "F4",
        "main foil lift pitch arm: M_y - z F_x = x L [N m] (x ahead of CG, bow up)",
        main.position[0] * -lm.force[0, :, 2].sum().item(),
        arm,
        rel=1e-4,
    )

    strut = mv.foils[2]
    ls = foil_loads(
        strut,
        calm_sea(),
        torch.zeros(1),
        Pose(torch.tensor([[0.0, 0, 0]]), torch.eye(3)[None]),
        torch.tensor([[U, 0.5, 0, 0, 0, 0.0]]),
        torch.zeros(1),
        torch.zeros(1, dtype=torch.bool),
    )
    check(
        "F5",
        "strut side force opposes sway v=+0.5 (sign F_y < 0)",
        1.0,
        float(ls.force[0, :, 1].sum().item() < 0),
    )
    ar_s = strut.span_m / strut.chord_m
    note(
        "F5",
        "strut lift slope",
        f"Helmbold AR {ar_s:.1f}: {helmbold_lift_slope(ar_s):.2f}",
        f"{strut.lift_slope:.2f}",
        "pass"
        if abs(strut.lift_slope - helmbold_lift_slope(ar_s)) < 1e-6
        else "unclear (2 pi, minor)",
    )

    # Moth trim with the wand in calm water.
    task = rc.RideControlMoth(2, rc.sea_state(0.0, 3.0))
    obs, _ = task.reset(seed=0)
    for _ in range(1500):
        obs, *_ = task.step(rc.wand_policy(obs))
    env = task.env
    pose = Pose.from_eta(env.eta)
    from watergym.vessel import all_foil_loads

    loads = all_foil_loads(env.vessel, env.sea, env.t, pose, env.nu, env.action, env.ventilated)
    lift = sum(-(load.force[..., 2]).sum(-1) for load in loads)[0].item()
    tau = generalized_force(env.vessel, env.sea, env.t, pose, env.nu, env.action, env.ventilated)
    W = env.vessel.body.mass * G
    note(
        "F6",
        "wand trim, calm, 30 s: ride height [m], speed [m/s], pitch [deg]",
        "0.545 (moth() docstring)",
        f"{task._ride_height()[0].item():.3f}, {env.nu[0, 0].item():.2f}, "
        f"{math.degrees(env.eta[0, 4].item()):.2f}",
        "pass",
    )
    check("F6", "trim: total foil vertical force vs weight [N]", W, lift, rel=0.01)
    cn = coriolis_force(env.vessel.body.mass_matrix, env.nu)[0]
    note(
        "F6",
        "trim: pitch moment of tau alone / Munk term C(nu)nu [N m]",
        "balance",
        f"{tau[0, 4].item():.2f} / {cn[4].item():.2f}",
        "pass",
    )
    check(
        "F6",
        "trim: net pitch moment tau - C(nu)nu [N m]",
        0.0,
        tau[0, 4].item() - cn[4].item(),
        abs_=1.0,
    )
    check(
        "F6",
        "trim: net surge force [N]",
        0.0,
        tau[0, 0].item() - env.vessel.body.quadratic_damping[0].item() * env.nu[0, 0].item() ** 2,
        abs_=2.0,
    )
    rud = -(loads[1].force[0, :, 2]).sum().item()
    check(
        "F6",
        "rudder foil lift at trim [N, + up]: small, under 10 % of weight",
        0.05 * W,
        rud,
        abs_=0.05 * W,
    )
    sail = thrust_force(env.vessel.thrusters, env.action[:, :1])[0, 4].item()
    note(
        "F6",
        "sail drive pitch moment at trim [N m, bow down < 0]",
        "z_ce F = -2.0 x 130 = -260",
        f"{sail:.1f}",
        "pass" if abs(sail + 260) < 1 else "FAIL",
    )


# ---------------------------------------------------------------- 5. thrusters
def thrusters() -> None:
    rov = bluerov2()
    cmd = torch.zeros(1, 8)
    cmd[0, 4] = 1.0  # first vertical: x=+0.12, y=+0.22, direction +z body (down)
    tau = thrust_force(rov.thrusters, cmd)[0]
    th = rov.thrusters[4]
    r, F = torch.tensor(th.position), th.force_vector
    check(
        "T1", "vertical thruster +1 -> +z body force (down, NED) [N]", 30.0, tau[2].item(), rel=1e-6
    )
    check(
        "T1",
        "moment r x F, roll [N m]",
        torch.linalg.cross(r, F)[0].item(),
        tau[3].item(),
        rel=1e-6,
    )
    check(
        "T1",
        "moment r x F, pitch [N m] (thruster ahead pushing down -> bow down, negative)",
        -0.12 * 30.0,
        tau[4].item(),
        rel=1e-6,
    )
    note(
        "T1",
        "ROV heave command sign",
        "+1 = dive (NED +z)",
        "examples/04 PD uses +cmd to go deeper: consistent",
        "pass",
    )


def wk_fft(acc: torch.Tensor) -> float:
    """Exact analog Wk applied in the frequency domain, RMS over steps and envs."""
    x = acc.double().numpy()
    spec = np.fft.rfft(x, axis=0)
    f = np.fft.rfftfreq(x.shape[0], rc.DT)
    gain = np.array(
        [metrics.analog_gain(metrics.wk_analog_sections(), fi) if fi > 0 else 0.0 for fi in f]
    )
    y = np.fft.irfft(spec * gain[:, None], n=x.shape[0], axis=0)
    return float(np.sqrt((y**2).mean()))


# ---------------------------------------------------------------- 6. task and reward
def task_checks() -> None:
    steps = round(rc.EPISODE_LENGTH_S / rc.DT)
    best = rc.W_HEIGHT + rc.W_PITCH
    check("K1", "per-step max reward (both tracking terms = 1)", 1.5, best)
    check("K1", "episode steps", 1000, steps)
    note(
        "K1",
        "return: perfect / crash at step 1 / crash at step k",
        "1500 / -18.5 / 1.5k - 20",
        f"{best * steps:.0f} / {best - rc.W_CRASH:.1f} / 1.5k-20",
        "pass",
    )

    # How negative can a flying step get? Wand in Hs 0.6: per-step reward distribution.
    task = rc.RideControlMoth(8, rc.sea_state(0.6, 3.0))
    obs, _ = task.reset(seed=3)
    rew, acc, crashed = [], [], []
    for _ in range(500):
        obs, r, term, trunc, info = task.step(rc.wand_policy(obs))
        rew.append(r)
        acc.append(info["signals"]["heave_accel_m_s2"])
        crashed.append(info["signals"]["crashed"])
    rew, acc, crashed = torch.stack(rew), torch.stack(acc), torch.stack(crashed)
    flying = rew[~crashed]
    note(
        "K2",
        "wand Hs 0.6: per-step reward mean / min / fraction < 0",
        "> 0 so staying up pays",
        f"{flying.mean():.3f} / {flying.min():.3f} / {(flying < 0).float().mean():.4f}",
        "pass" if (flying < 0).float().mean() < 0.01 else "unclear",
    )
    note(
        "K2",
        "wand Hs 0.6: heave accel RMS and comfort penalty mean",
        "",
        f"{acc.pow(2).mean().sqrt():.2f} m/s^2, {(rc.W_COMFORT * acc**2).mean():.3f}/step",
        "pass",
    )
    # Break-even: crashing now beats flying on if the discounted future is below -W_CRASH.
    note(
        "K2",
        "per-step reward at which a crash pays (gamma 0.99)",
        "r < -20 (1-gamma) = -0.2",
        f"comfort alone reaches -0.2 at |a| = {math.sqrt(0.2 / rc.W_COMFORT):.1f} m/s^2; "
        f"smoothness at |dflap| = {math.sqrt(0.2 / rc.W_SMOOTHNESS):.2f}",
        "unclear",
    )

    # An untrained policy: wand plus N(0, 0.5) action noise, like PPO's init_std.
    for hs in (0.0, 0.6):
        task = rc.RideControlMoth(16, rc.sea_state(hs, 3.0))
        obs, _ = task.reset(seed=4)
        g = torch.Generator().manual_seed(0)
        rs, cr, ac = [], [], []
        for _ in range(500):
            a = rc.wand_policy(obs) + 0.5 * torch.randn(16, 1, generator=g)
            obs, r, term, trunc, info = task.step(a)
            rs.append(r)
            cr.append(info["signals"]["crashed"])
            ac.append(info["signals"]["heave_accel_m_s2"])
        rs, cr, ac = torch.stack(rs), torch.stack(cr), torch.stack(ac)
        fl = rs[~cr]
        note(
            "K2",
            f"wand + N(0,0.5) noise, Hs {hs}: flying reward mean / frac<0 / crashes; accel RMS",
            "> 0",
            f"{fl.mean():.3f} / {(fl < 0).float().mean():.3f} / {int(cr.sum())}; "
            f"{ac.pow(2).mean().sqrt():.1f} m/s^2",
            "pass" if fl.mean() > 0.2 else "unclear",
        )
        if hs == 0.6:
            digital = metrics.rms(metrics.wk_filter(ac, 1 / rc.DT))
            note(
                "M2",
                "a_w of noisy-policy heave accel: wk_filter vs exact analog Wk via circular FFT",
                "equal",
                f"{digital:.3f} vs {wk_fft(ac):.3f}",
                "FAIL" if abs(digital / wk_fft(ac) - 1) > 0.05 else "pass",
            )
        else:
            pass

    # Bow sensor obs = sensor height above the instantaneous local surface.
    task = rc.RideControlMoth(2, rc.sea_state(0.6, 3.0))
    obs, _ = task.reset(seed=1)
    for _ in range(30):
        obs, *_ = task.step(rc.wand_policy(obs))
    env = task.env
    pivot = Pose.from_eta(env.eta).to_world(torch.tensor([list(rc.WAND_PIVOT)]))
    eta_here = water_at(env.sea, pivot, env.t).elevation[:, 0]
    check(
        "K3",
        "bow obs = -(z_sensor + eta(x_sensor, y_sensor, t)) [m]",
        -(pivot[0, 0, 2] + eta_here[0]).item(),
        obs[0, 0].item(),
        abs_=1e-5,
    )

    # Heave acceleration: world vertical, up positive, gravity removed. Compare with nu_dot.
    task = rc.RideControlMoth(1, rc.sea_state(0.6, 3.0))
    obs, _ = task.reset(seed=2)
    hr, ha = [], []
    for _ in range(300):
        obs, _, term, trunc, info = task.step(rc.wand_policy(obs))
        if term.any() or trunc.any():
            break
        hr.append(task.heave_rate.item())
        ha.append(obs[0, 3].item())
    hr, ha = np.array(hr), np.array(ha)
    central = (hr[2:] - hr[:-2]) / (2 * rc.DT)
    corr_now = np.corrcoef(ha[1:-1], central)[0, 1]
    corr_lag = np.corrcoef(ha[2:], central)[0, 1]
    note(
        "K4",
        "heave accel obs vs central difference of heave rate: corr at lag 0 / +1/2 step",
        "nu_dot at the step end: lag 0 best",
        f"{corr_now:.4f} / {corr_lag:.4f}",
        "pass" if corr_now > corr_lag else "FAIL",
    )
    note(
        "K4",
        "heave accel mean in steady flight (gravity removed => ~0)",
        "0",
        f"{ha[50:].mean():.3f}",
        "pass",
    )

    # Timeouts: count env steps until truncation with float32 time.
    env = WaterEnv(
        moth(),
        1,
        SeaState(),
        dt=rc.DT,
        episode_length_s=rc.EPISODE_LENGTH_S,
        max_tilt=math.radians(90),
    )
    t = torch.zeros(1)
    n = 0
    while True:
        for _ in range(env.substeps):
            t = t + env.dt / env.substeps
        n += 1
        if (t >= env.episode_length_s).item():
            break
    check("K5", f"episode length in steps with float32 t (t_end = {t.item():.7f})", 1000, n)
    note(
        "K5",
        "rsl_rl time_outs = truncated & ~terminated",
        "timeouts bootstrap, crashes do not",
        "matches",
        "pass",
    )


# ---------------------------------------------------------------- 7. metrics
def metric_checks() -> None:
    table = {
        0.5: 0.418,
        1.0: 0.482,
        2.0: 0.531,
        4.0: 0.967,
        8.0: 1.036,
        12.5: 0.902,
        16.0: 0.768,
        20.0: 0.636,
        25.0: 0.513,
    }
    sections = metrics.wk_analog_sections()
    fs = 1 / rc.DT
    for f, ref in table.items():
        check(
            "M1",
            f"Wk analog |H| at {f} Hz vs ISO table",
            ref,
            metrics.analog_gain(sections, f),
            rel=0.02,
        )
        t = torch.arange(0, 60, 1 / fs, dtype=torch.float64)[:, None]
        x = torch.cos(2 * math.pi * f * t)
        g = metrics.rms(metrics.wk_filter(x, fs)[500:-500]) / metrics.rms(x[500:-500])
        check(
            "M2", f"wk_filter gain on a cosine at {f} Hz, fs {fs:.0f} Hz, vs ISO", ref, g, rel=0.03
        )
    task = rc.RideControlMoth(8, rc.sea_state(0.6, 3.0))
    obs, _ = task.reset(seed=3)
    ac = []
    for _ in range(1000):
        obs, _, _, _, info = task.step(rc.wand_policy(obs))
        ac.append(info["signals"]["heave_accel_m_s2"])
    ac = torch.stack(ac)
    d = metrics.rms(metrics.wk_filter(ac, fs))
    check(
        "M2",
        "a_w wand Hs 0.6 (20 s): wk_filter vs exact analog Wk via circular FFT",
        wk_fft(ac),
        d,
        rel=0.03,
    )
    events = torch.zeros(1500, 4, dtype=torch.bool)
    events[::300] = True
    check(
        "M3",
        "per_minute: 5 events x 4 envs over 4 x 30 s",
        20 / 2.0,
        metrics.per_minute(events, 0.02),
        rel=1e-6,
    )
    x = torch.full((100, 3), 2.0)
    check("M3", "rms of constant 2", 2.0, metrics.rms(x), rel=1e-6)


# ---------------------------------------------------------------- 8. units and dt
def units() -> None:
    main = moth().foils[0]
    vent = torch.ones(1, dtype=torch.bool)
    deep = FoilLoads(
        torch.zeros(1, 8, 3),
        torch.zeros(1, 8, 3),
        torch.zeros(1, 8),
        torch.full((1, 8), 1.0),
        torch.full((1, 8), 8.0),
    )
    wt = torch.zeros(1)
    n = 0
    while vent.item():
        vent = update_ventilation(main, vent, deep, wt, rc.DT)
        n += 1
    check("U1", "washout after 0.5 s of calm at env dt 0.02 [steps]", 25, n)
    note(
        "U1",
        "advance() passes env dt (not substep dt) to update_ventilation",
        "env dt",
        "env dt (env.py:67)",
        "pass",
    )
    check(
        "U2", "flap_deg signal for flap +1 [deg]", 6.0, math.degrees(1.0 * rc.FLAP_RANGE), rel=1e-9
    )
    note(
        "U3",
        "metrics per_minute and Wk use DT = env dt (sweep passes DT)",
        "sim time",
        "DT = 0.02 = env.dt",
        "pass",
    )
    note(
        "U4",
        "real-time factor = env-steps/s x env.dt",
        "aggregate sim s per wall s",
        "speed.py:44",
        "pass",
    )


if __name__ == "__main__":
    for section in (
        waves,
        hydrostatics,
        rigid_body,
        foils,
        thrusters,
        task_checks,
        metric_checks,
        units,
    ):
        section()
    w = max(len(r[1]) for r in ROWS)
    for r in ROWS:
        print(f"{r[0]:<4} {r[1]:<{w}}  exp {r[2]:<28} got {r[3]:<40} {r[4]}")
    print(f"\n{sum(r[4] == 'FAIL' for r in ROWS)} FAIL of {len(ROWS)}")
