"""RideControl-Moth-v0: hold a foiling Moth at a set ride height across sea states.

The policy moves the main-foil flap; the sail is held at the wand example's 0.65 so speed is
not a control. Everything below is frozen for v0. Changing any constant, the reward, the
observations or the terminations makes a new task version.

The env's reward_fn hook runs after the physics step and before finished envs reset, so it
is where this task reads the state: reward, observations, termination causes and metric
signals all come from `_after_physics`.
"""

import math

import torch
from torch import Tensor

from watergym.env import SeaState, WaterEnv
from watergym.rigid_body import Pose
from watergym.vessels import moth
from watergym.vessels.moth_vessel import FLAP_RANGE, HULL_BOTTOM_Z, WAND_PIVOT, wand_flap
from watergym.waves import elevation

TASK_ID = "RideControl-Moth-v0"

DT = 0.02
EPISODE_LENGTH_S = 20.0
SAIL_COMMAND = 0.65
TARGET_RIDE_HEIGHT_M = 0.6
MAX_PITCH = math.radians(15)
SPREADING = 10.0
NUM_WAVE_COMPONENTS = 48
# Held out: train on any other seeds.
EVAL_SEEDS = (20261001, 20261002, 20261003)

W_HEIGHT, HEIGHT_SCALE_M = 1.0, 0.1
W_PITCH, PITCH_SCALE = 0.5, math.radians(2)
W_EFFORT = 0.05
W_SMOOTHNESS = 0.5
W_COMFORT = 0.01
W_CRASH = 20.0

POLICY_OBSERVATIONS = (
    "bow_height_m",  # ultrasonic or wand: bow sensor to the water directly below it
    "pitch_rad",  # IMU attitude
    "pitch_rate_rad_s",  # gyro
    "heave_accel_m_s2",  # accelerometer, up positive, gravity removed
    "speed_m_s",  # GPS or log
    "last_flap",  # the previous action
)
PRIVILEGED_OBSERVATIONS = (
    *POLICY_OBSERVATIONS,
    "ride_height_m",  # hull bottom above the mean water level, the tracked quantity
    "clearance_m",  # hull bottom above the local surface
    "heave_rate_m_s",  # up positive
    "main_foil_depth_m",  # below the local surface
    "main_foil_ventilated",
    "rudder_foil_ventilated",
    "hs_m",
    "tp_s",
    "cos_heading",
    "sin_heading",
)

# Body-frame points the task measures at: bow sensor, hull bottom under the CG, main foil.
BOW_SENSOR, UNDER_CG, MAIN_FOIL = 0, 1, 2


def sea_state(hs: float, tp: float, heading_rad: float = math.pi) -> SeaState:
    """The task's sea at one point of the benchmark grid. heading_rad = pi is head seas."""
    return SeaState(
        hs=hs,
        tp=tp,
        heading_rad=heading_rad,
        spreading=SPREADING,
        num_components=NUM_WAVE_COMPONENTS,
    )


def ride_control_reward(
    height_error: Tensor,
    pitch: Tensor,
    flap: Tensor,
    flap_change: Tensor,
    heave_accel: Tensor,
    crashed: Tensor,
) -> Tensor:
    """Per-step reward. The tracking terms are bounded in [0, 1] so staying up always pays."""
    return (
        W_HEIGHT * torch.exp(-((height_error / HEIGHT_SCALE_M) ** 2))
        + W_PITCH * torch.exp(-((pitch / PITCH_SCALE) ** 2))
        - W_EFFORT * flap**2
        - W_SMOOTHNESS * flap_change**2
        - W_COMFORT * heave_accel**2
        - W_CRASH * crashed.float()
    )


def wand_policy(obs: Tensor) -> Tensor:
    """The Moth's mechanical wand, reading the bow sensor as its pivot height."""
    return (wand_flap(obs[:, 0]) / FLAP_RANGE)[:, None]


def zero_policy(obs: Tensor) -> Tensor:
    return torch.zeros(obs.shape[0], 1, device=obs.device)


class RideControlMoth:
    """A batch of Moths with the RideControl-Moth-v0 reward, observations and terminations.

    step(flap [envs, 1]) -> (obs, reward, terminated, truncated, info). info holds
    "privileged_obs" for a critic, "final_obs" for envs that just finished, and per-step
    "signals" for the metrics in watergym.tasks.metrics.
    """

    task_id = TASK_ID
    action_size = 1
    observation_size = len(POLICY_OBSERVATIONS)
    privileged_observation_size = len(PRIVILEGED_OBSERVATIONS)

    def __init__(
        self, num_envs: int, sea: SeaState | None = None, device: str | torch.device = "cpu"
    ) -> None:
        self.sea_state = sea or sea_state(0.0, 3.0)
        self.env = WaterEnv(
            moth(),
            num_envs,
            self.sea_state,
            dt=DT,
            episode_length_s=EPISODE_LENGTH_S,
            max_tilt=MAX_PITCH,
            reward_fn=self._after_physics,
            termination_fn=lambda env: self._signals["crashed"],
            device=device,
        )
        self.num_envs = num_envs
        self.device = self.env.device
        hull = self.env.vessel.hull
        hull_bottoms = hull.centers + torch.stack(
            (torch.zeros_like(hull.heights), torch.zeros_like(hull.heights), hull.heights / 2),
            dim=-1,
        )
        main_foil = self.env.vessel.foils[0].position
        self.points = torch.cat(
            (torch.tensor([WAND_PIVOT, (0.0, 0.0, HULL_BOTTOM_Z), main_foil]), hull_bottoms)
        ).to(self.device)
        self.lifting_foils = [i for i, f in enumerate(self.env.vessel.foils) if f.is_horizontal]
        self.last_flap = torch.zeros(num_envs, device=self.device)
        self.heave_rate = torch.zeros(num_envs, device=self.device)
        self.heave_accel = torch.zeros(num_envs, device=self.device)
        self.was_ventilated = torch.zeros_like(self.env.ventilated)

    def reset(self, seed: int | None = None) -> tuple[Tensor, dict]:
        self.env.reset(seed)
        self._restart(torch.arange(self.num_envs, device=self.device))
        obs, privileged = self._observe()
        return obs, {"privileged_obs": privileged}

    def step(self, flap: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, dict]:
        flap = flap.to(self.device).clamp(-1, 1).reshape(self.num_envs)
        sail = torch.full_like(flap, SAIL_COMMAND)
        _, reward, terminated, truncated, _ = self.env.step(torch.stack((sail, flap), -1))

        signals = self._signals
        done = (terminated | truncated).nonzero().squeeze(-1)
        if len(done):
            self._restart(done)

        obs, privileged = self._observe()
        info = {
            "privileged_obs": privileged,
            "final_obs": self._final_obs,
            "signals": signals,
        }
        return obs, reward, terminated, truncated, info

    def _after_physics(self, env: WaterEnv) -> Tensor:
        """The env's reward_fn: runs on the post-step state before any env resets."""
        flap = env.action[:, 1]
        heave_rate = self._heave_rate()
        self.heave_accel = (heave_rate - self.heave_rate) / DT
        self.heave_rate = heave_rate

        heights = self._heights_above_surface()
        ride_height = self._ride_height()
        touchdown = heights[:, 3:].amin(-1) < 0
        ventilation_crash = env.ventilated[:, 0]
        pitch = env.eta[:, 4]
        too_tilted = pitch.abs() > MAX_PITCH
        crashed = touchdown | ventilation_crash | too_tilted

        lifting = env.ventilated[:, self.lifting_foils]
        onsets = (lifting & ~self.was_ventilated[:, self.lifting_foils]).sum(-1)
        self.was_ventilated = env.ventilated.clone()

        height_error = ride_height - TARGET_RIDE_HEIGHT_M
        reward = ride_control_reward(
            height_error, pitch, flap, flap - self.last_flap, self.heave_accel, crashed
        )
        self.last_flap = flap.clone()
        self._final_obs, _ = self._observe()
        self._signals = {
            "height_error_m": height_error,
            "flap_deg": torch.rad2deg(flap * FLAP_RANGE),
            "heave_accel_m_s2": self.heave_accel.clone(),  # _restart zeroes it in place
            "touchdown": touchdown,
            "ventilation_crash": ventilation_crash,
            "too_tilted": too_tilted,
            "crashed": crashed,
            "ventilation_onsets": onsets,
        }
        return reward

    def _restart(self, env_ids: Tensor) -> None:
        self.last_flap[env_ids] = 0.0
        self.heave_rate[env_ids] = self._heave_rate()[env_ids]
        self.heave_accel[env_ids] = 0.0
        self.was_ventilated[env_ids] = False

    def _heave_rate(self) -> Tensor:
        """World vertical velocity of the CG, up positive."""
        rotation = Pose.from_eta(self.env.eta).rotation
        return -(rotation[:, 2] * self.env.nu[:, :3]).sum(-1)

    def _ride_height(self) -> Tensor:
        world = Pose.from_eta(self.env.eta).to_world(self.points[UNDER_CG : UNDER_CG + 1])
        return -world[:, 0, 2]

    def _heights_above_surface(self) -> Tensor:
        """Height of every task point above the local water surface [envs, points]."""
        env = self.env
        world = Pose.from_eta(env.eta).to_world(self.points)
        return -(world[..., 2] + elevation(env.sea, world, env.t))

    def _observe(self) -> tuple[Tensor, Tensor]:
        env = self.env
        heights = self._heights_above_surface()
        obs = torch.stack(
            (
                heights[:, BOW_SENSOR],
                env.eta[:, 4],
                env.nu[:, 4],
                self.heave_accel,
                env.nu[:, 0],
                self.last_flap,
            ),
            dim=-1,
        )
        s = self.sea_state
        sea = torch.tensor(
            [s.hs, s.tp, math.cos(s.heading_rad), math.sin(s.heading_rad)], device=self.device
        ).expand(self.num_envs, 4)
        privileged = torch.cat(
            (
                obs,
                torch.stack(
                    (
                        self._ride_height(),
                        heights[:, UNDER_CG],
                        self.heave_rate,
                        -heights[:, MAIN_FOIL],
                        env.ventilated[:, 0].float(),
                        env.ventilated[:, 1].float(),
                    ),
                    dim=-1,
                ),
                sea,
            ),
            dim=-1,
        )
        return obs, privileged
