"""RideControlPreview-Moth-v1: RideControl-Moth-v1 plus k range readings of the sea ahead.

The Moth, reward, terminations and the first six observations are v1's. A forward-looking
range sensor at the bow (a lidar or radar, or an array of ultrasonic sensors) adds k
readings: the height of the bow sensor above the water surface at k points ahead of the bow,
spread from PREVIEW_START_M to L along the body x axis. A reading is what the sensor gets
from its range and the IMU's attitude, so it is in the same units as v1's `bow_height_m`.

The sensor model:
- footprint: each reading averages the surface over a patch of radius slant range times
  tan(BEAM_HALF_ANGLE), sampled at its centre and four points on its rim.
- range noise: Gaussian, standard deviation RANGE_NOISE_M + RANGE_NOISE_PER_M * slant range.
- dropouts: spray blocks a reading in bursts. Each reading follows a two-state Markov chain,
  entering a dropout with DROPOUT_START_PROBABILITY per step and leaving it with
  DROPOUT_END_PROBABILITY, and the sensor holds its last good value while it is out.
Slant ranges use the nominal geometry: the bow sensor at the target ride height.

The critic gets the true, footprint-averaged readings, without noise or dropouts. With k = 0
the task is v1 exactly. The sensor's random numbers come from their own generator, so a seed
gives the same seas as v1.
"""

import math
from functools import partial

import torch
from torch import Tensor

from watergym.env import SeaState
from watergym.rigid_body import Pose
from watergym.tasks.ride_control import TARGET_RIDE_HEIGHT_M, RideControlMoth
from watergym.vessels.moth_vessel import HULL_BOTTOM_Z, WAND_PIVOT
from watergym.waves import elevation

TASK_FAMILY = "RideControlPreview-Moth-v1"

PREVIEW_START_M = 0.5
RANGE_NOISE_M = 0.01
RANGE_NOISE_PER_M = 0.005
BEAM_HALF_ANGLE = math.radians(2.0)
# A burst lasts 1 / 0.2 = 5 steps (0.1 s) on average; 0.02 / (0.02 + 0.2) = 9 % of readings
# are lost.
DROPOUT_START_PROBABILITY = 0.02
DROPOUT_END_PROBABILITY = 0.2
# The bow sensor's height above the mean water level when the Moth flies at its target.
NOMINAL_SENSOR_HEIGHT_M = HULL_BOTTOM_Z - WAND_PIVOT[2] + TARGET_RIDE_HEIGHT_M

PREVIEW_COUNTS = (0, 1, 2, 4, 8)
PREVIEW_LENGTHS_M = (1.0, 2.0, 4.0, 8.0)


def task_id(k: int, length_m: float) -> str:
    """Without readings the length means nothing, so every k = 0 task is one id."""
    if k == 0:
        return f"{TASK_FAMILY}-k0"
    return f"{TASK_FAMILY}-k{k}-L{length_m:g}"


def preview_distances(k: int, length_m: float) -> Tensor:
    """Distances ahead of the bow sensor [k]. One sensor looks the full length L ahead."""
    if k == 1:
        return torch.tensor([length_m])
    return torch.linspace(PREVIEW_START_M, length_m, k)


def slant_ranges(distances: Tensor) -> Tensor:
    return torch.sqrt(distances**2 + NOMINAL_SENSOR_HEIGHT_M**2)


def footprint_points(distances: Tensor) -> Tensor:
    """Body-frame points [k, 5, 3] each reading averages: the patch centre on the body x axis
    ahead of the bow sensor, and four points on the patch rim."""
    radius = slant_ranges(distances) * math.tan(BEAM_HALF_ANGLE)
    pattern = torch.tensor([[0.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
    xy = pattern * radius[:, None, None]
    xy[..., 0] += distances[:, None]
    points = torch.cat((xy, torch.zeros(*xy.shape[:-1], 1)), dim=-1)
    return points + torch.tensor(WAND_PIVOT)


class RideControlPreviewMoth(RideControlMoth):
    """RideControl-Moth-v1 with k preview readings appended to the policy's and the
    critic's observations. obs = [v1 obs (6), noisy readings (k)]; privileged obs =
    [v1 privileged obs (16), true readings (k)]."""

    def __init__(
        self,
        num_envs: int,
        sea: SeaState | None = None,
        device: str | torch.device = "cpu",
        k: int = 2,
        length_m: float = 8.0,
    ) -> None:
        super().__init__(num_envs, sea, device)
        self.task_id = task_id(k, length_m)
        self.k = k
        self.sensor_point = torch.tensor([WAND_PIVOT], device=self.device)
        self.distances = preview_distances(k, length_m)
        self.footprint = footprint_points(self.distances).reshape(-1, 3).to(self.device)
        self.noise_std = (RANGE_NOISE_M + RANGE_NOISE_PER_M * slant_ranges(self.distances)).to(
            self.device
        )
        self.observation_size = super().observation_size + k
        self.privileged_observation_size = super().privileged_observation_size + k
        self.sensor_generator = torch.Generator().manual_seed(0)
        self.true_preview = torch.zeros(num_envs, k, device=self.device)
        self.preview = torch.zeros(num_envs, k, device=self.device)
        self.dropped = torch.zeros(num_envs, k, dtype=torch.bool, device=self.device)

    def reset(self, seed: int | None = None) -> tuple[Tensor, dict]:
        if seed is not None:
            self.sensor_generator.manual_seed(seed)
        return super().reset(seed)

    def true_readings(self) -> Tensor:
        """Bow sensor height above the footprint-averaged surface at each point [envs, k]."""
        env = self.env
        pose = Pose.from_eta(env.eta)
        sensor_z = pose.to_world(self.sensor_point)[:, 0, 2]
        surface = elevation(env.sea, pose.to_world(self.footprint), env.t)
        return -sensor_z[:, None] - surface.view(self.num_envs, self.k, -1).mean(-1)

    def _uniform(self) -> Tensor:
        return torch.rand(self.num_envs, self.k, generator=self.sensor_generator).to(self.device)

    def _noisy(self, readings: Tensor) -> Tensor:
        noise = torch.randn(self.num_envs, self.k, generator=self.sensor_generator)
        return readings + self.noise_std * noise.to(self.device)

    def _read_preview(self) -> None:
        """Advance the sensor one step: new true readings, dropouts, noise."""
        self.true_preview = self.true_readings()
        chance = self._uniform()
        self.dropped = torch.where(
            self.dropped, chance >= DROPOUT_END_PROBABILITY, chance < DROPOUT_START_PROBABILITY
        )
        self.preview = torch.where(self.dropped, self.preview, self._noisy(self.true_preview))

    def _after_physics(self, env) -> Tensor:
        if self.k:
            self._read_preview()
        return super()._after_physics(env)

    def _restart(self, env_ids: Tensor) -> None:
        super()._restart(env_ids)
        if self.k:
            fresh = self.true_readings()
            self.true_preview[env_ids] = fresh[env_ids]
            self.preview[env_ids] = self._noisy(fresh)[env_ids]
            self.dropped[env_ids] = False

    def _observe(self) -> tuple[Tensor, Tensor]:
        obs, privileged = super()._observe()
        return (
            torch.cat((obs, self.preview), dim=-1),
            torch.cat((privileged, self.true_preview), dim=-1),
        )


PREVIEW_TASKS = {
    task_id(k, length_m): partial(RideControlPreviewMoth, k=k, length_m=length_m)
    for k in PREVIEW_COUNTS
    for length_m in PREVIEW_LENGTHS_M
}
