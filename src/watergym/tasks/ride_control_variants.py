"""Variants of RideControl-Moth-v1 that change only what the policy sees.

RideControlBowOnly-Moth-v1: the policy reads the bow height and its own last flap, the
signal a mechanical wand works from. Reward, terminations, critic observations and the Moth
are v1's.
"""

from torch import Tensor

from watergym.tasks.ride_control import POLICY_OBSERVATIONS, RideControlMoth

BOW_ONLY_OBSERVATIONS = ("bow_height_m", "last_flap")
BOW_ONLY_COLUMNS = [POLICY_OBSERVATIONS.index(name) for name in BOW_ONLY_OBSERVATIONS]


class RideControlBowOnlyMoth(RideControlMoth):
    task_id = "RideControlBowOnly-Moth-v1"
    observation_size = len(BOW_ONLY_OBSERVATIONS)

    def _observe(self) -> tuple[Tensor, Tensor]:
        obs, privileged = super()._observe()
        return obs[:, BOW_ONLY_COLUMNS], privileged
