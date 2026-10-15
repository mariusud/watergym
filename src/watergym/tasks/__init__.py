"""Versioned benchmark tasks. A task id names a frozen definition: change it, bump the id."""

from watergym.tasks.ride_control import RideControlMoth
from watergym.tasks.ride_control_preview import PREVIEW_TASKS, RideControlPreviewMoth
from watergym.tasks.ride_control_variants import RideControlBowOnlyMoth

TASKS = {RideControlMoth.task_id: RideControlMoth}
TASKS.update(PREVIEW_TASKS)
TASKS[RideControlBowOnlyMoth.task_id] = RideControlBowOnlyMoth


def make(task_id: str, *args: object, **kwargs: object) -> RideControlMoth:
    return TASKS[task_id](*args, **kwargs)


__all__ = ["TASKS", "RideControlMoth", "RideControlPreviewMoth", "make"]
