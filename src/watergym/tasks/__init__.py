"""Versioned benchmark tasks. A task id names a frozen definition: change it, bump the id."""

from watergym.tasks.ride_control import RideControlMoth

TASKS = {RideControlMoth.task_id: RideControlMoth}


def make(task_id: str, *args: object, **kwargs: object) -> RideControlMoth:
    return TASKS[task_id](*args, **kwargs)


__all__ = ["TASKS", "RideControlMoth", "make"]
