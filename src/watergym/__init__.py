"""WaterGym: batched vessels in waves, in plain PyTorch."""

from watergym.current import Current
from watergym.env import SeaState, WaterEnv
from watergym.foils import Foil
from watergym.geometry import box_mesh, box_samples
from watergym.rigid_body import RigidBody
from watergym.vessel import Thruster, Vessel
from watergym.waves import Sea, elevation, make_sea, orbital_velocity, regular_wave
from watergym.wind import Wind, WindState, make_wind

__all__ = [
    "Current",
    "Foil",
    "RigidBody",
    "Sea",
    "SeaState",
    "Thruster",
    "Vessel",
    "WaterEnv",
    "Wind",
    "WindState",
    "box_mesh",
    "box_samples",
    "elevation",
    "make_sea",
    "make_wind",
    "orbital_velocity",
    "regular_wave",
]
