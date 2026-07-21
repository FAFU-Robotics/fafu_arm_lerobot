"""FAFU arm plugin for LeRobot."""

from .kinematics import (
    FafuArmKinematics,
    Pose,
    default_urdf_path,
    rotation_matrix_to_rotvec,
    rotation_vector_to_matrix,
)

__version__ = "0.2.0"

# LeRobot discovers third-party devices by importing distributions whose name
# starts with ``lerobot_robot_``. Importing the config modules performs the
# draccus registrations. Keeping the fallback narrow still allows the
# kinematics helpers and the diagnostic CLI to explain a missing LeRobot
# installation instead of failing during module discovery.
try:
    from .config import FafuFollowerConfig, FafuLeaderConfig
    from .follower import FafuFollower
    from .leader import FafuLeader
except ModuleNotFoundError as exc:
    if not (exc.name == "lerobot" or (exc.name and exc.name.startswith("lerobot."))):
        raise

__all__ = [
    "FafuArmKinematics",
    "FafuFollower",
    "FafuFollowerConfig",
    "FafuLeader",
    "FafuLeaderConfig",
    "Pose",
    "rotation_matrix_to_rotvec",
    "rotation_vector_to_matrix",
    "default_urdf_path",
]
