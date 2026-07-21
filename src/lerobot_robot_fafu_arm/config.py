"""LeRobot configuration registrations for the FAFU arm."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

from lerobot.cameras import CameraConfig
from lerobot.robots.config import RobotConfig
from lerobot.teleoperators.config import TeleoperatorConfig

JOINT_NAMES = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6")


@RobotConfig.register_subclass("fafu_follower")
@dataclass
class FafuFollowerConfig(RobotConfig):
    """Configuration used by ``--robot.type=fafu_follower``."""

    sdk_path: Path | None = None
    sdk_config_path: Path | None = None
    port: str | None = None
    baudrate: int | None = None
    gripper_motor_id: int = 7
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    use_servo: bool = True
    servo_watchdog_ms: int = 250
    servo_max_velocity: float = 1.0
    servo_max_step_rad: float = 0.10
    servo_max_lag_rad: float = 0.35
    servo_rate_hz: float = 30.0
    servo_use_mit: bool = False

    move_speed: int = 50
    max_relative_target: float | dict[str, float] | None = 0.15
    enforce_urdf_limits: bool = True
    gripper_min_rad: float = 0.0
    gripper_max_rad: float = math.radians(105.0)
    gripper_effort: int | None = 300

    joint_release: str = "stop"
    gripper_release: str = "brake"
    urdf_path: Path | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.gripper_motor_id < 1:
            raise ValueError("gripper_motor_id must be positive")
        if not 1 <= self.move_speed <= 100:
            raise ValueError("move_speed must be in [1, 100]")
        if self.gripper_min_rad >= self.gripper_max_rad:
            raise ValueError("gripper_min_rad must be lower than gripper_max_rad")
        if self.joint_release not in {"stop", "brake", "hold"}:
            raise ValueError("joint_release must be stop, brake, or hold")
        if self.gripper_release not in {"stop", "brake", "hold"}:
            raise ValueError("gripper_release must be stop, brake, or hold")


@TeleoperatorConfig.register_subclass("fafu_leader")
@dataclass
class FafuLeaderConfig(TeleoperatorConfig):
    """Configuration used by ``--teleop.type=fafu_leader``."""

    sdk_path: Path | None = None
    sdk_config_path: Path | None = None
    port: str | None = None
    baudrate: int | None = None
    gripper_motor_id: int = 7
    joint_release: str = "stop"
    gripper_release: str = "stop"

    def __post_init__(self) -> None:
        if self.gripper_motor_id < 1:
            raise ValueError("gripper_motor_id must be positive")
        if self.joint_release not in {"stop", "brake"}:
            raise ValueError("A leader arm must release joints with stop or brake")
        if self.gripper_release not in {"stop", "brake"}:
            raise ValueError("A leader gripper must use stop or brake")
