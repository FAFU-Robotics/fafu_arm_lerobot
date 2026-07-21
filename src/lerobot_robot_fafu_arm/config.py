"""LeRobot configuration registrations for the FAFU arm."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

from lerobot.cameras import CameraConfig
from lerobot.robots.config import RobotConfig
from lerobot.teleoperators.config import TeleoperatorConfig

from .representation import (
    ACTION_MODES,
    CARTESIAN_CONTROL_MODES,
    OBSERVATION_MODES,
    ActionMode,
    ObservationMode,
)


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
    action_mode: ActionMode = "joint"
    all_control_source: ActionMode = "joint"
    observation_mode: ObservationMode = "all"
    record_joint_velocity: bool = True
    record_motor_effort: bool = False
    delta_reset_timeout_s: float = 0.5
    max_ee_translation_step_m: float | None = 0.03
    max_ee_rotation_step_rad: float | None = 0.20
    ee_workspace_min: tuple[float, float, float] | None = None
    ee_workspace_max: tuple[float, float, float] | None = None

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
        if self.action_mode not in ACTION_MODES:
            raise ValueError(f"action_mode must be one of {sorted(ACTION_MODES)}")
        valid_sources = {"joint", *CARTESIAN_CONTROL_MODES}
        if self.all_control_source not in valid_sources:
            raise ValueError(f"all_control_source must be one of {sorted(valid_sources)}")
        if self.observation_mode not in OBSERVATION_MODES:
            raise ValueError(f"observation_mode must be one of {sorted(OBSERVATION_MODES)}")
        if self.delta_reset_timeout_s <= 0:
            raise ValueError("delta_reset_timeout_s must be positive")
        for name, value in (
            ("max_ee_translation_step_m", self.max_ee_translation_step_m),
            ("max_ee_rotation_step_rad", self.max_ee_rotation_step_rad),
        ):
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive or None")
        if (self.ee_workspace_min is None) != (self.ee_workspace_max is None):
            raise ValueError("ee_workspace_min and ee_workspace_max must be set together")
        if self.ee_workspace_min is not None and self.ee_workspace_max is not None:
            bounds = (*self.ee_workspace_min, *self.ee_workspace_max)
            if (
                len(self.ee_workspace_min) != 3
                or len(self.ee_workspace_max) != 3
                or not all(math.isfinite(value) for value in bounds)
                or any(
                    low >= high
                    for low, high in zip(self.ee_workspace_min, self.ee_workspace_max, strict=True)
                )
            ):
                raise ValueError("EE workspace bounds must contain three finite, ordered min/max values")
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
    action_mode: ActionMode = "joint"
    delta_reset_timeout_s: float = 0.5
    urdf_path: Path | None = None
    joint_release: str = "stop"
    gripper_release: str = "stop"

    def __post_init__(self) -> None:
        if self.gripper_motor_id < 1:
            raise ValueError("gripper_motor_id must be positive")
        if self.action_mode not in ACTION_MODES:
            raise ValueError(f"action_mode must be one of {sorted(ACTION_MODES)}")
        if self.delta_reset_timeout_s <= 0:
            raise ValueError("delta_reset_timeout_s must be positive")
        if self.joint_release not in {"stop", "brake"}:
            raise ValueError("A leader arm must release joints with stop or brake")
        if self.gripper_release not in {"stop", "brake"}:
            raise ValueError("A leader gripper must use stop or brake")
