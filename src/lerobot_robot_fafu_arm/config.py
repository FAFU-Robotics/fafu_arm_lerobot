"""LeRobot configuration registrations for the FAFU arm."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from numbers import Integral, Real
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

_MAX_DELTA_RESET_TIMEOUT_S = 10.0
_MAX_EE_TRANSLATION_STEP_M = 0.25
_MAX_EE_ROTATION_STEP_RAD = math.pi
_MAX_WORKSPACE_COORDINATE_M = 2.0
_MAX_GRIPPER_ANGLE_RAD = math.tau
_MIN_SERVO_WATCHDOG_MS = 30
_MAX_SERVO_WATCHDOG_MS = 1_000
_MAX_SERVO_VELOCITY_RAD_S = 5.0
_MAX_SERVO_STEP_RAD = 0.5
_MAX_SERVO_LAG_RAD = 1.0
_MIN_SERVO_RATE_HZ = 10.0
_MAX_SERVO_RATE_HZ = 200.0
_MAX_SERVO_LAG_ABORT_CONSECUTIVE = 100
_MAX_RELATIVE_TARGET_RAD = 0.5
_MAX_GRIPPER_EFFORT = 32_767


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number, not {type(value).__name__}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _bounded_positive(value: object, name: str, maximum: float) -> float:
    result = _finite_number(value, name)
    if not 0 < result <= maximum:
        raise ValueError(f"{name} must be in (0, {maximum}]")
    return result


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
    servo_feedforward_vel: bool = False
    servo_lag_abort_consecutive: int = 5
    servo_use_mit: bool = False

    move_speed: int = 50
    max_relative_target: float | dict[str, float] | None = 0.15
    enforce_urdf_limits: bool = True
    strict_action_features: bool = True
    write_sent_action_back: bool = True
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
        _bounded_positive(
            self.delta_reset_timeout_s,
            "delta_reset_timeout_s",
            _MAX_DELTA_RESET_TIMEOUT_S,
        )
        if self.max_ee_translation_step_m is not None:
            _bounded_positive(
                self.max_ee_translation_step_m,
                "max_ee_translation_step_m",
                _MAX_EE_TRANSLATION_STEP_M,
            )
        if self.max_ee_rotation_step_rad is not None:
            _bounded_positive(
                self.max_ee_rotation_step_rad,
                "max_ee_rotation_step_rad",
                _MAX_EE_ROTATION_STEP_RAD,
            )
        if (self.ee_workspace_min is None) != (self.ee_workspace_max is None):
            raise ValueError("ee_workspace_min and ee_workspace_max must be set together")
        if self.ee_workspace_min is not None and self.ee_workspace_max is not None:
            if len(self.ee_workspace_min) != 3 or len(self.ee_workspace_max) != 3:
                raise ValueError("EE workspace bounds must contain three finite, ordered min/max values")
            minimum = tuple(
                _finite_number(value, f"ee_workspace_min[{index}]")
                for index, value in enumerate(self.ee_workspace_min)
            )
            maximum = tuple(
                _finite_number(value, f"ee_workspace_max[{index}]")
                for index, value in enumerate(self.ee_workspace_max)
            )
            if any(abs(value) > _MAX_WORKSPACE_COORDINATE_M for value in (*minimum, *maximum)):
                raise ValueError(
                    "EE workspace coordinates must be within "
                    f"[-{_MAX_WORKSPACE_COORDINATE_M}, {_MAX_WORKSPACE_COORDINATE_M}] m"
                )
            if any(low >= high for low, high in zip(minimum, maximum, strict=True)):
                raise ValueError("EE workspace bounds must contain three finite, ordered min/max values")

        if isinstance(self.servo_watchdog_ms, bool) or not isinstance(self.servo_watchdog_ms, Integral):
            raise ValueError("servo_watchdog_ms must be an integer")
        if not _MIN_SERVO_WATCHDOG_MS <= self.servo_watchdog_ms <= _MAX_SERVO_WATCHDOG_MS:
            raise ValueError(
                "servo_watchdog_ms must be in "
                f"[{_MIN_SERVO_WATCHDOG_MS}, {_MAX_SERVO_WATCHDOG_MS}]; disabling it is unsafe"
            )
        _bounded_positive(
            self.servo_max_velocity,
            "servo_max_velocity",
            _MAX_SERVO_VELOCITY_RAD_S,
        )
        _bounded_positive(self.servo_max_step_rad, "servo_max_step_rad", _MAX_SERVO_STEP_RAD)
        _bounded_positive(self.servo_max_lag_rad, "servo_max_lag_rad", _MAX_SERVO_LAG_RAD)
        servo_rate = _finite_number(self.servo_rate_hz, "servo_rate_hz")
        if not _MIN_SERVO_RATE_HZ <= servo_rate <= _MAX_SERVO_RATE_HZ:
            raise ValueError(f"servo_rate_hz must be in [{_MIN_SERVO_RATE_HZ}, {_MAX_SERVO_RATE_HZ}]")
        if not isinstance(self.servo_feedforward_vel, bool):
            raise ValueError("servo_feedforward_vel must be a boolean")
        if isinstance(self.servo_lag_abort_consecutive, bool) or not isinstance(
            self.servo_lag_abort_consecutive, Integral
        ):
            raise ValueError("servo_lag_abort_consecutive must be an integer")
        if not 0 <= self.servo_lag_abort_consecutive <= _MAX_SERVO_LAG_ABORT_CONSECUTIVE:
            raise ValueError(
                "servo_lag_abort_consecutive must be in "
                f"[0, {_MAX_SERVO_LAG_ABORT_CONSECUTIVE}]"
            )

        if self.max_relative_target is not None:
            if isinstance(self.max_relative_target, dict):
                if not self.max_relative_target:
                    raise ValueError("max_relative_target dict must not be empty")
                for name, value in self.max_relative_target.items():
                    if not isinstance(name, str) or not name:
                        raise ValueError("max_relative_target keys must be non-empty strings")
                    _bounded_positive(
                        value,
                        f"max_relative_target[{name!r}]",
                        _MAX_RELATIVE_TARGET_RAD,
                    )
            else:
                _bounded_positive(
                    self.max_relative_target,
                    "max_relative_target",
                    _MAX_RELATIVE_TARGET_RAD,
                )

        if self.gripper_motor_id < 1:
            raise ValueError("gripper_motor_id must be positive")
        if not 1 <= self.move_speed <= 100:
            raise ValueError("move_speed must be in [1, 100]")
        gripper_min = _finite_number(self.gripper_min_rad, "gripper_min_rad")
        gripper_max = _finite_number(self.gripper_max_rad, "gripper_max_rad")
        if not 0 <= gripper_min < gripper_max <= _MAX_GRIPPER_ANGLE_RAD:
            raise ValueError(f"gripper bounds must satisfy 0 <= min < max <= {_MAX_GRIPPER_ANGLE_RAD}")
        if self.gripper_effort is not None:
            if isinstance(self.gripper_effort, bool) or not isinstance(self.gripper_effort, Integral):
                raise ValueError("gripper_effort must be an integer or None")
            if not 0 <= self.gripper_effort <= _MAX_GRIPPER_EFFORT:
                raise ValueError(f"gripper_effort must be in [0, {_MAX_GRIPPER_EFFORT}] or None")
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
