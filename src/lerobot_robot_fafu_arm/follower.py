"""LeRobot follower implementation backed by ``FafuRobotController``."""

from __future__ import annotations

import logging
import math
import time
from functools import cached_property
from typing import Any

import numpy as np
from lerobot.robots.robot import Robot

from .compat import make_cameras, read_camera_depth, read_camera_rgb
from .config import JOINT_NAMES, FafuFollowerConfig
from .kinematics import FafuArmKinematics
from .sdk import default_sdk_config_path, load_sdk

logger = logging.getLogger(__name__)


class FafuFollower(Robot):
    """Six-axis FAFU follower with a seventh gripper motor."""

    config_class = FafuFollowerConfig
    name = "fafu_follower"

    def __init__(self, config: FafuFollowerConfig):
        super().__init__(config)
        self.config = config
        self.cameras = make_cameras(config.cameras)
        self.kinematics = FafuArmKinematics(config.urdf_path)
        self._controller: Any | None = None
        self._last_joint_goal = np.zeros(len(JOINT_NAMES), dtype=np.float64)
        self._last_gripper_goal = config.gripper_min_rad

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {f"{name}.pos": float for name in (*JOINT_NAMES, "gripper")}

    @property
    def _cameras_ft(self) -> dict[str, tuple[int, int, int]]:
        features: dict[str, tuple[int, int, int]] = {}
        for name, camera_config in self.config.cameras.items():
            if getattr(camera_config, "use_rgb", True):
                features[name] = (camera_config.height, camera_config.width, 3)
            if getattr(camera_config, "use_depth", False):
                features[f"{name}_depth"] = (camera_config.height, camera_config.width, 1)
        return features

    @cached_property
    def observation_features(self) -> dict[str, type | tuple[int, int, int]]:
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        return self._controller is not None and all(camera.is_connected for camera in self.cameras.values())

    @property
    def is_calibrated(self) -> bool:
        # The hardware SDK applies the configured absolute positions and soft limits.
        return True

    def calibrate(self) -> None:
        logger.info("%s uses SDK-side absolute joint calibration; no LeRobot calibration is required", self)

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise RuntimeError(f"{self} is already connected")

        bindings = load_sdk(self.config.sdk_path)
        sdk_config = self.config.sdk_config_path or default_sdk_config_path()
        controller = bindings.controller_class(
            cfg_path=str(sdk_config),
            port=self.config.port,
            baudrate=self.config.baudrate,
            has_gripper=True,
            gripper_motor_id=self.config.gripper_motor_id,
            auto_enable=True,
            auto_polling=True,
        )
        self._controller = controller

        try:
            for camera in self.cameras.values():
                camera.connect()
            self._last_joint_goal = np.asarray(controller.get_joint_values(), dtype=np.float64)
            self._last_gripper_goal = self._read_gripper_angle()
            self.configure()
        except Exception:
            self._disconnect_after_failed_connect()
            raise

        if calibrate:
            self.calibrate()
        logger.info("%s connected", self)

    def configure(self) -> None:
        controller = self._require_controller()
        if not self.config.use_servo:
            return

        bindings = load_sdk(self.config.sdk_path)
        options = bindings.servo_options_class(
            watchdog_ms=self.config.servo_watchdog_ms,
            max_vel=self.config.servo_max_velocity,
            max_step_rad=self.config.servo_max_step_rad,
            max_lag_rad=self.config.servo_max_lag_rad,
            rate_hz=self.config.servo_rate_hz,
            use_mit=self.config.servo_use_mit,
        )
        controller.servo_start(options)

    def get_observation(self) -> dict[str, Any]:
        controller = self._require_controller()
        started_at = time.perf_counter()
        joints = np.asarray(controller.get_joint_values(), dtype=np.float64)
        if joints.shape != (len(JOINT_NAMES),):
            raise RuntimeError(f"SDK returned {joints.size} joints; expected {len(JOINT_NAMES)}")

        observation: dict[str, Any] = {
            f"{name}.pos": float(joints[index]) for index, name in enumerate(JOINT_NAMES)
        }
        observation["gripper.pos"] = self._read_gripper_angle()
        logger.debug("%s read motor state in %.1f ms", self, (time.perf_counter() - started_at) * 1e3)

        for name, camera in self.cameras.items():
            if getattr(camera, "use_rgb", True):
                observation[name] = read_camera_rgb(camera)
            if getattr(camera, "use_depth", False):
                observation[f"{name}_depth"] = read_camera_depth(camera)
        return observation

    def send_action(self, action: dict[str, Any]) -> dict[str, float]:
        controller = self._require_controller()
        current_joints = np.asarray(controller.get_joint_values(), dtype=np.float64)
        target_joints = self._last_joint_goal.copy()

        for index, name in enumerate(JOINT_NAMES):
            key = f"{name}.pos"
            if key in action:
                target_joints[index] = self._finite_scalar(action[key], key)

        target_joints = self._limit_joint_action(target_joints, current_joints)
        if self.config.use_servo:
            if not controller.servo_j(target_joints):
                reason = getattr(controller, "servo_aborted_reason", None)
                suffix = f": {reason}" if reason else ""
                raise RuntimeError(f"FAFU SDK rejected the servo action{suffix}")
        else:
            controller.move_j(
                target_joints,
                is_radians=True,
                speed=self.config.move_speed,
                block=False,
            )
        self._last_joint_goal = target_joints

        gripper_key = "gripper.pos"
        if gripper_key in action:
            current_gripper = self._read_gripper_angle()
            gripper_goal = self._finite_scalar(action[gripper_key], gripper_key)
            gripper_goal = self._limit_relative("gripper", gripper_goal, current_gripper)
            gripper_goal = float(
                np.clip(gripper_goal, self.config.gripper_min_rad, self.config.gripper_max_rad)
            )
            controller.gripper_control(
                angle=gripper_goal,
                effort=self.config.gripper_effort,
                is_radians=True,
                block=False,
            )
            self._last_gripper_goal = gripper_goal

        sent = {f"{name}.pos": float(target_joints[index]) for index, name in enumerate(JOINT_NAMES)}
        sent[gripper_key] = float(self._last_gripper_goal)
        return sent

    def disconnect(self) -> None:
        if self._controller is None:
            return

        controller = self._controller
        self._controller = None
        try:
            if getattr(controller, "is_servoing", False):
                controller.servo_end(finish_mode=self.config.joint_release)
        finally:
            try:
                controller.close_connection(
                    joint_release=self.config.joint_release,
                    gripper_release=self.config.gripper_release,
                )
            finally:
                for camera in self.cameras.values():
                    if camera.is_connected:
                        camera.disconnect()
        logger.info("%s disconnected", self)

    def _disconnect_after_failed_connect(self) -> None:
        controller, self._controller = self._controller, None
        for camera in self.cameras.values():
            try:
                if camera.is_connected:
                    camera.disconnect()
            except Exception:
                logger.exception("Failed to close camera after connection error")
        if controller is not None:
            try:
                controller.close_connection(joint_release="stop", gripper_release="brake")
            except Exception:
                logger.exception("Failed to close FAFU controller after connection error")

    def _require_controller(self) -> Any:
        if self._controller is None:
            raise RuntimeError(f"{self} is not connected")
        return self._controller

    def _read_gripper_angle(self) -> float:
        state = self._require_controller().get_gripper_state()
        return float(state.position) * math.tau

    def _limit_joint_action(self, target: np.ndarray, current: np.ndarray) -> np.ndarray:
        limited = target.copy()
        for index, name in enumerate(JOINT_NAMES):
            limited[index] = self._limit_relative(name, limited[index], current[index])
        if self.config.enforce_urdf_limits:
            lower, upper = self.kinematics.joint_limits
            limited = np.clip(limited, lower, upper)
        return limited

    def _limit_relative(self, name: str, target: float, current: float) -> float:
        setting = self.config.max_relative_target
        if setting is None:
            return float(target)
        if isinstance(setting, dict):
            limit = setting.get(name, setting.get(f"{name}.pos"))
            if limit is None:
                return float(target)
        else:
            limit = setting
        limit = float(limit)
        if limit <= 0:
            raise ValueError("max_relative_target values must be positive")
        return float(np.clip(target, current - limit, current + limit))

    @staticmethod
    def _finite_scalar(value: Any, name: str) -> float:
        result = float(value)
        if not math.isfinite(result):
            raise ValueError(f"{name} must be finite")
        return result
