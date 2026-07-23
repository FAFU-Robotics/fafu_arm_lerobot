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
from .config import FafuFollowerConfig
from .kinematics import FafuArmKinematics, Pose
from .representation import (
    EE_COMPONENTS,
    JOINT_NAMES,
    action_features,
    apply_pose_delta,
    delta_action,
    delta_from_action,
    joint_action,
    limit_vector_norm,
    pose_action,
    pose_delta,
    pose_from_action,
)
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
        self._last_observation_pose: Pose | None = None
        self._last_observation_time: float | None = None

    @property
    def _state_ft(self) -> dict[str, type]:
        features: dict[str, type] = {}
        if self.config.observation_mode in {"joint", "all"}:
            features.update({f"{name}.pos": float for name in (*JOINT_NAMES, "gripper")})
            if self.config.record_joint_velocity:
                features.update({f"{name}.vel": float for name in (*JOINT_NAMES, "gripper")})
            if self.config.record_motor_effort:
                features.update({f"{name}.effort": float for name in (*JOINT_NAMES, "gripper")})

        if self.config.observation_mode in {"ee_pose", "all"}:
            features.update({f"ee.{name}": float for name in EE_COMPONENTS})
            features["gripper.pos"] = float

        if self.config.observation_mode == "all":
            features.update({f"ee_delta.{name}": float for name in EE_COMPONENTS})

        return features

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
        return {**self._state_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return action_features(self.config.action_mode)

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
            self._last_observation_pose = None
            self._last_observation_time = None
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
            is_radians=True,
            rate_hz=self.config.servo_rate_hz,
            feedforward_vel=self.config.servo_feedforward_vel,
            lag_abort_consecutive=self.config.servo_lag_abort_consecutive,
            use_mit=self.config.servo_use_mit,
        )
        controller.servo_start(options)

    def get_observation(self) -> dict[str, Any]:
        controller = self._require_controller()
        started_at = time.perf_counter()
        joints = np.asarray(controller.get_joint_values(), dtype=np.float64)
        if joints.shape != (len(JOINT_NAMES),):
            raise RuntimeError(f"SDK returned {joints.size} joints; expected {len(JOINT_NAMES)}")

        gripper_state = controller.get_gripper_state()
        gripper_position = self._gripper_angle(gripper_state)
        observation: dict[str, Any] = {}

        if self.config.observation_mode in {"joint", "all"}:
            observation.update(
                {f"{name}.pos": float(joints[index]) for index, name in enumerate(JOINT_NAMES)}
            )
            observation["gripper.pos"] = gripper_position

            if self.config.record_joint_velocity:
                velocities = np.asarray(controller.get_joint_velocities(), dtype=np.float64)
                if velocities.shape != (len(JOINT_NAMES),):
                    raise RuntimeError(
                        f"SDK returned {velocities.size} joint velocities; expected {len(JOINT_NAMES)}"
                    )
                for index, name in enumerate(JOINT_NAMES):
                    observation[f"{name}.vel"] = self._finite_scalar(velocities[index], f"{name}.vel")
                observation["gripper.vel"] = (
                    self._finite_scalar(gripper_state.velocity, "gripper.vel") * math.tau
                )

            if self.config.record_motor_effort:
                states = controller.get_motor_states()
                for name, motor_id in zip(JOINT_NAMES, controller.joint_motor_ids, strict=True):
                    state = states.get(motor_id)
                    if state is None:
                        raise RuntimeError(f"SDK did not return motor state for {name} (id={motor_id})")
                    observation[f"{name}.effort"] = self._finite_scalar(state.torque, f"{name}.effort")
                observation["gripper.effort"] = self._finite_scalar(gripper_state.torque, "gripper.effort")

        if self.config.observation_mode in {"ee_pose", "all"}:
            pose = self.kinematics.forward(joints)
            observation.update(pose_action(pose, gripper_position))

            if self.config.observation_mode == "all":
                now = time.perf_counter()
                if (
                    self._last_observation_pose is None
                    or self._last_observation_time is None
                    or now - self._last_observation_time > self.config.delta_reset_timeout_s
                ):
                    translation_delta = np.zeros(3, dtype=np.float64)
                    rotation_delta = np.zeros(3, dtype=np.float64)
                else:
                    translation_delta, rotation_delta = pose_delta(self._last_observation_pose, pose)
                observation.update(delta_action(translation_delta, rotation_delta, gripper_position))
                self._last_observation_pose = pose
                self._last_observation_time = now
        logger.debug("%s read motor state in %.1f ms", self, (time.perf_counter() - started_at) * 1e3)

        for name, camera in self.cameras.items():
            if getattr(camera, "use_rgb", True):
                observation[name] = read_camera_rgb(camera)
            if getattr(camera, "use_depth", False):
                observation[f"{name}_depth"] = read_camera_depth(camera)
        return observation

    def send_action(self, action: dict[str, Any]) -> dict[str, float]:
        controller = self._require_controller()
        validated_action = self._validate_action(action)
        current_joints = np.asarray(controller.get_joint_values(), dtype=np.float64)
        if current_joints.shape != (len(JOINT_NAMES),):
            raise RuntimeError(f"SDK returned {current_joints.size} joints; expected {len(JOINT_NAMES)}")
        if not np.all(np.isfinite(current_joints)):
            raise RuntimeError("SDK returned non-finite joint positions")
        current_pose = None if self.config.action_mode == "joint" else self.kinematics.forward(current_joints)
        control_mode = (
            self.config.all_control_source if self.config.action_mode == "all" else self.config.action_mode
        )

        if control_mode == "joint":
            target_joints = self._last_joint_goal.copy()
            for index, name in enumerate(JOINT_NAMES):
                key = f"{name}.pos"
                if key in validated_action:
                    target_joints[index] = validated_action[key]
        else:
            if current_pose is None:
                raise RuntimeError("Cartesian action mode requires a current EE pose")
            target_joints = self._cartesian_joint_target(
                validated_action,
                control_mode=control_mode,
                current_joints=current_joints,
                current_pose=current_pose,
            )

        target_joints = self._limit_joint_action(target_joints, current_joints)

        # Validate and limit every actuator target before sending anything. Hardware
        # writes cannot be transactional, but malformed gripper input must not allow
        # the arm command to be sent first.
        gripper_key = "gripper.pos"
        should_send_gripper = gripper_key in validated_action
        gripper_goal = float(self._last_gripper_goal)
        if should_send_gripper:
            current_gripper = self._read_gripper_angle()
            gripper_goal = self._limit_relative("gripper", validated_action[gripper_key], current_gripper)
            gripper_goal = float(
                np.clip(gripper_goal, self.config.gripper_min_rad, self.config.gripper_max_rad)
            )

        if self.config.action_mode == "joint":
            sent_action = joint_action(target_joints, gripper_goal)
        else:
            if current_pose is None:
                raise RuntimeError("Cartesian action mode requires a current EE pose")
            actual_pose = self.kinematics.forward(target_joints)
            sent_action = self._format_sent_action(
                target_joints,
                current_pose=current_pose,
                actual_pose=actual_pose,
                gripper=gripper_goal,
            )

        # This is the final validation boundary before either actuator can be
        # commanded. Keep it after IK, limiting, FK, and action formatting so a
        # malformed result from any of those stages cannot reach the SDK.
        target_joints = np.asarray(target_joints, dtype=np.float64)
        if target_joints.shape != (len(JOINT_NAMES),):
            raise RuntimeError(f"Target must contain exactly {len(JOINT_NAMES)} joint positions")
        if not np.all(np.isfinite(target_joints)):
            raise RuntimeError("Target joint positions must be finite; no motion command was sent")
        if not math.isfinite(gripper_goal):
            raise RuntimeError("Target gripper position must be finite; no motion command was sent")

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

        if should_send_gripper:
            controller.gripper_control(
                angle=gripper_goal,
                effort=self.config.gripper_effort,
                is_radians=True,
                block=False,
            )
            self._last_gripper_goal = gripper_goal

        if self.config.write_sent_action_back:
            # LeRobot 0.4-0.6 default recording pipelines retain the input dict.
            # Updating it in place makes the recorder persist the safety-limited
            # command instead of the raw leader request.
            action.clear()
            action.update(sent_action)
        return sent_action

    def disconnect(self) -> None:
        if self._controller is None:
            return

        controller = self._controller
        self._controller = None
        self._last_observation_pose = None
        self._last_observation_time = None
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
                controller.close_connection(
                    joint_release=self.config.joint_release,
                    gripper_release=self.config.gripper_release,
                )
            except Exception:
                logger.exception("Failed to close FAFU controller after connection error")

    def _require_controller(self) -> Any:
        if self._controller is None:
            raise RuntimeError(f"{self} is not connected")
        return self._controller

    def _read_gripper_angle(self) -> float:
        state = self._require_controller().get_gripper_state()
        return self._gripper_angle(state)

    @staticmethod
    def _gripper_angle(state: Any) -> float:
        angle = float(state.position) * math.tau
        if not math.isfinite(angle):
            raise RuntimeError("SDK returned a non-finite gripper position")
        return angle

    def _cartesian_joint_target(
        self,
        action: dict[str, Any],
        *,
        control_mode: str,
        current_joints: np.ndarray,
        current_pose: Pose,
    ) -> np.ndarray:
        if control_mode == "ee_pose":
            desired_pose = pose_from_action(action)
        elif control_mode == "ee_delta":
            translation, rotation = delta_from_action(action)
            desired_pose = apply_pose_delta(current_pose, translation, rotation)
        else:
            raise ValueError(f"Unsupported Cartesian control mode: {control_mode}")

        desired_pose = self._limit_cartesian_target(desired_pose, current_pose)
        solution = self.kinematics.inverse(
            desired_pose.position,
            desired_pose.rotation,
            seed=current_joints,
        )
        if solution is None:
            raise RuntimeError("pytracik could not find an IK solution; no motion command was sent")
        return np.asarray(solution, dtype=np.float64)

    def _limit_cartesian_target(self, target: Pose, current: Pose) -> Pose:
        workspace_min = self.config.ee_workspace_min
        workspace_max = self.config.ee_workspace_max
        if workspace_min is not None and workspace_max is not None:
            minimum = np.asarray(workspace_min, dtype=np.float64)
            maximum = np.asarray(workspace_max, dtype=np.float64)
            if np.any(current.position < minimum) or np.any(current.position > maximum):
                raise RuntimeError(
                    "Current EE position is outside the configured workspace; "
                    "no motion command was sent. Recover manually or widen the validated workspace."
                )
            target = Pose(
                position=np.clip(target.position, minimum, maximum),
                rotation=target.rotation,
            )

        translation, rotation = pose_delta(current, target)
        translation = limit_vector_norm(translation, self.config.max_ee_translation_step_m)
        rotation = limit_vector_norm(rotation, self.config.max_ee_rotation_step_rad)
        limited = apply_pose_delta(current, translation, rotation)

        if workspace_min is not None and workspace_max is not None:
            limited = Pose(
                position=np.clip(limited.position, minimum, maximum),
                rotation=limited.rotation,
            )
        return limited

    def _format_sent_action(
        self,
        joints: np.ndarray,
        *,
        current_pose: Pose,
        actual_pose: Pose,
        gripper: float | None = None,
    ) -> dict[str, float]:
        mode = self.config.action_mode
        gripper = float(self._last_gripper_goal if gripper is None else gripper)
        sent: dict[str, float] = {}
        if mode in {"joint", "all"}:
            sent.update(joint_action(joints, gripper))
        if mode in {"ee_pose", "all"}:
            sent.update(pose_action(actual_pose, gripper))
        if mode in {"ee_delta", "all"}:
            translation, rotation = pose_delta(current_pose, actual_pose)
            sent.update(delta_action(translation, rotation, gripper))
        return sent

    def _validate_action(self, action: dict[str, Any]) -> dict[str, float]:
        if not isinstance(action, dict):
            raise TypeError(f"action must be a dict, got {type(action).__name__}")

        expected = set(self.action_features)
        provided = set(action)
        unexpected = provided - expected
        missing = expected - provided
        if unexpected:
            names = ", ".join(sorted(str(name) for name in unexpected))
            raise ValueError(f"Unexpected action fields for mode {self.config.action_mode}: {names}")
        if self.config.strict_action_features and missing:
            names = ", ".join(sorted(missing))
            raise ValueError(f"Missing action fields for mode {self.config.action_mode}: {names}")

        return {name: self._finite_scalar(value, name) for name, value in action.items()}

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
