"""LeRobot teleoperator implementation for a manually moved FAFU leader arm."""

from __future__ import annotations

import logging
import math
import time
from functools import cached_property
from typing import Any

import numpy as np
from lerobot.teleoperators.teleoperator import Teleoperator

from .config import FafuLeaderConfig
from .kinematics import FafuArmKinematics, Pose
from .representation import (
    JOINT_NAMES,
    action_features,
    delta_action,
    joint_action,
    pose_action,
    pose_delta,
)
from .sdk import default_sdk_config_path, load_sdk

logger = logging.getLogger(__name__)


class FafuLeader(Teleoperator):
    """Read joint positions from a released FAFU arm for demonstrations."""

    config_class = FafuLeaderConfig
    name = "fafu_leader"

    def __init__(self, config: FafuLeaderConfig):
        super().__init__(config)
        self.config = config
        self._controller: Any | None = None
        self.kinematics: FafuArmKinematics | None = (
            None if config.action_mode == "joint" else FafuArmKinematics(config.urdf_path)
        )
        self._previous_pose: Pose | None = None
        self._previous_action_time: float | None = None

    @cached_property
    def action_features(self) -> dict[str, type]:
        return action_features(self.config.action_mode)

    @cached_property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return self._controller is not None

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        logger.info("%s uses SDK-side absolute joint calibration", self)

    def configure(self) -> None:
        # The controller is intentionally disabled so a person can move it.
        return None

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
            auto_enable=False,
            auto_polling=True,
        )
        try:
            controller.disable()
        except Exception:
            controller.close_connection(joint_release="stop", gripper_release="stop")
            raise
        self._controller = controller
        self._previous_pose = None
        self._previous_action_time = None
        if calibrate:
            self.calibrate()
        logger.info("%s connected in manual leader mode", self)

    def get_action(self) -> dict[str, float]:
        controller = self._require_controller()
        joints = np.asarray(controller.get_joint_values(), dtype=np.float64)
        if joints.shape != (len(JOINT_NAMES),):
            raise RuntimeError(f"SDK returned {joints.size} joints; expected {len(JOINT_NAMES)}")

        gripper = float(controller.get_gripper_state().position) * math.tau
        if not math.isfinite(gripper):
            raise RuntimeError("SDK returned a non-finite gripper position")

        mode = self.config.action_mode
        action: dict[str, float] = {}
        if mode in {"joint", "all"}:
            action.update(joint_action(joints, gripper))

        if mode in {"ee_pose", "ee_delta", "all"}:
            pose = self._require_kinematics().forward(joints)
            if mode in {"ee_pose", "all"}:
                action.update(pose_action(pose, gripper))
            if mode in {"ee_delta", "all"}:
                now = time.perf_counter()
                if (
                    self._previous_pose is None
                    or self._previous_action_time is None
                    or now - self._previous_action_time > self.config.delta_reset_timeout_s
                ):
                    translation_delta = np.zeros(3, dtype=np.float64)
                    rotation_delta = np.zeros(3, dtype=np.float64)
                else:
                    translation_delta, rotation_delta = pose_delta(self._previous_pose, pose)
                action.update(delta_action(translation_delta, rotation_delta, gripper))
                self._previous_pose = pose
                self._previous_action_time = now
        return action

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        # Force feedback is not implemented by the current SDK.
        return None

    def disconnect(self) -> None:
        if self._controller is None:
            return
        controller, self._controller = self._controller, None
        self._previous_pose = None
        self._previous_action_time = None
        controller.close_connection(
            joint_release=self.config.joint_release,
            gripper_release=self.config.gripper_release,
        )
        logger.info("%s disconnected", self)

    def _require_controller(self) -> Any:
        if self._controller is None:
            raise RuntimeError(f"{self} is not connected")
        return self._controller

    def _require_kinematics(self) -> FafuArmKinematics:
        if self.kinematics is None:
            raise RuntimeError("Cartesian action mode requires a kinematics model")
        return self.kinematics
