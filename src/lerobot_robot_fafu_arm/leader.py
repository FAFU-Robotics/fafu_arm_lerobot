"""LeRobot teleoperator implementation for a manually moved FAFU leader arm."""

from __future__ import annotations

import logging
import math
from functools import cached_property
from typing import Any

import numpy as np
from lerobot.teleoperators.teleoperator import Teleoperator

from .config import JOINT_NAMES, FafuLeaderConfig
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

    @cached_property
    def action_features(self) -> dict[str, type]:
        return {f"{name}.pos": float for name in (*JOINT_NAMES, "gripper")}

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
        if calibrate:
            self.calibrate()
        logger.info("%s connected in manual leader mode", self)

    def get_action(self) -> dict[str, float]:
        controller = self._require_controller()
        joints = np.asarray(controller.get_joint_values(), dtype=np.float64)
        if joints.shape != (len(JOINT_NAMES),):
            raise RuntimeError(f"SDK returned {joints.size} joints; expected {len(JOINT_NAMES)}")

        action = {f"{name}.pos": float(joints[index]) for index, name in enumerate(JOINT_NAMES)}
        action["gripper.pos"] = float(controller.get_gripper_state().position) * math.tau
        return action

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        # Force feedback is not implemented by the current SDK.
        return None

    def disconnect(self) -> None:
        if self._controller is None:
            return
        controller, self._controller = self._controller, None
        controller.close_connection(
            joint_release=self.config.joint_release,
            gripper_release=self.config.gripper_release,
        )
        logger.info("%s disconnected", self)

    def _require_controller(self) -> Any:
        if self._controller is None:
            raise RuntimeError(f"{self} is not connected")
        return self._controller
