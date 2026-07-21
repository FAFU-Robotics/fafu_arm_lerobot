from __future__ import annotations

import importlib
import math
import sys
import types
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np

from lerobot_robot_fafu_arm.kinematics import Pose


def install_fake_lerobot(monkeypatch):
    lerobot = types.ModuleType("lerobot")
    lerobot.__path__ = []
    cameras = types.ModuleType("lerobot.cameras")
    robots = types.ModuleType("lerobot.robots")
    robots.__path__ = []
    robot_config = types.ModuleType("lerobot.robots.config")
    robot_module = types.ModuleType("lerobot.robots.robot")
    teleoperators = types.ModuleType("lerobot.teleoperators")
    teleoperators.__path__ = []
    teleop_config = types.ModuleType("lerobot.teleoperators.config")
    teleop_module = types.ModuleType("lerobot.teleoperators.teleoperator")

    class Registry:
        @classmethod
        def register_subclass(cls, name):
            return lambda subclass: subclass

    class RobotConfig(Registry):
        def __post_init__(self):
            return None

    class TeleoperatorConfig(Registry):
        pass

    class Robot:
        def __init__(self, config):
            self.id = getattr(config, "id", None)

    class Teleoperator(Robot):
        pass

    cameras.CameraConfig = object
    cameras.make_cameras_from_configs = lambda configs: {}
    robot_config.RobotConfig = RobotConfig
    robot_module.Robot = Robot
    teleop_config.TeleoperatorConfig = TeleoperatorConfig
    teleop_module.Teleoperator = Teleoperator

    modules = {
        "lerobot": lerobot,
        "lerobot.cameras": cameras,
        "lerobot.robots": robots,
        "lerobot.robots.config": robot_config,
        "lerobot.robots.robot": robot_module,
        "lerobot.teleoperators": teleoperators,
        "lerobot.teleoperators.config": teleop_config,
        "lerobot.teleoperators.teleoperator": teleop_module,
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)


class FakeKinematics:
    joint_limits = (np.full(6, -0.5), np.full(6, 0.5))

    def __init__(self, urdf_path=None):
        self.urdf_path = urdf_path

    def forward(self, joints):
        joints = np.asarray(joints, dtype=float)
        return Pose(position=np.array([0.2 + joints[0], 0.0, 0.2]), rotation=np.eye(3))

    def inverse(self, position, rotation, seed=None):
        result = np.asarray(seed, dtype=float).copy()
        result[0] = float(position[0]) - 0.2
        return result


@dataclass
class FakeServoOptions:
    watchdog_ms: int
    max_vel: float
    max_step_rad: float
    max_lag_rad: float
    rate_hz: float
    use_mit: bool


class FakeController:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.joints = np.zeros(6)
        self.gripper_turns = 0.0
        self.joint_velocities = np.full(6, 0.1)
        self.joint_motor_ids = list(range(1, 7))
        self.is_servoing = False
        self.last_servo = None
        self.last_gripper = None
        self.closed = False

    def get_joint_values(self):
        return self.joints.copy()

    def get_gripper_state(self):
        return SimpleNamespace(position=self.gripper_turns, velocity=0.02, torque=17.0)

    def get_joint_velocities(self):
        return self.joint_velocities.copy()

    def get_motor_states(self):
        return {motor_id: SimpleNamespace(torque=float(motor_id * 10)) for motor_id in self.joint_motor_ids}

    def servo_start(self, options):
        self.options = options
        self.is_servoing = True

    def servo_j(self, target):
        self.last_servo = np.asarray(target)
        self.joints = self.last_servo.copy()
        return True

    def servo_end(self, finish_mode):
        self.is_servoing = False
        self.finish_mode = finish_mode

    def gripper_control(self, **kwargs):
        self.last_gripper = kwargs

    def disable(self):
        self.disabled = True

    def close_connection(self, **kwargs):
        self.closed = True
        self.close_options = kwargs


def test_follower_clips_and_streams_actions(monkeypatch, tmp_path):
    install_fake_lerobot(monkeypatch)
    config_module = importlib.import_module("lerobot_robot_fafu_arm.config")
    follower_module = importlib.import_module("lerobot_robot_fafu_arm.follower")
    monkeypatch.setattr(follower_module, "FafuArmKinematics", FakeKinematics)

    controller = FakeController()
    bindings = SimpleNamespace(
        controller_class=lambda **kwargs: controller,
        servo_options_class=FakeServoOptions,
    )
    monkeypatch.setattr(follower_module, "load_sdk", lambda path: bindings)

    config = config_module.FafuFollowerConfig(
        calibration_dir=tmp_path / "calibration",
        max_relative_target=0.15,
    )
    robot = follower_module.FafuFollower(config)
    robot.connect()
    sent = robot.send_action({"joint1.pos": 1.0, "gripper.pos": 1.0})

    assert math.isclose(controller.last_servo[0], 0.15)
    assert np.all(controller.last_servo <= 0.5)
    assert math.isclose(sent["joint1.pos"], 0.15)
    assert controller.last_gripper["block"] is False

    robot.disconnect()
    assert controller.closed
    assert controller.close_options == {"joint_release": "stop", "gripper_release": "brake"}


def test_follower_converts_ee_delta_to_safe_joint_target(monkeypatch, tmp_path):
    install_fake_lerobot(monkeypatch)
    config_module = importlib.import_module("lerobot_robot_fafu_arm.config")
    follower_module = importlib.import_module("lerobot_robot_fafu_arm.follower")
    monkeypatch.setattr(follower_module, "FafuArmKinematics", FakeKinematics)

    controller = FakeController()
    bindings = SimpleNamespace(
        controller_class=lambda **kwargs: controller,
        servo_options_class=FakeServoOptions,
    )
    monkeypatch.setattr(follower_module, "load_sdk", lambda path: bindings)

    config = config_module.FafuFollowerConfig(
        calibration_dir=tmp_path / "cartesian_calibration",
        action_mode="ee_delta",
        observation_mode="all",
        max_relative_target=None,
        max_ee_translation_step_m=0.03,
    )
    robot = follower_module.FafuFollower(config)
    robot.connect()
    sent = robot.send_action(
        {
            "ee_delta.x": 0.20,
            "ee_delta.y": 0.0,
            "ee_delta.z": 0.0,
            "ee_delta.wx": 0.0,
            "ee_delta.wy": 0.0,
            "ee_delta.wz": 0.0,
            "gripper.pos": 0.2,
        }
    )

    assert math.isclose(controller.last_servo[0], 0.03)
    assert math.isclose(sent["ee_delta.x"], 0.03)
    assert math.isclose(sent["gripper.pos"], 0.2)
    robot.disconnect()


def test_follower_records_joint_and_cartesian_state_together(monkeypatch, tmp_path):
    install_fake_lerobot(monkeypatch)
    config_module = importlib.import_module("lerobot_robot_fafu_arm.config")
    follower_module = importlib.import_module("lerobot_robot_fafu_arm.follower")
    monkeypatch.setattr(follower_module, "FafuArmKinematics", FakeKinematics)

    controller = FakeController()
    bindings = SimpleNamespace(
        controller_class=lambda **kwargs: controller,
        servo_options_class=FakeServoOptions,
    )
    monkeypatch.setattr(follower_module, "load_sdk", lambda path: bindings)

    config = config_module.FafuFollowerConfig(
        calibration_dir=tmp_path / "observation_calibration",
        observation_mode="all",
        record_joint_velocity=True,
        record_motor_effort=True,
    )
    robot = follower_module.FafuFollower(config)
    robot.connect()

    first = robot.get_observation()
    assert first["joint1.pos"] == 0.0
    assert first["joint1.vel"] == 0.1
    assert first["joint1.effort"] == 10.0
    assert first["ee.x"] == 0.2
    assert first["ee_delta.x"] == 0.0

    controller.joints[0] = 0.02
    second = robot.get_observation()
    assert math.isclose(second["ee.x"], 0.22)
    assert math.isclose(second["ee_delta.x"], 0.02)
    assert set(second) == set(robot.observation_features)
    robot.disconnect()


def test_leader_all_mode_emits_all_action_representations(monkeypatch, tmp_path):
    install_fake_lerobot(monkeypatch)
    config_module = importlib.import_module("lerobot_robot_fafu_arm.config")
    leader_module = importlib.import_module("lerobot_robot_fafu_arm.leader")
    monkeypatch.setattr(leader_module, "FafuArmKinematics", FakeKinematics)

    controller = FakeController()
    bindings = SimpleNamespace(controller_class=lambda **kwargs: controller)
    monkeypatch.setattr(leader_module, "load_sdk", lambda path: bindings)

    config = config_module.FafuLeaderConfig(
        calibration_dir=tmp_path / "leader_calibration",
        action_mode="all",
    )
    leader = leader_module.FafuLeader(config)
    leader.connect()
    first = leader.get_action()
    assert first["joint1.pos"] == 0.0
    assert first["ee.x"] == 0.2
    assert first["ee_delta.x"] == 0.0

    controller.joints[0] = 0.01
    second = leader.get_action()
    assert math.isclose(second["ee_delta.x"], 0.01)
    assert set(second) == set(leader.action_features)
    assert controller.disabled
    leader.disconnect()
