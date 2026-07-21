from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np
import pytest

from lerobot_robot_fafu_arm.kinematics import FafuArmKinematics, default_urdf_path


class FakeSolver:
    dof = 6
    joint_limits = (np.full(6, -1.0), np.full(6, 1.0))

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def fk(self, q):
        return np.array([q.sum(), 0.0, 0.175]), np.eye(3)

    def ik(self, position, rotation, seed_jnt_values):
        if position[0] < 0:
            return None
        return seed_jnt_values + 0.1


def test_packaged_urdf_chain_and_updated_tcp():
    root = ET.parse(default_urdf_path()).getroot()
    joints = {joint.attrib["name"]: joint for joint in root.findall("joint")}

    assert list(joints) == ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "tool_joint"]
    assert joints["joint6"].find("axis").attrib["xyz"] == "1 0 0"
    assert joints["tool_joint"].find("origin").attrib["xyz"] == "0.175 0 0"
    assert joints["tool_joint"].find("child").attrib["link"] == "tool_link"


def test_forward_and_inverse_validation():
    model = FafuArmKinematics(solver_class=FakeSolver)
    pose = model.forward(np.arange(6, dtype=float))

    np.testing.assert_allclose(pose.position, [15.0, 0.0, 0.175])
    np.testing.assert_allclose(pose.rotation, np.eye(3))
    np.testing.assert_allclose(model.inverse([0.2, 0.0, 0.1], np.eye(3), seed=np.zeros(6)), 0.1)
    assert model.inverse([-0.2, 0.0, 0.1], np.eye(3)) is None

    with pytest.raises(ValueError, match="joints must have shape"):
        model.forward([0.0, 0.0])
    with pytest.raises(ValueError, match="right-handed"):
        model.inverse([0.2, 0.0, 0.1], np.zeros((3, 3)))
