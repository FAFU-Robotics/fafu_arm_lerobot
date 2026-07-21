from __future__ import annotations

import numpy as np

from lerobot_robot_fafu_arm.kinematics import (
    Pose,
    rotation_matrix_to_rotvec,
    rotation_vector_to_matrix,
)
from lerobot_robot_fafu_arm.representation import (
    action_features,
    apply_pose_delta,
    delta_action,
    pose_action,
    pose_delta,
)


def test_action_modes_have_stable_feature_contracts():
    assert set(action_features("joint")) == {
        "joint1.pos",
        "joint2.pos",
        "joint3.pos",
        "joint4.pos",
        "joint5.pos",
        "joint6.pos",
        "gripper.pos",
    }
    assert set(action_features("ee_pose")) == {
        "ee.x",
        "ee.y",
        "ee.z",
        "ee.wx",
        "ee.wy",
        "ee.wz",
        "gripper.pos",
    }
    assert set(action_features("ee_delta")) == {
        "ee_delta.x",
        "ee_delta.y",
        "ee_delta.z",
        "ee_delta.wx",
        "ee_delta.wy",
        "ee_delta.wz",
        "gripper.pos",
    }
    assert len(action_features("all")) == 19


def test_pose_delta_round_trip_and_serialization():
    reference = Pose(
        position=np.array([0.2, -0.1, 0.3]),
        rotation=rotation_vector_to_matrix([0.1, -0.2, 0.3]),
    )
    expected_translation = np.array([0.01, -0.02, 0.03])
    expected_rotation = np.array([-0.04, 0.02, 0.01])
    target = apply_pose_delta(reference, expected_translation, expected_rotation)

    translation, rotation = pose_delta(reference, target)
    np.testing.assert_allclose(translation, expected_translation, atol=1e-10)
    np.testing.assert_allclose(rotation, expected_rotation, atol=1e-10)

    absolute = pose_action(target, 0.4)
    relative = delta_action(translation, rotation, 0.4)
    assert absolute["gripper.pos"] == 0.4
    assert np.isclose(relative["ee_delta.z"], expected_translation[2])


def test_rotation_vector_round_trip_near_pi():
    rotation_vector = np.array([0.0, np.pi - 1e-6, 0.0])
    matrix = rotation_vector_to_matrix(rotation_vector)
    recovered = rotation_matrix_to_rotvec(matrix)
    np.testing.assert_allclose(
        rotation_vector_to_matrix(recovered),
        matrix,
        atol=1e-8,
    )
