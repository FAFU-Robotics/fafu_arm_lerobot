"""Action and observation representations shared by the FAFU devices."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .kinematics import Pose, rotation_matrix_to_rotvec, rotation_vector_to_matrix

JOINT_NAMES = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6")
EE_COMPONENTS = ("x", "y", "z", "wx", "wy", "wz")

ActionMode = Literal["joint", "ee_pose", "ee_delta", "all"]
ObservationMode = Literal["joint", "ee_pose", "all"]

ACTION_MODES = frozenset({"joint", "ee_pose", "ee_delta", "all"})
OBSERVATION_MODES = frozenset({"joint", "ee_pose", "all"})
CARTESIAN_CONTROL_MODES = frozenset({"ee_pose", "ee_delta"})


def action_features(mode: ActionMode) -> dict[str, type]:
    """Return the scalar feature contract for an action representation."""

    features: dict[str, type] = {}
    if mode in {"joint", "all"}:
        features.update({f"{name}.pos": float for name in JOINT_NAMES})
    if mode in {"ee_pose", "all"}:
        features.update({f"ee.{name}": float for name in EE_COMPONENTS})
    if mode in {"ee_delta", "all"}:
        features.update({f"ee_delta.{name}": float for name in EE_COMPONENTS})
    features["gripper.pos"] = float
    return features


def joint_action(joints: ArrayLike, gripper: float) -> dict[str, float]:
    values = _finite_vector(joints, 6, "joints")
    action = {f"{name}.pos": float(values[index]) for index, name in enumerate(JOINT_NAMES)}
    action["gripper.pos"] = _finite_scalar(gripper, "gripper.pos")
    return action


def pose_action(pose: Pose, gripper: float, *, prefix: str = "ee") -> dict[str, float]:
    rotation_vector = rotation_matrix_to_rotvec(pose.rotation)
    values = np.concatenate((pose.position, rotation_vector))
    action = {f"{prefix}.{name}": float(values[index]) for index, name in enumerate(EE_COMPONENTS)}
    action["gripper.pos"] = _finite_scalar(gripper, "gripper.pos")
    return action


def delta_action(
    translation: ArrayLike,
    rotation_vector: ArrayLike,
    gripper: float,
) -> dict[str, float]:
    values = np.concatenate(
        (
            _finite_vector(translation, 3, "translation delta"),
            _finite_vector(rotation_vector, 3, "rotation delta"),
        )
    )
    action = {f"ee_delta.{name}": float(values[index]) for index, name in enumerate(EE_COMPONENTS)}
    action["gripper.pos"] = _finite_scalar(gripper, "gripper.pos")
    return action


def pose_from_action(action: Mapping[str, Any], *, prefix: str = "ee") -> Pose:
    values = _required_components(action, prefix)
    return Pose(position=values[:3], rotation=rotation_vector_to_matrix(values[3:]))


def delta_from_action(action: Mapping[str, Any]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    values = _required_components(action, "ee_delta")
    return values[:3], values[3:]


def pose_delta(
    previous: Pose,
    current: Pose,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return base-frame translation and previous-tool-frame rotation deltas."""

    translation = np.asarray(current.position - previous.position, dtype=np.float64)
    rotation = previous.rotation.T @ current.rotation
    return translation, rotation_matrix_to_rotvec(rotation)


def apply_pose_delta(
    reference: Pose,
    translation: ArrayLike,
    rotation_vector: ArrayLike,
) -> Pose:
    """Apply a base-frame translation and local rotation delta to a pose."""

    delta_position = _finite_vector(translation, 3, "translation delta")
    delta_rotation = _finite_vector(rotation_vector, 3, "rotation delta")
    return Pose(
        position=np.asarray(reference.position + delta_position, dtype=np.float64),
        rotation=np.asarray(
            reference.rotation @ rotation_vector_to_matrix(delta_rotation),
            dtype=np.float64,
        ),
    )


def limit_vector_norm(values: ArrayLike, maximum: float | None) -> NDArray[np.float64]:
    vector = np.asarray(values, dtype=np.float64)
    if maximum is None:
        return vector.copy()
    norm = float(np.linalg.norm(vector))
    if norm > maximum and norm > 0.0:
        return vector * (maximum / norm)
    return vector.copy()


def _required_components(action: Mapping[str, Any], prefix: str) -> NDArray[np.float64]:
    keys = [f"{prefix}.{name}" for name in EE_COMPONENTS]
    missing = [key for key in keys if key not in action]
    if missing:
        raise ValueError(f"Missing required {prefix} action fields: {missing}")
    return _finite_vector([action[key] for key in keys], 6, prefix)


def _finite_vector(values: ArrayLike, size: int, name: str) -> NDArray[np.float64]:
    result = np.asarray(values, dtype=np.float64)
    if result.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},), got {result.shape}")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} contains NaN or infinity")
    return result


def _finite_scalar(value: Any, name: str) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result
