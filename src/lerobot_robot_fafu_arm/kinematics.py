"""FK/IK backed by the ROS-free pytracik package."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class Pose:
    position: NDArray[np.float64]
    rotation: NDArray[np.float64]


def default_urdf_path() -> Path:
    return Path(__file__).resolve().parent / "resources" / "fafu_arm.urdf"


def _vector(values: ArrayLike, size: int, name: str) -> NDArray[np.float64]:
    result = np.asarray(values, dtype=np.float64)
    if result.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},), got {result.shape}")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} contains NaN or infinity")
    return result


def _rotation(values: ArrayLike) -> NDArray[np.float64]:
    result = np.asarray(values, dtype=np.float64)
    if result.shape != (3, 3):
        raise ValueError(f"rotation must have shape (3, 3), got {result.shape}")
    if not np.all(np.isfinite(result)):
        raise ValueError("rotation contains NaN or infinity")
    if not np.allclose(result.T @ result, np.eye(3), atol=1e-5) or not np.isclose(
        np.linalg.det(result), 1.0, atol=1e-5
    ):
        raise ValueError("rotation must be an orthonormal right-handed matrix")
    return result


def rotation_vector_to_matrix(values: ArrayLike) -> NDArray[np.float64]:
    """Convert an axis-angle rotation vector to a 3x3 rotation matrix."""

    vector = _vector(values, 3, "rotation vector")
    theta = float(np.linalg.norm(vector))
    skew = np.array(
        [
            [0.0, -vector[2], vector[1]],
            [vector[2], 0.0, -vector[0]],
            [-vector[1], vector[0], 0.0],
        ],
        dtype=np.float64,
    )
    if theta < 1e-8:
        sine_scale = 1.0 - theta**2 / 6.0
        cosine_scale = 0.5 - theta**2 / 24.0
    else:
        sine_scale = math.sin(theta) / theta
        cosine_scale = (1.0 - math.cos(theta)) / theta**2
    return np.eye(3, dtype=np.float64) + sine_scale * skew + cosine_scale * (skew @ skew)


def rotation_matrix_to_rotvec(values: ArrayLike) -> NDArray[np.float64]:
    """Convert a 3x3 rotation matrix to the shortest axis-angle vector."""

    rotation = _rotation(values)
    trace = float(np.trace(rotation))

    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quaternion = np.array(
            [
                0.25 * scale,
                (rotation[2, 1] - rotation[1, 2]) / scale,
                (rotation[0, 2] - rotation[2, 0]) / scale,
                (rotation[1, 0] - rotation[0, 1]) / scale,
            ],
            dtype=np.float64,
        )
    else:
        index = int(np.argmax(np.diag(rotation)))
        if index == 0:
            scale = math.sqrt(max(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2], 0.0)) * 2.0
            quaternion = np.array(
                [
                    (rotation[2, 1] - rotation[1, 2]) / scale,
                    0.25 * scale,
                    (rotation[0, 1] + rotation[1, 0]) / scale,
                    (rotation[0, 2] + rotation[2, 0]) / scale,
                ],
                dtype=np.float64,
            )
        elif index == 1:
            scale = math.sqrt(max(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2], 0.0)) * 2.0
            quaternion = np.array(
                [
                    (rotation[0, 2] - rotation[2, 0]) / scale,
                    (rotation[0, 1] + rotation[1, 0]) / scale,
                    0.25 * scale,
                    (rotation[1, 2] + rotation[2, 1]) / scale,
                ],
                dtype=np.float64,
            )
        else:
            scale = math.sqrt(max(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1], 0.0)) * 2.0
            quaternion = np.array(
                [
                    (rotation[1, 0] - rotation[0, 1]) / scale,
                    (rotation[0, 2] + rotation[2, 0]) / scale,
                    (rotation[1, 2] + rotation[2, 1]) / scale,
                    0.25 * scale,
                ],
                dtype=np.float64,
            )

    norm = float(np.linalg.norm(quaternion))
    if norm < 1e-12:
        raise ValueError("rotation could not be converted to a quaternion")
    quaternion /= norm
    if quaternion[0] < 0.0:
        quaternion = -quaternion

    axis_part = quaternion[1:]
    sine_half = float(np.linalg.norm(axis_part))
    if sine_half < 1e-10:
        return 2.0 * axis_part
    angle = 2.0 * math.atan2(sine_half, float(quaternion[0]))
    return np.asarray(axis_part * (angle / sine_half), dtype=np.float64)


class FafuArmKinematics:
    """Six-axis kinematics using ``base_link -> tool_link`` by default."""

    def __init__(
        self,
        urdf_path: str | Path | None = None,
        *,
        base_link: str = "base_link",
        tip_link: str = "tool_link",
        timeout: float = 0.005,
        epsilon: float = 1e-5,
        solver_type: str = "Distance",
        solver_class: type[Any] | None = None,
    ) -> None:
        self.urdf_path = Path(urdf_path) if urdf_path else default_urdf_path()
        if not self.urdf_path.is_file():
            raise FileNotFoundError(f"URDF not found: {self.urdf_path}")
        if solver_class is None:
            try:
                from trac_ik import TracIK
            except (ImportError, OSError) as exc:
                raise ImportError(
                    "pytracik could not be loaded. Install pytracik>=0.0.3; on Windows, "
                    "make sure its runtime DLLs match the active Python architecture."
                ) from exc
            solver_class = TracIK

        self._solver = solver_class(
            base_link_name=base_link,
            tip_link_name=tip_link,
            urdf_path=str(self.urdf_path),
            timeout=timeout,
            epsilon=epsilon,
            solver_type=solver_type,
        )
        if self.dof != 6:
            raise RuntimeError(f"Expected a 6-DoF FAFU chain, got {self.dof}")

    @property
    def dof(self) -> int:
        return int(self._solver.dof)

    @property
    def joint_limits(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        lower, upper = self._solver.joint_limits
        return np.asarray(lower, dtype=np.float64), np.asarray(upper, dtype=np.float64)

    def forward(self, joints: ArrayLike) -> Pose:
        q = _vector(joints, self.dof, "joints")
        position, rotation = self._solver.fk(q)
        return Pose(
            position=np.asarray(position, dtype=np.float64),
            rotation=np.asarray(rotation, dtype=np.float64),
        )

    def inverse(
        self,
        position: ArrayLike,
        rotation: ArrayLike,
        *,
        seed: ArrayLike | None = None,
    ) -> NDArray[np.float64] | None:
        target_position = _vector(position, 3, "position")
        target_rotation = _rotation(rotation)
        seed_values = (
            np.zeros(self.dof, dtype=np.float64) if seed is None else _vector(seed, self.dof, "seed")
        )
        result = self._solver.ik(target_position, target_rotation, seed_jnt_values=seed_values)
        if result is None:
            return None
        return _vector(result, self.dof, "IK result")
