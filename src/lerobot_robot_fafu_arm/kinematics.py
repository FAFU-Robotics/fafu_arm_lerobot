"""FK/IK backed by the ROS-free pytracik package."""

from __future__ import annotations

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
