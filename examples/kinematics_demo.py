"""Check an FK -> IK -> FK round trip without connecting to hardware."""

import argparse
from pathlib import Path

import numpy as np

from lerobot_robot_fafu_arm import FafuArmKinematics, rotation_matrix_to_rotvec

POSITION_TOLERANCE_M = 1e-5
ROTATION_TOLERANCE_RAD = 1e-4
JOINT_LIMIT_TOLERANCE_RAD = 1e-6


def _joint_limit_violations(
    joints: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> np.ndarray:
    """Return joints outside finite URDF bounds; infinite bounds are unbounded."""

    finite_lower = np.isfinite(lower)
    finite_upper = np.isfinite(upper)
    below = finite_lower & (joints < lower - JOINT_LIMIT_TOLERANCE_RAD)
    above = finite_upper & (joints > upper + JOINT_LIMIT_TOLERANCE_RAD)
    return np.flatnonzero(below | above)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", type=Path, help="custom FAFU Arm URDF")
    args = parser.parse_args(argv)

    kinematics = FafuArmKinematics(args.urdf)
    target_joints = np.array([0.0, 0.5, 1.0, 0.0, 0.0, 0.0])
    seed = np.zeros(6, dtype=np.float64)
    target = kinematics.forward(target_joints)
    solved = kinematics.inverse(target.position, target.rotation, seed=seed)
    if solved is None:
        print("[FAIL] IK did not find a solution")
        return 1

    solved = np.asarray(solved, dtype=np.float64)
    if solved.shape != seed.shape:
        print(f"[FAIL] IK returned shape {solved.shape}; expected {seed.shape}")
        return 1
    if not np.all(np.isfinite(solved)):
        print("[FAIL] IK returned NaN or infinity")
        return 1

    lower, upper = (np.asarray(bound, dtype=np.float64) for bound in kinematics.joint_limits)
    invalid_limits = (
        lower.shape != seed.shape
        or upper.shape != seed.shape
        or np.isnan(lower).any()
        or np.isnan(upper).any()
        or np.isposinf(lower).any()
        or np.isneginf(upper).any()
        or np.any(lower > upper)
    )
    if invalid_limits:
        print("[FAIL] URDF returned invalid joint limits")
        return 1
    violations = _joint_limit_violations(solved, lower, upper)
    if violations.size:
        joint_numbers = ", ".join(str(index + 1) for index in violations)
        print(f"[FAIL] IK solution violates finite URDF limits for joint(s): {joint_numbers}")
        return 1

    recovered = kinematics.forward(solved)
    position_error = float(np.linalg.norm(recovered.position - target.position))
    rotation_error = float(np.linalg.norm(rotation_matrix_to_rotvec(target.rotation.T @ recovered.rotation)))
    seed_distance = float(np.linalg.norm(solved - seed))
    print(f"position error: {position_error:.3e} m")
    print(f"rotation error: {rotation_error:.3e} rad")
    print(f"distance from independent IK seed: {seed_distance:.3e} rad")
    if position_error > POSITION_TOLERANCE_M or rotation_error > ROTATION_TOLERANCE_RAD:
        print("[FAIL] FK/IK round-trip error exceeds tolerance")
        return 1
    print("[OK] FK/IK round trip")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
