"""Installation and hardware diagnostic command."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from . import __version__
from .kinematics import FafuArmKinematics
from .sdk import default_sdk_config_path, load_sdk


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check the FAFU LeRobot integration")
    parser.add_argument("--sdk-path", type=Path, help="Path to fafu_arm_sdk or fafu_robot_python")
    parser.add_argument("--config", type=Path, help="SDK robot.cfg path")
    parser.add_argument("--connect", action="store_true", help="Open the hardware in non-actuated mode")
    parser.add_argument("--port", help="Serial port override, for example COM14 or /dev/ttyUSB0")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print(f"fafu_arm_lerobot {__version__}")

    try:
        kinematics = FafuArmKinematics()
        pose = kinematics.forward(np.zeros(6, dtype=np.float64))
        print(f"[OK] URDF / pytracik: {kinematics.urdf_path}")
        print(f"[OK] zero-pose TCP: {np.array2string(pose.position, precision=6)} m")
    except Exception as exc:
        print(f"[FAIL] URDF / pytracik: {exc}", file=sys.stderr)
        return 1

    try:
        bindings = load_sdk(args.sdk_path)
        where = bindings.python_dir or "installed Python package"
        print(f"[OK] FAFU SDK: {where}")
    except Exception as exc:
        print(f"[FAIL] FAFU SDK: {exc}", file=sys.stderr)
        return 1

    if not args.connect:
        print("[OK] software checks passed (hardware was not opened)")
        return 0

    config_path = args.config or default_sdk_config_path()
    controller = None
    try:
        controller = bindings.controller_class(
            cfg_path=str(config_path),
            port=args.port,
            has_gripper=True,
            gripper_motor_id=7,
            auto_enable=False,
            auto_polling=True,
        )
        joints = np.asarray(controller.get_joint_values(), dtype=np.float64)
        print(f"[OK] hardware joint state: {np.array2string(joints, precision=5)} rad")
        print("[OK] hardware check passed; no motion command was sent")
        return 0
    except Exception as exc:
        print(f"[FAIL] hardware: {exc}", file=sys.stderr)
        return 2
    finally:
        if controller is not None:
            controller.close_connection(joint_release="stop", gripper_release="brake")


if __name__ == "__main__":
    raise SystemExit(main())
