"""Command-line entry point for strict FAFU policy inference."""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import math
import sys
from collections.abc import Callable
from contextlib import nullcontext, redirect_stdout
from pathlib import Path
from typing import Any

import yaml

from ..config import FafuFollowerConfig
from ..follower import FafuFollower
from ..kinematics import rotation_matrix_to_rotvec, rotation_vector_to_matrix
from ..representation import EE_COMPONENTS, JOINT_NAMES
from .act import (
    ActPolicyRuntime,
    derive_observation_settings,
    run_control_loop,
    validate_robot_schema,
)
from .manifest import InferenceManifest, load_inference_manifest

_CAMERA_CONFIG_MODULES = (
    "lerobot.cameras.opencv.configuration_opencv",
    "lerobot.cameras.realsense.configuration_realsense",
    "lerobot.cameras.zmq.configuration_zmq",
    "lerobot.cameras.reachy2_camera.configuration_reachy2_camera",
)

_MIN_FPS = 10.0
_MAX_FPS = 200.0
_MAX_DURATION_S = 300.0
_MAX_RELATIVE_TARGET_RAD = 0.5
_MAX_EE_TRANSLATION_STEP_M = 0.05
_MAX_EE_ROTATION_STEP_RAD = 0.5
_MAX_SERVO_VELOCITY_RAD_S = 3.0
_MAX_SERVO_STEP_RAD = 0.2
_MAX_SERVO_LAG_RAD = 0.7
_MAX_WORKSPACE_COORDINATE_M = 2.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Preflight and run FAFU ACT inference")
    subparsers = parser.add_subparsers(dest="algorithm", required=True)
    act = subparsers.add_parser("act", help="Run an official ACT or FAFU ACT-demo checkpoint")
    _add_act_arguments(act)
    return parser


def build_act_parser() -> argparse.ArgumentParser:
    """Build the standalone parser used by ``examples/act_inference.py``."""

    parser = argparse.ArgumentParser(description="Preflight and run FAFU ACT inference")
    _add_act_arguments(parser)
    return parser


def _add_act_arguments(act: argparse.ArgumentParser) -> None:
    act.add_argument("--checkpoint", type=Path, required=True, help="Local pretrained_model directory")
    act.add_argument(
        "--dataset-root",
        type=Path,
        help="Original local dataset; required only when an old checkpoint has no manifest",
    )
    act.add_argument(
        "--action-mode",
        choices=("joint", "ee_delta", "ee_pose"),
        help="Required with --dataset-root for an old checkpoint; otherwise cross-checks the manifest",
    )
    act.add_argument(
        "--cameras",
        type=Path,
        required=True,
        help="YAML/JSON mapping of LeRobot camera configs",
    )
    act.add_argument("--port", help="Follower serial port; required with --run")
    act.add_argument("--robot-id", default="fafu_inference")
    act.add_argument("--calibration-dir", type=Path)
    act.add_argument("--sdk-path", type=Path)
    act.add_argument("--sdk-config-path", type=Path)
    act.add_argument("--baudrate", type=int)
    act.add_argument("--urdf-path", type=Path)
    act.add_argument("--device", help="cpu, cuda, cuda:N, or mps; defaults to the best available")
    act.add_argument("--task", default="", help="Task label retained in the inference frame")
    act.add_argument("--fps", type=float, help="Must equal the training dataset FPS")
    act.add_argument("--duration", type=float, default=5.0, help="Finite rollout duration in seconds")
    act.add_argument(
        "--start-joints",
        type=float,
        nargs=7,
        metavar=("J1", "J2", "J3", "J4", "J5", "J6", "GRIPPER"),
        help="Required joint/gripper start state for --run when joint positions are observed",
    )
    act.add_argument(
        "--start-ee",
        type=float,
        nargs=7,
        metavar=("X", "Y", "Z", "WX", "WY", "WZ", "GRIPPER"),
        help="Required TCP rotvec/gripper start state when the policy observes EE pose only",
    )
    act.add_argument("--start-joint-tolerance-rad", type=float, default=0.15)
    act.add_argument("--start-ee-translation-tolerance-m", type=float, default=0.03)
    act.add_argument("--start-ee-rotation-tolerance-rad", type=float, default=0.15)
    act.add_argument("--start-gripper-tolerance-rad", type=float, default=0.25)
    act.add_argument("--max-relative-target", type=float, default=0.03)
    act.add_argument("--max-ee-translation-step-m", type=float, default=0.01)
    act.add_argument("--max-ee-rotation-step-rad", type=float, default=0.10)
    act.add_argument("--ee-workspace-min", type=float, nargs=3, metavar=("X", "Y", "Z"))
    act.add_argument("--ee-workspace-max", type=float, nargs=3, metavar=("X", "Y", "Z"))
    act.add_argument("--servo-watchdog-ms", type=int, default=250)
    act.add_argument("--servo-max-velocity", type=float, default=0.3)
    act.add_argument("--servo-max-step-rad", type=float, default=0.10)
    act.add_argument("--servo-max-lag-rad", type=float, default=0.35)
    act.add_argument("--gripper-effort", type=int, default=300)
    act.add_argument("--max-consecutive-overruns", type=int, default=3)
    act.add_argument("--max-overruns-per-second", type=int, default=3)
    act.add_argument("--run", action="store_true", help="Connect hardware and execute actions")
    act.add_argument("--json", action="store_true", dest="as_json")


def load_camera_configs(path: str | Path) -> dict[str, Any]:
    """Decode a camera mapping with LeRobot's registered config classes."""

    import draccus
    from lerobot.cameras import CameraConfig
    from lerobot.utils.import_utils import register_third_party_plugins

    for module in _CAMERA_CONFIG_MODULES:
        importlib.import_module(module)
    register_third_party_plugins()

    source = Path(path).expanduser().resolve()
    try:
        raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"camera config not found: {source}") from exc
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"could not read camera config {source}: {exc}") from exc
    if not isinstance(raw, dict) or not raw:
        raise ValueError("camera config must be a non-empty mapping")
    if not all(isinstance(name, str) and name for name in raw):
        raise ValueError("camera names must be non-empty strings")
    try:
        return draccus.decode(dict[str, CameraConfig], raw)
    except Exception as exc:
        raise ValueError(f"invalid camera config {source}: {exc}") from exc


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return _run_act(args)


def act_main(argv: list[str] | None = None) -> int:
    """Run the ACT command without requiring the console subcommand token."""

    args = build_act_parser().parse_args(argv)
    return _run_act(args)


def _run_act(args: argparse.Namespace) -> int:
    if args.as_json and args.run:
        print("[FAIL] --json cannot be combined with --run", file=sys.stderr)
        return 2

    try:
        output_context = redirect_stdout(sys.stderr) if args.as_json else nullcontext()
        with output_context:
            manifest = load_inference_manifest(
                args.checkpoint,
                dataset_root=args.dataset_root,
                action_mode=args.action_mode,
                urdf_path=args.urdf_path,
            )
            fps = manifest.fps if args.fps is None else args.fps
            _validate_cli_safety(args, manifest, fps)
            cameras = load_camera_configs(args.cameras)

            runtime = ActPolicyRuntime.load(
                args.checkpoint,
                manifest,
                allow_legacy_checkpoint=manifest.checkpoint_files is None,
                device=args.device,
                task=args.task,
            )
            warmup_latency = runtime.synthetic_warmup()
            realtime_budget = _validate_warmup_latency(warmup_latency, fps, args.servo_watchdog_ms)
            robot = _build_robot(args, manifest, cameras, fps)
            validate_robot_schema(robot, manifest, fps=fps)
            initial_observation_validator = _build_initial_observation_validator(args, manifest)
            report = _preflight_report(args, runtime, manifest, fps, warmup_latency, realtime_budget)

        if args.as_json:
            print(json.dumps(report, ensure_ascii=False, indent=2))
        else:
            _print_preflight(report)
        if not args.run:
            if not args.as_json:
                print("[DRY-RUN] no hardware was connected; add --run after reviewing the checks")
            return 0

        result = run_control_loop(
            robot,
            runtime,
            fps=fps,
            duration_s=args.duration,
            max_consecutive_overruns=args.max_consecutive_overruns,
            max_overruns_per_second=args.max_overruns_per_second,
            initial_observation_validator=initial_observation_validator,
        )
        print(
            f"[OK] ACT rollout: {result.steps} step(s), {result.elapsed_s:.2f} s, "
            f"{result.overruns} overrun(s), max latency {result.max_control_latency_s * 1e3:.1f} ms"
        )
        return 0
    except KeyboardInterrupt:
        print("[STOP] interrupted; the robot was disconnected", file=sys.stderr)
        return 130
    except Exception as exc:
        logging.debug("ACT inference failure", exc_info=True)
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 2


def _validate_cli_safety(args: argparse.Namespace, manifest: InferenceManifest, fps: float) -> None:
    _require_float_range("--fps", fps, _MIN_FPS, _MAX_FPS)
    if not math.isclose(fps, manifest.fps, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(f"--fps {fps:g} must equal training FPS {manifest.fps:g}")
    _require_float_range("--duration", args.duration, 0.0, _MAX_DURATION_S, lower_open=True)
    _require_float_range(
        "--max-relative-target",
        args.max_relative_target,
        0.0,
        _MAX_RELATIVE_TARGET_RAD,
        lower_open=True,
    )
    _require_float_range(
        "--max-ee-translation-step-m",
        args.max_ee_translation_step_m,
        0.0,
        _MAX_EE_TRANSLATION_STEP_M,
        lower_open=True,
    )
    _require_float_range(
        "--max-ee-rotation-step-rad",
        args.max_ee_rotation_step_rad,
        0.0,
        _MAX_EE_ROTATION_STEP_RAD,
        lower_open=True,
    )
    _require_float_range(
        "--servo-max-velocity",
        args.servo_max_velocity,
        0.0,
        _MAX_SERVO_VELOCITY_RAD_S,
        lower_open=True,
    )
    _require_float_range(
        "--servo-max-step-rad", args.servo_max_step_rad, 0.0, _MAX_SERVO_STEP_RAD, lower_open=True
    )
    _require_float_range(
        "--servo-max-lag-rad", args.servo_max_lag_rad, 0.0, _MAX_SERVO_LAG_RAD, lower_open=True
    )

    minimum_watchdog_ms = min(1000, max(30, math.ceil(2000.0 / fps)))
    _require_int_range("--servo-watchdog-ms", args.servo_watchdog_ms, minimum_watchdog_ms, 1000)
    _require_int_range("--gripper-effort", args.gripper_effort, 0, 32767)
    _require_int_range("--max-consecutive-overruns", args.max_consecutive_overruns, 1, 100)
    _require_int_range("--max-overruns-per-second", args.max_overruns_per_second, 1, max(1, math.ceil(fps)))
    if args.baudrate is not None:
        _require_int_range("--baudrate", args.baudrate, 1200, 10_000_000)
    if args.run and (not isinstance(args.port, str) or not args.port.strip()):
        raise ValueError("--port is required with --run")

    bounds_complete = (args.ee_workspace_min is None) == (args.ee_workspace_max is None)
    if not bounds_complete:
        raise ValueError("--ee-workspace-min and --ee-workspace-max must be set together")
    if args.ee_workspace_min is not None and args.ee_workspace_max is not None:
        for name, values in (
            ("--ee-workspace-min", args.ee_workspace_min),
            ("--ee-workspace-max", args.ee_workspace_max),
        ):
            for value in values:
                _require_float_range(name, value, -_MAX_WORKSPACE_COORDINATE_M, _MAX_WORKSPACE_COORDINATE_M)
        if any(low >= high for low, high in zip(args.ee_workspace_min, args.ee_workspace_max, strict=True)):
            raise ValueError("EE workspace min values must be lower than max values")
    if args.run and manifest.action_mode in {"ee_delta", "ee_pose"} and args.ee_workspace_min is None:
        raise ValueError("EE policy execution requires explicit workspace min/max bounds")
    _validate_start_state_args(args, manifest)


def _validate_start_state_args(args: argparse.Namespace, manifest: InferenceManifest) -> None:
    _require_float_range(
        "--start-joint-tolerance-rad",
        args.start_joint_tolerance_rad,
        0.0,
        0.5,
        lower_open=True,
    )
    _require_float_range(
        "--start-ee-translation-tolerance-m",
        args.start_ee_translation_tolerance_m,
        0.0,
        0.25,
        lower_open=True,
    )
    _require_float_range(
        "--start-ee-rotation-tolerance-rad",
        args.start_ee_rotation_tolerance_rad,
        0.0,
        math.pi,
        lower_open=True,
    )
    _require_float_range(
        "--start-gripper-tolerance-rad",
        args.start_gripper_tolerance_rad,
        0.0,
        math.tau,
        lower_open=True,
    )
    if args.start_joints is not None and args.start_ee is not None:
        raise ValueError("--start-joints and --start-ee are mutually exclusive")

    joint_names = tuple(f"{name}.pos" for name in JOINT_NAMES)
    ee_names = tuple(f"ee.{name}" for name in EE_COMPONENTS)
    has_joint_pose = all(name in manifest.state_names for name in joint_names)
    has_ee_pose = all(name in manifest.state_names for name in ee_names)

    if args.start_joints is not None:
        if not has_joint_pose:
            raise ValueError("--start-joints requires joint position fields in observation.state")
        for value in args.start_joints[:6]:
            _require_float_range("--start-joints", value, -math.tau, math.tau)
        _require_float_range("--start-joints gripper", args.start_joints[6], 0.0, math.tau)

    if args.start_ee is not None:
        if not has_ee_pose:
            raise ValueError("--start-ee requires EE pose fields in observation.state")
        for value in args.start_ee[:3]:
            _require_float_range(
                "--start-ee position",
                value,
                -_MAX_WORKSPACE_COORDINATE_M,
                _MAX_WORKSPACE_COORDINATE_M,
            )
        for value in args.start_ee[3:6]:
            _require_float_range("--start-ee rotvec", value, -math.pi, math.pi)
        rotation_norm = math.sqrt(sum(value * value for value in args.start_ee[3:6]))
        if rotation_norm > math.pi:
            raise ValueError("--start-ee rotation vector norm must not exceed pi")
        _require_float_range("--start-ee gripper", args.start_ee[6], 0.0, math.tau)

    if args.run:
        if has_joint_pose and args.start_joints is None:
            raise ValueError("--start-joints is required with --run for this observation schema")
        if not has_joint_pose and has_ee_pose and args.start_ee is None:
            raise ValueError("--start-ee is required with --run for this observation schema")
        if not has_joint_pose and not has_ee_pose:
            raise ValueError("manifest observation.state has no supported start-pose fields")


def _build_initial_observation_validator(
    args: argparse.Namespace,
    manifest: InferenceManifest,
) -> Callable[[dict[str, Any]], None] | None:
    if args.start_joints is not None:
        names = tuple(f"{name}.pos" for name in JOINT_NAMES)
        expected_joints = tuple(args.start_joints[:6])
        expected_gripper = float(args.start_joints[6])

        def validate_joint_start(observation: dict[str, Any]) -> None:
            errors = [
                abs(float(observation[name]) - expected)
                for name, expected in zip(names, expected_joints, strict=True)
            ]
            gripper_error = abs(float(observation["gripper.pos"]) - expected_gripper)
            if max(errors) > args.start_joint_tolerance_rad or (
                gripper_error > args.start_gripper_tolerance_rad
            ):
                raise RuntimeError(
                    "initial joint/gripper state is outside the authorized start envelope: "
                    f"max joint error={max(errors):.4f} rad, gripper error={gripper_error:.4f} rad"
                )

        return validate_joint_start

    if args.start_ee is not None:
        expected_position = tuple(args.start_ee[:3])
        expected_rotation = rotation_vector_to_matrix(args.start_ee[3:6])
        expected_gripper = float(args.start_ee[6])

        def validate_ee_start(observation: dict[str, Any]) -> None:
            actual_position = tuple(float(observation[f"ee.{axis}"]) for axis in ("x", "y", "z"))
            actual_rotation = rotation_vector_to_matrix(
                [observation[f"ee.{axis}"] for axis in ("wx", "wy", "wz")]
            )
            translation_error = math.dist(actual_position, expected_position)
            rotation_delta = rotation_matrix_to_rotvec(expected_rotation.T @ actual_rotation)
            rotation_error = math.sqrt(sum(float(value) ** 2 for value in rotation_delta))
            gripper_error = abs(float(observation["gripper.pos"]) - expected_gripper)
            if (
                translation_error > args.start_ee_translation_tolerance_m
                or rotation_error > args.start_ee_rotation_tolerance_rad
                or gripper_error > args.start_gripper_tolerance_rad
            ):
                raise RuntimeError(
                    "initial EE/gripper state is outside the authorized start envelope: "
                    f"translation error={translation_error:.4f} m, "
                    f"rotation error={rotation_error:.4f} rad, "
                    f"gripper error={gripper_error:.4f} rad"
                )

        return validate_ee_start

    return None


def _validate_warmup_latency(latency_s: float, fps: float, watchdog_ms: int) -> float:
    if not math.isfinite(latency_s) or latency_s < 0:
        raise RuntimeError("ACT warmup returned an invalid latency")
    budget_s = min(0.8 / fps, watchdog_ms / 1000.0 * 0.8)
    if latency_s >= budget_s:
        raise RuntimeError(
            f"warmed ACT full-chunk latency {latency_s * 1e3:.1f} ms exceeds the "
            f"real-time budget {budget_s * 1e3:.1f} ms for {fps:g} FPS"
        )
    return budget_s


def _require_float_range(
    name: str,
    value: Any,
    minimum: float,
    maximum: float,
    *,
    lower_open: bool = False,
) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite")
    below = value <= minimum if lower_open else value < minimum
    if below or value > maximum:
        opening = "(" if lower_open else "["
        raise ValueError(f"{name} must be in {opening}{minimum:g}, {maximum:g}]")


def _require_int_range(name: str, value: Any, minimum: int, maximum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")


def _build_robot(
    args: argparse.Namespace,
    manifest: InferenceManifest,
    cameras: dict[str, Any],
    fps: float,
) -> FafuFollower:
    observation_mode, record_velocity, record_effort = derive_observation_settings(manifest)
    config = FafuFollowerConfig(
        id=args.robot_id,
        calibration_dir=args.calibration_dir,
        sdk_path=args.sdk_path,
        sdk_config_path=args.sdk_config_path,
        port=args.port,
        baudrate=args.baudrate,
        cameras=cameras,
        action_mode=manifest.action_mode,
        observation_mode=observation_mode,
        record_joint_velocity=record_velocity,
        record_motor_effort=record_effort,
        max_ee_translation_step_m=args.max_ee_translation_step_m,
        max_ee_rotation_step_rad=args.max_ee_rotation_step_rad,
        ee_workspace_min=tuple(args.ee_workspace_min) if args.ee_workspace_min else None,
        ee_workspace_max=tuple(args.ee_workspace_max) if args.ee_workspace_max else None,
        servo_watchdog_ms=args.servo_watchdog_ms,
        servo_max_velocity=args.servo_max_velocity,
        servo_max_step_rad=args.servo_max_step_rad,
        servo_max_lag_rad=args.servo_max_lag_rad,
        servo_rate_hz=fps,
        max_relative_target=args.max_relative_target,
        strict_action_features=True,
        write_sent_action_back=True,
        gripper_effort=args.gripper_effort,
        joint_release="brake",
        urdf_path=args.urdf_path,
    )
    return FafuFollower(config)


def _preflight_report(
    args: argparse.Namespace,
    runtime: ActPolicyRuntime,
    manifest: InferenceManifest,
    fps: float,
    warmup_latency: float,
    realtime_budget: float,
) -> dict[str, Any]:
    return {
        "ok": True,
        "checkpoint": str(args.checkpoint.expanduser().resolve()),
        "policy_type": runtime.policy.config.type,
        "device": str(runtime.device),
        "action_mode": manifest.action_mode,
        "action_names": list(manifest.action_names),
        "state_names": list(manifest.state_names),
        "cameras": {key: feature["shape"] for key, feature in manifest.camera_features.items()},
        "fps": fps,
        "synthetic_full_chunk_latency_ms": warmup_latency * 1e3,
        "realtime_budget_ms": realtime_budget * 1e3,
        "hardware_connected": False,
    }


def _print_preflight(report: dict[str, Any]) -> None:
    print(f"[OK] checkpoint: {report['checkpoint']}")
    print(f"[OK] policy / device: {report['policy_type']} / {report['device']}")
    print(f"[OK] action mode / fields: {report['action_mode']} / {len(report['action_names'])}")
    print(f"[OK] state fields / cameras: {len(report['state_names'])} / {len(report['cameras'])}")
    print(f"[OK] training/inference FPS: {report['fps']:g}")
    print(
        f"[OK] synthetic full-chunk latency / budget: "
        f"{report['synthetic_full_chunk_latency_ms']:.1f} / {report['realtime_budget_ms']:.1f} ms"
    )


if __name__ == "__main__":
    raise SystemExit(main())
