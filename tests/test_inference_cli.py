from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from lerobot_robot_fafu_arm.inference import cli as inference_cli
from lerobot_robot_fafu_arm.inference.cli import (
    _validate_cli_safety,
    _validate_warmup_latency,
    build_act_parser,
    build_parser,
    load_camera_configs,
)
from lerobot_robot_fafu_arm.inference.manifest import InferenceManifest
from lerobot_robot_fafu_arm.representation import EE_COMPONENTS, JOINT_NAMES, action_features


def _manifest(mode: str = "joint", state: str = "joint") -> InferenceManifest:
    if state == "joint":
        state_names = [f"{name}.pos" for name in JOINT_NAMES] + ["gripper.pos"]
    elif state == "ee_pose":
        state_names = [f"ee.{name}" for name in EE_COMPONENTS] + ["gripper.pos"]
    else:
        raise ValueError(f"unsupported test state: {state}")
    return InferenceManifest(
        action_mode=mode,
        robot_type="fafu_follower",
        fps=30,
        features={
            "observation.state": {
                "dtype": "float32",
                "shape": [len(state_names)],
                "names": state_names,
            },
            "observation.images.front": {
                "dtype": "video",
                "shape": [480, 640, 3],
                "names": ["height", "width", "channels"],
            },
            "action": {
                "dtype": "float32",
                "shape": [7],
                "names": list(action_features(mode)),
            },
        },
    )


def _standalone_args(*extra: str):
    return build_act_parser().parse_args(["--checkpoint", "checkpoint", "--cameras", "cameras.yaml", *extra])


def test_console_and_example_parsers_use_their_documented_syntax():
    standalone = _standalone_args()
    console = build_parser().parse_args(["act", "--checkpoint", "checkpoint", "--cameras", "cameras.yaml"])

    assert standalone.checkpoint == Path("checkpoint")
    assert console.algorithm == "act"


def test_example_camera_config_decodes_with_lerobot():
    path = Path(__file__).parents[1] / "configs" / "inference" / "opencv_camera.yaml"

    cameras = load_camera_configs(path)

    assert tuple(cameras) == ("front",)
    assert cameras["front"].width == 640
    assert cameras["front"].height == 480
    assert cameras["front"].fps == 30


def test_hardware_run_requires_port_and_cartesian_workspace():
    joint_args = _standalone_args("--run")
    with pytest.raises(ValueError, match="--port is required"):
        _validate_cli_safety(joint_args, _manifest(), 30)

    ee_args = _standalone_args("--run", "--port", "COM7")
    with pytest.raises(ValueError, match="workspace min/max"):
        _validate_cli_safety(ee_args, _manifest("ee_delta"), 30)


def test_hardware_run_requires_and_accepts_declared_start_state():
    missing_start = _standalone_args("--run", "--port", "COM7")
    with pytest.raises(ValueError, match="--start-joints is required"):
        _validate_cli_safety(missing_start, _manifest(), 30)

    valid_joint = _standalone_args(
        "--run",
        "--port",
        "COM7",
        "--start-joints",
        "0",
        "0",
        "0",
        "0",
        "0",
        "0",
        "0",
    )
    _validate_cli_safety(valid_joint, _manifest(), 30)

    valid_ee = _standalone_args(
        "--run",
        "--port",
        "COM7",
        "--start-ee",
        "0.4",
        "0",
        "0.3",
        "0",
        "0",
        "0",
        "0",
        "--ee-workspace-min",
        "0.2",
        "-0.5",
        "0.1",
        "--ee-workspace-max",
        "0.8",
        "0.5",
        "0.8",
    )
    _validate_cli_safety(valid_ee, _manifest("ee_pose", state="ee_pose"), 30)


def test_initial_joint_validator_rejects_state_outside_envelope():
    args = _standalone_args(
        "--start-joints",
        "0",
        "0",
        "0",
        "0",
        "0",
        "0",
        "0",
    )
    manifest = _manifest()
    validator = inference_cli._build_initial_observation_validator(args, manifest)
    assert validator is not None
    observation = {name: 0.0 for name in manifest.state_names}
    validator(observation)

    observation["joint1.pos"] = 0.2
    with pytest.raises(RuntimeError, match="outside the authorized start envelope"):
        validator(observation)


def test_initial_ee_validator_compares_rotation_geometry_not_rotvec_sign():
    args = _standalone_args(
        "--start-ee",
        "0.4",
        "0",
        "0.3",
        "0",
        "0",
        "3.141592653589793",
        "0",
    )
    manifest = _manifest("ee_pose", state="ee_pose")
    validator = inference_cli._build_initial_observation_validator(args, manifest)
    assert validator is not None
    observation = {name: 0.0 for name in manifest.state_names}
    observation.update({"ee.x": 0.4, "ee.z": 0.3, "ee.wz": -3.141592653589793})

    validator(observation)


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (("--max-relative-target", "nan"), "must be finite"),
        (("--servo-max-lag-rad", "0"), "must be in"),
        (("--servo-watchdog-ms", "0"), "must be an integer"),
        (("--gripper-effort", "-1"), "must be an integer"),
        (("--max-overruns-per-second", "31"), "must be an integer"),
        (("--duration", "3601"), "must be in"),
    ],
)
def test_cli_rejects_unsafe_numeric_limits(extra, message):
    args = _standalone_args(*extra)

    with pytest.raises(ValueError, match=message):
        _validate_cli_safety(args, _manifest(), 30)


def test_warmup_latency_must_fit_control_period_not_only_watchdog():
    assert _validate_warmup_latency(0.01, 30, 250) == pytest.approx(0.8 / 30)

    with pytest.raises(RuntimeError, match="real-time budget"):
        _validate_warmup_latency(0.027, 30, 250)


def test_dry_run_completes_without_calling_robot_connect(tmp_path, monkeypatch, capsys):
    manifest = _manifest()
    connected = []
    runtime = SimpleNamespace(
        policy=SimpleNamespace(config=SimpleNamespace(type="act")),
        device="cpu",
        synthetic_warmup=lambda: 0.001,
    )
    robot = SimpleNamespace(connect=lambda: connected.append(True))

    monkeypatch.setattr(inference_cli, "load_inference_manifest", lambda *args, **kwargs: manifest)
    monkeypatch.setattr(inference_cli, "load_camera_configs", lambda path: {"front": object()})
    monkeypatch.setattr(
        inference_cli,
        "ActPolicyRuntime",
        SimpleNamespace(load=lambda *args, **kwargs: runtime),
    )
    monkeypatch.setattr(inference_cli, "_build_robot", lambda *args, **kwargs: robot)
    monkeypatch.setattr(inference_cli, "validate_robot_schema", lambda *args, **kwargs: None)

    result = inference_cli.act_main(
        [
            "--checkpoint",
            str(tmp_path / "checkpoint"),
            "--cameras",
            str(tmp_path / "cameras.yaml"),
        ]
    )

    assert result == 0
    assert not connected
    assert "[DRY-RUN]" in capsys.readouterr().out


def test_inference_robot_forces_brake_release(monkeypatch):
    args = _standalone_args()
    camera = SimpleNamespace(
        height=480,
        width=640,
        fps=30,
        use_rgb=True,
        use_depth=False,
    )
    monkeypatch.setattr(inference_cli, "FafuFollower", lambda config: config)

    config = inference_cli._build_robot(
        args,
        _manifest(),
        {"front": camera},
        30,
    )

    assert config.joint_release == "brake"
