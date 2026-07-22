# ruff: noqa: E402
from __future__ import annotations

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("lerobot")

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.act.processor_act import make_act_pre_post_processors

from lerobot_robot_fafu_arm.inference import cli as inference_cli
from lerobot_robot_fafu_arm.inference.act import ActPolicyRuntime
from lerobot_robot_fafu_arm.inference.manifest import InferenceManifest, write_inference_manifest
from lerobot_robot_fafu_arm.representation import JOINT_NAMES, action_features


def _state_names() -> tuple[str, ...]:
    return (
        *(f"{name}.pos" for name in JOINT_NAMES),
        "gripper.pos",
        *(f"{name}.vel" for name in JOINT_NAMES),
        "gripper.vel",
    )


def _manifest(action_mode: str, *, fps: float = 30.0) -> InferenceManifest:
    state_names = _state_names()
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": [14],
            "names": list(state_names),
        },
        "observation.images.front": {
            "dtype": "video",
            "shape": [64, 64, 3],
            "names": ["height", "width", "channels"],
        },
        "action": {
            "dtype": "float32",
            "shape": [7],
            "names": list(action_features(action_mode)),
        },
    }
    return InferenceManifest(
        action_mode=action_mode,
        robot_type="fafu_follower",
        fps=fps,
        features=features,
    )


def _tiny_config(config_class=ACTConfig):
    return config_class(
        device="cpu",
        input_features={
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(14,)),
            "observation.images.front": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 64, 64)),
        },
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(7,))},
        chunk_size=2,
        n_action_steps=1,
        dim_model=32,
        n_heads=4,
        dim_feedforward=64,
        n_encoder_layers=1,
        n_decoder_layers=1,
        use_vae=False,
        pretrained_backbone_weights=None,
    )


def _dataset_stats():
    return {
        "observation.state": {
            "mean": torch.zeros(14),
            "std": torch.ones(14),
            "min": -torch.ones(14),
            "max": torch.ones(14),
        },
        "action": {
            "mean": torch.zeros(7),
            "std": torch.ones(7),
            "min": -torch.ones(7),
            "max": torch.ones(7),
        },
    }


def _save_checkpoint(checkpoint, manifest, config, policy, processor_factory):
    checkpoint = checkpoint / "pretrained_model"
    preprocessor, postprocessor = processor_factory(config, dataset_stats=_dataset_stats())
    policy.save_pretrained(checkpoint)
    preprocessor.save_pretrained(checkpoint)
    postprocessor.save_pretrained(checkpoint)
    write_inference_manifest(manifest, checkpoint)
    return checkpoint


def _observation():
    observation = {name: 0.0 for name in _state_names()}
    observation["front"] = np.zeros((64, 64, 3), dtype=np.uint8)
    return observation


@pytest.mark.parametrize("action_mode", ["joint", "ee_delta", "ee_pose"])
def test_real_act_checkpoint_processor_round_trip_for_every_action_mode(tmp_path, action_mode):
    manifest = _manifest(action_mode)
    config = _tiny_config()
    policy = ACTPolicy(config).eval()
    checkpoint = _save_checkpoint(tmp_path, manifest, config, policy, make_act_pre_post_processors)

    runtime = ActPolicyRuntime.load(checkpoint, manifest, device="cpu", task="smoke")
    action = runtime.predict(_observation())

    assert runtime.policy.name == "act"
    assert tuple(action) == tuple(action_features(action_mode))
    assert tuple(action) == manifest.action_names
    assert np.isfinite(list(action.values())).all()


def test_real_act_json_dry_run_stdout_is_strict_json_without_connecting_camera_or_robot(
    tmp_path,
    monkeypatch,
    capsys,
):
    manifest = _manifest("joint", fps=10.0)
    config = _tiny_config()
    policy = ACTPolicy(config).eval()
    checkpoint = _save_checkpoint(tmp_path, manifest, config, policy, make_act_pre_post_processors)
    camera_path = tmp_path / "cameras.yaml"
    camera_path.write_text(
        """front:
  type: opencv
  index_or_path: 0
  width: 64
  height: 64
  fps: 10
""",
        encoding="utf-8",
    )

    def fail_if_hardware_connects(*args, **kwargs):
        pytest.fail("--json dry-run must not connect the camera or robot")

    monkeypatch.setattr(inference_cli.FafuFollower, "connect", fail_if_hardware_connects)
    capsys.readouterr()

    result = inference_cli.main(
        [
            "act",
            "--checkpoint",
            str(checkpoint),
            "--cameras",
            str(camera_path),
            "--calibration-dir",
            str(tmp_path / "calibration"),
            "--device",
            "cpu",
            "--fps",
            "10",
            "--servo-watchdog-ms",
            "1000",
            "--json",
        ]
    )
    captured = capsys.readouterr()

    assert result == 0, captured.err
    report = json.loads(captured.out)
    assert report["ok"] is True
    assert report["policy_type"] == "act"
    assert report["hardware_connected"] is False


def test_real_fafu_act_demo_checkpoint_strict_load_processor_and_predict(tmp_path):
    demo = pytest.importorskip(
        "lerobot_policy_fafu_act_demo",
        reason="the optional FAFU ACT demo policy is not installed",
    )
    manifest = _manifest("joint")
    config = _tiny_config(demo.FafuActDemoConfig)
    policy = demo.FafuActDemoPolicy(config).eval()
    expected_residual_parameters = policy.residual_parameter_count
    checkpoint = _save_checkpoint(
        tmp_path,
        manifest,
        config,
        policy,
        demo.make_fafu_act_demo_pre_post_processors,
    )

    runtime = ActPolicyRuntime.load(checkpoint, manifest, device="cpu", task="smoke")
    action = runtime.predict(_observation())

    assert runtime.policy.name == "fafu_act_demo"
    assert runtime.policy.residual_parameter_count == expected_residual_parameters
    assert runtime.policy.model.action_head.__class__.__name__ == "ResidualActionHead"
    assert tuple(action) == manifest.action_names
    assert np.isfinite(list(action.values())).all()
