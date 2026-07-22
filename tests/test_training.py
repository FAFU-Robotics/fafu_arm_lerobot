from __future__ import annotations

import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from lerobot_robot_fafu_arm.training.act import ActTrainConfig, build_act_command
from lerobot_robot_fafu_arm.training.cli import main
from lerobot_robot_fafu_arm.training.common import check_training_dataset


def write_training_dataset(root, *, action_names, episodes=50, camera_shapes=None):
    camera_shapes = camera_shapes or {"front": [3, 480, 640]}
    meta = root / "meta"
    data = root / "data" / "chunk-000"
    meta.mkdir(parents=True)
    data.mkdir(parents=True)
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": [7],
            "names": [f"joint{i}.pos" for i in range(1, 7)] + ["gripper.pos"],
        },
        "action": {
            "dtype": "float32",
            "shape": [len(action_names)],
            "names": action_names,
        },
    }
    features.update(
        {
            f"observation.images.{name}": {"dtype": "video", "shape": shape}
            for name, shape in camera_shapes.items()
        }
    )
    info = {
        "robot_type": "fafu_follower",
        "codebase_version": "v3.0",
        "fps": 30,
        "total_episodes": episodes,
        "total_frames": 2,
        "features": features,
    }
    (meta / "info.json").write_text(json.dumps(info), encoding="utf-8")
    (meta / "stats.json").write_text("{}", encoding="utf-8")
    table = pa.table(
        {
            "episode_index": [0, 0],
            "frame_index": [0, 1],
            "observation.state": [[0.0] * 7, [0.1] * 7],
            "action": [[0.0] * len(action_names), [0.1] * len(action_names)],
        }
    )
    pq.write_table(table, data / "file-000.parquet")


def joint_names():
    return [f"joint{i}.pos" for i in range(1, 7)] + ["gripper.pos"]


def test_training_preflight_accepts_joint_dataset(tmp_path):
    write_training_dataset(tmp_path, action_names=joint_names())

    report = check_training_dataset(tmp_path, "joint")

    assert report.ok
    assert report.sampled_episode == 0
    assert report.camera_features == ("observation.images.front",)


def test_training_preflight_rejects_wrong_action_and_camera_shape(tmp_path):
    write_training_dataset(
        tmp_path,
        action_names=joint_names(),
        camera_shapes={"front": [3, 480, 640], "wrist": [3, 240, 320]},
    )

    report = check_training_dataset(tmp_path, "ee_delta")

    assert not report.ok
    assert any("missing fields" in error for error in report.errors)
    assert any("same shape" in error for error in report.errors)


def test_act_command_is_local_and_private_by_default(tmp_path):
    config = ActTrainConfig(
        dataset_repo_id="FAFU-Robotics/demo",
        dataset_root=tmp_path / "dataset",
        output_dir=tmp_path / "output",
        action_mode="joint",
    )

    command = build_act_command(config)

    assert "--policy.type=act" in command
    assert "--policy.push_to_hub=false" in command
    assert "--wandb.enable=false" in command
    assert "--policy.n_action_steps=10" in command
    assert not any(argument.startswith("--policy.repo_id=") for argument in command)


def test_act_command_upload_is_private_unless_explicitly_public(tmp_path):
    config = ActTrainConfig(
        dataset_repo_id="FAFU-Robotics/demo",
        dataset_root=tmp_path / "dataset",
        output_dir=tmp_path / "output",
        action_mode="ee_pose",
        push_to_hub=True,
        policy_repo_id="FAFU-Robotics/act-demo",
    )

    command = build_act_command(config)

    assert "--policy.private=true" in command
    assert "--policy.repo_id=FAFU-Robotics/act-demo" in command


def test_act_command_supports_out_of_tree_policy(tmp_path):
    config = ActTrainConfig(
        dataset_repo_id="FAFU-Robotics/demo",
        dataset_root=tmp_path / "dataset",
        output_dir=tmp_path / "output",
        action_mode="joint",
        policy_type="fafu_act_demo",
        extra_args=("policy.residual_hidden_dim=128",),
    )

    command = build_act_command(config)

    assert "--policy.type=fafu_act_demo" in command
    assert "--policy.residual_hidden_dim=128" in command
    assert "--job_name=fafu_act_demo_fafu_joint" in command


def test_act_config_rejects_unsafe_or_inconsistent_options(tmp_path):
    base = {
        "dataset_repo_id": "FAFU-Robotics/demo",
        "dataset_root": tmp_path / "dataset",
        "output_dir": tmp_path / "output",
        "action_mode": "joint",
    }

    with pytest.raises(ValueError, match="requires policy_repo_id"):
        ActTrainConfig(**base, push_to_hub=True).validate()
    with pytest.raises(ValueError, match="n_action_steps=1"):
        ActTrainConfig(**base, temporal_ensemble_coeff=0.01).validate()
    with pytest.raises(ValueError, match="dedicated option"):
        ActTrainConfig(**base, extra_args=("policy.push_to_hub=true",)).validate()
    with pytest.raises(ValueError, match="lowercase"):
        ActTrainConfig(**base, policy_type="Bad/Policy").validate()


def test_training_cli_is_a_dry_run_by_default(tmp_path, capsys):
    dataset = tmp_path / "dataset"
    output = tmp_path / "output"
    write_training_dataset(dataset, action_names=joint_names())

    exit_code = main(
        [
            "act",
            "--dataset-root",
            str(dataset),
            "--dataset-repo-id",
            "FAFU-Robotics/demo",
            "--action-mode",
            "joint",
            "--output-dir",
            str(output),
        ]
    )

    assert exit_code == 0
    assert "[DRY-RUN]" in capsys.readouterr().out
    assert not output.exists()


def test_training_cli_loads_yaml_and_cli_overrides_it(tmp_path, capsys):
    dataset = tmp_path / "dataset"
    output = tmp_path / "output"
    write_training_dataset(dataset, action_names=joint_names())
    config_file = tmp_path / "act.yaml"
    config_file.write_text(
        f"""
schema_version: 1
algorithm: act
dataset:
  repo_id: FAFU-Robotics/demo
  root: {json.dumps(str(dataset))}
  action_mode: joint
run:
  output_dir: {json.dumps(str(output))}
  steps: 50000
policy:
  type: fafu_act_demo
  chunk_size: 60
  n_action_steps: 5
  extra:
    residual_hidden_dim: 128
tracking:
  wandb: false
hub:
  push_to_hub: false
  public: false
""".strip(),
        encoding="utf-8",
    )

    exit_code = main(
        [
            "act",
            "--config",
            str(config_file),
            "--steps",
            "1234",
        ]
    )

    output_text = capsys.readouterr().out
    assert exit_code == 0
    assert "--policy.type=fafu_act_demo" in output_text
    assert "--policy.residual_hidden_dim=128" in output_text
    assert "--steps=1234" in output_text
    assert "--steps=50000" not in output_text
    assert not output.exists()
