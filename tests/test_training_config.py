from __future__ import annotations

import pytest

from lerobot_robot_fafu_arm.training.config_file import TrainingConfigError, load_act_yaml


def test_load_act_yaml_maps_policy_tuning_and_privacy(tmp_path):
    config = tmp_path / "act.yaml"
    config.write_text(
        """
schema_version: 1
algorithm: act
dataset:
  repo_id: FAFU-Robotics/demo
  root: ./datasets/demo
  action_mode: joint
run:
  output_dir: ./outputs/demo
  steps: 50000
policy:
  type: fafu_act_demo
  chunk_size: 60
  optimizer_lr: 0.00003
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

    values = load_act_yaml(config)

    assert values["policy_type"] == "fafu_act_demo"
    assert values["chunk_size"] == 60
    assert values["steps"] == 50_000
    assert values["push_to_hub"] is False
    assert "policy.optimizer_lr=3e-05" in values["extra_args"]
    assert "policy.residual_hidden_dim=128" in values["extra_args"]


def test_load_act_yaml_rejects_typos(tmp_path):
    config = tmp_path / "bad.yaml"
    config.write_text(
        """
schema_version: 1
algorithm: act
dataset:
  repo_id: FAFU-Robotics/demo
  root: ./datasets/demo
  action_mode: joint
policy:
  chunk_sze: 100
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(TrainingConfigError, match="chunk_sze"):
        load_act_yaml(config)


def test_load_act_yaml_rejects_wrong_known_parameter_type(tmp_path):
    config = tmp_path / "bad_type.yaml"
    config.write_text(
        """
schema_version: 1
algorithm: act
dataset:
  repo_id: FAFU-Robotics/demo
  root: ./datasets/demo
  action_mode: joint
run:
  output_dir: ./outputs/demo
policy:
  chunk_size: 100
  dropout: high
tracking:
  wandb: false
hub:
  push_to_hub: false
  public: false
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(TrainingConfigError, match="policy.dropout"):
        load_act_yaml(config)
