from __future__ import annotations

import csv
import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from lerobot_robot_fafu_arm.dataset_cli import collection_main
from lerobot_robot_fafu_arm.kinematics import FafuArmKinematics
from lerobot_robot_fafu_arm.local_dataset import (
    DatasetReadError,
    export_episode_csv,
    load_dataset_info,
    load_episode,
)
from lerobot_robot_fafu_arm.wrs_viewer import UrdfSkeleton
from lerobot_robot_fafu_arm.wrs_viewer import main as wrs_main


def make_dataset(root):
    state_names = [f"joint{index}.pos" for index in range(1, 7)] + [
        "gripper.pos",
        "ee.x",
        "ee.y",
        "ee.z",
    ]
    action_names = [f"joint{index}.pos" for index in range(1, 7)] + ["gripper.pos"]
    info = {
        "codebase_version": "v3.0",
        "robot_type": "fafu_follower",
        "fps": 30,
        "total_episodes": 2,
        "total_frames": 3,
        "features": {
            "observation.state": {
                "dtype": "float32",
                "shape": [len(state_names)],
                "names": state_names,
            },
            "action": {
                "dtype": "float32",
                "shape": [len(action_names)],
                "names": action_names,
            },
            "observation.images.front": {"dtype": "video", "shape": [480, 640, 3]},
        },
    }
    meta = root / "meta"
    data = root / "data" / "chunk-000"
    meta.mkdir(parents=True)
    data.mkdir(parents=True)
    (meta / "info.json").write_text(json.dumps(info), encoding="utf-8")

    states = [
        [0.0] * len(state_names),
        [0.1, 0.2, 0.3, 0.1, -0.1, 0.2, 0.4, 0.2, 0.0, 0.3],
        [0.5] * len(state_names),
    ]
    actions = [
        [0.01] * len(action_names),
        [0.02] * len(action_names),
        [0.50] * len(action_names),
    ]
    table = pa.table(
        {
            "timestamp": [0.0, 1 / 30, 0.0],
            "frame_index": [0, 1, 0],
            "episode_index": [0, 0, 1],
            "index": [0, 1, 2],
            "task_index": [0, 0, 0],
            "observation.state": states,
            "action": actions,
        }
    )
    pq.write_table(table, data / "file-000.parquet")
    return state_names, action_names


def test_load_episode_and_extract_semantic_joint_trajectory(tmp_path):
    make_dataset(tmp_path)

    info = load_dataset_info(tmp_path)
    episode = load_episode(tmp_path, 0)

    assert info.total_episodes == 2
    assert len(episode) == 2
    np.testing.assert_allclose(episode.joint_trajectory("observation")[1], [0.1, 0.2, 0.3, 0.1, -0.1, 0.2])
    np.testing.assert_allclose(episode.joint_trajectory("action")[0], np.full(6, 0.01))
    assert "observation.images.front" not in episode.columns


def test_export_episode_csv_is_flat_and_does_not_overwrite(tmp_path):
    _, action_names = make_dataset(tmp_path / "dataset")
    episode = load_episode(tmp_path / "dataset", 0)
    output = export_episode_csv(episode, tmp_path / "episode.csv")

    with output.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 2
    assert rows[0][f"action.{action_names[0]}"] == "0.01"

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        export_episode_csv(episode, output)


def test_episode_bounds_are_validated_before_reading_parquet(tmp_path):
    make_dataset(tmp_path)
    with pytest.raises(DatasetReadError, match="out of range"):
        load_episode(tmp_path, 2)


def test_unified_dataset_cli_info_preview_and_export(tmp_path, capsys):
    dataset = tmp_path / "dataset"
    make_dataset(dataset)

    assert collection_main(["info", "--root", str(dataset)]) == 0
    assert "episodes / frames: 2 / 3" in capsys.readouterr().out

    assert collection_main(["preview", "--root", str(dataset), "--episode", "0", "--rows", "1"]) == 0
    assert "showing 1 row" in capsys.readouterr().out

    output = tmp_path / "episode.csv"
    assert collection_main(["export", "--root", str(dataset), "--episode", "0", "--output", str(output)]) == 0
    assert output.is_file()


def test_wrs_urdf_skeleton_matches_pytracik_tcp():
    skeleton = UrdfSkeleton()
    kinematics = FafuArmKinematics()
    joints = np.array([0.2, 0.6, 1.1, -0.2, 0.3, -0.4])

    position, rotation = skeleton.frames(joints)[-1]
    expected = kinematics.forward(joints)

    np.testing.assert_allclose(position, expected.position, atol=1e-9)
    np.testing.assert_allclose(rotation, expected.rotation, atol=1e-9)


def test_wrs_dry_run_does_not_import_wrs(tmp_path, capsys):
    make_dataset(tmp_path)
    code = wrs_main(["--root", str(tmp_path), "--episode", "0", "--dry-run"])

    assert code == 0
    assert "TCP bounds" in capsys.readouterr().out
