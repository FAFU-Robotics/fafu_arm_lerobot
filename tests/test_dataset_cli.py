from __future__ import annotations

import json

from lerobot_robot_fafu_arm.dataset_cli import inspect_dataset, main


def write_info(root, *, action_names, total_episodes=2):
    meta = root / "meta"
    meta.mkdir()
    info = {
        "robot_type": "fafu_follower",
        "fps": 30,
        "total_episodes": total_episodes,
        "features": {
            "action": {"dtype": "float32", "shape": [len(action_names)], "names": action_names},
            "observation.images.front": {"dtype": "video", "shape": [480, 640, 3]},
        },
    }
    (meta / "info.json").write_text(json.dumps(info), encoding="utf-8")


def test_dataset_check_accepts_matching_joint_contract(tmp_path):
    names = [f"joint{index}.pos" for index in range(1, 7)] + ["gripper.pos"]
    write_info(tmp_path, action_names=names)

    report = inspect_dataset(tmp_path, "joint", 1)

    assert report["ok"]
    assert report["camera_features"] == ["observation.images.front"]


def test_dataset_check_rejects_wrong_mode_and_episode(tmp_path, capsys):
    names = [f"joint{index}.pos" for index in range(1, 7)] + ["gripper.pos"]
    write_info(tmp_path, action_names=names, total_episodes=1)

    exit_code = main(["--root", str(tmp_path), "--action-mode", "ee_delta", "--episode", "3"])

    assert exit_code == 2
    captured = capsys.readouterr()
    assert "missing action fields" in captured.err
    assert "out of range" in captured.err


def test_dataset_check_reports_missing_metadata(tmp_path):
    report = inspect_dataset(tmp_path, "joint", 0)
    assert not report["ok"]
    assert "metadata file not found" in report["errors"][0]
