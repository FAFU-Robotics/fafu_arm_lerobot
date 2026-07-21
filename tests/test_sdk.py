from pathlib import Path

from lerobot_robot_fafu_arm.sdk import default_sdk_config_path, sdk_search_paths


def test_sdk_path_accepts_repo_or_python_directory(tmp_path: Path):
    sdk_root = tmp_path / "fafu_arm_sdk"
    python_dir = sdk_root / "fafu_robot_python"
    python_dir.mkdir(parents=True)
    (python_dir / "fafu_robot_controller.py").write_text("# test\n", encoding="utf-8")

    assert sdk_search_paths(sdk_root)[0] == python_dir.resolve()
    assert sdk_search_paths(python_dir)[0] == python_dir.resolve()


def test_default_controller_config_is_packaged():
    config = default_sdk_config_path()
    assert config.is_file()
    text = config.read_text(encoding="utf-8")
    assert "motor_ids = 1, 2, 3, 4, 5, 6, 7" in text
    assert "limits.7 = 0.0000, 105.0000" in text
