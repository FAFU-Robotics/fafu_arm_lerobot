from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from lerobot_robot_fafu_arm.inference.manifest import MANIFEST_FILENAME
from lerobot_robot_fafu_arm.training import cli as training_cli


def write_dataset(root):
    meta = root / "meta"
    data = root / "data" / "chunk-000"
    meta.mkdir(parents=True)
    data.mkdir(parents=True)
    action_names = [f"joint{index}.pos" for index in range(1, 7)] + ["gripper.pos"]
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": [7],
            "names": action_names,
        },
        "action": {
            "dtype": "float32",
            "shape": [7],
            "names": action_names,
        },
        "observation.images.front": {
            "dtype": "video",
            "shape": [3, 480, 640],
            "names": ["channels", "height", "width"],
        },
    }
    info = {
        "robot_type": "fafu_follower",
        "codebase_version": "v3.0",
        "fps": 30,
        "total_episodes": 50,
        "total_frames": 2,
        "features": features,
    }
    (meta / "info.json").write_text(json.dumps(info), encoding="utf-8")
    (meta / "stats.json").write_text("{}", encoding="utf-8")
    pq.write_table(
        pa.table(
            {
                "episode_index": [0, 0],
                "frame_index": [0, 1],
                "observation.state": [[0.0] * 7, [0.1] * 7],
                "action": [[0.0] * 7, [0.1] * 7],
            }
        ),
        data / "file-000.parquet",
    )


def write_checkpoint(root, *, marker="checkpoint"):
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(json.dumps({"type": "act", "marker": marker}), encoding="utf-8")
    (root / "model.safetensors").write_bytes(f"model:{marker}".encode())
    (root / "policy_preprocessor.json").write_text(
        json.dumps({"name": "policy_preprocessor", "steps": []}),
        encoding="utf-8",
    )
    (root / "policy_postprocessor.json").write_text(
        json.dumps({"name": "policy_postprocessor", "steps": []}),
        encoding="utf-8",
    )
    return root


def training_args(dataset, output, *, run=True):
    args = [
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
    if run:
        args.append("--run")
    return args


def test_successful_training_writes_root_template_and_bound_checkpoint_manifest(
    tmp_path, monkeypatch, capsys
):
    dataset = tmp_path / "dataset"
    output = tmp_path / "output"
    checkpoint = output / "checkpoints" / "000100" / "pretrained_model"
    write_dataset(dataset)

    def fake_run(command, check):
        assert check is False
        write_checkpoint(checkpoint)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(training_cli.shutil, "which", lambda executable: executable)
    monkeypatch.setattr(training_cli.subprocess, "run", fake_run)

    assert training_cli.main(training_args(dataset, output)) == 0

    root_manifest = output / MANIFEST_FILENAME
    checkpoint_manifest = checkpoint / MANIFEST_FILENAME
    root_raw = json.loads(root_manifest.read_text(encoding="utf-8"))
    checkpoint_raw = json.loads(checkpoint_manifest.read_text(encoding="utf-8"))
    assert root_raw["action_mode"] == "joint"
    assert root_raw["checkpoint_files"] is None
    assert set(checkpoint_raw["checkpoint_files"]) == {
        "config.json",
        "model.safetensors",
        "policy_preprocessor.json",
        "policy_postprocessor.json",
    }
    assert checkpoint_manifest.read_bytes() != root_manifest.read_bytes()
    assert "[MANIFEST]" in capsys.readouterr().out


def test_failed_training_does_not_write_manifest(tmp_path, monkeypatch):
    dataset = tmp_path / "dataset"
    output = tmp_path / "output"
    write_dataset(dataset)
    monkeypatch.setattr(training_cli.shutil, "which", lambda executable: executable)
    monkeypatch.setattr(
        training_cli.subprocess,
        "run",
        lambda command, check: SimpleNamespace(returncode=7),
    )

    assert training_cli.main(training_args(dataset, output)) == 7
    assert not (output / MANIFEST_FILENAME).exists()


def test_successful_training_manifest_uses_custom_urdf_fingerprint(tmp_path, monkeypatch):
    dataset = tmp_path / "dataset"
    output = tmp_path / "output"
    checkpoint = output / "checkpoints" / "last" / "pretrained_model"
    custom_urdf = tmp_path / "custom.urdf"
    custom_urdf.write_text("<robot name='custom'/>", encoding="utf-8")
    write_dataset(dataset)

    def fake_run(command, check):
        assert check is False
        assert str(custom_urdf.resolve()) not in command
        assert not any(argument.startswith(("--urdf", "--dataset.urdf")) for argument in command)
        write_checkpoint(checkpoint)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(training_cli.shutil, "which", lambda executable: executable)
    monkeypatch.setattr(training_cli.subprocess, "run", fake_run)
    args = [*training_args(dataset, output), "--urdf-path", str(custom_urdf)]

    assert training_cli.main(args) == 0

    raw_text = (output / MANIFEST_FILENAME).read_text(encoding="utf-8")
    root_manifest = json.loads(raw_text)
    checkpoint_manifest = json.loads((checkpoint / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    expected = hashlib.sha256(custom_urdf.read_bytes()).hexdigest()
    assert root_manifest["kinematics"]["urdf_sha256"] == expected
    assert checkpoint_manifest["kinematics"]["urdf_sha256"] == expected
    assert checkpoint_manifest["checkpoint_files"] is not None
    assert str(custom_urdf.resolve()) not in raw_text


def test_sync_manifest_does_not_train_and_refreshes_checkpoint_binding(tmp_path, monkeypatch, capsys):
    dataset = tmp_path / "dataset"
    output = tmp_path / "output"
    checkpoint = output / "checkpoints" / "last" / "pretrained_model"
    write_dataset(dataset)
    write_checkpoint(checkpoint, marker="before")

    def fail_run(*args, **kwargs):
        pytest.fail("--sync-manifest must not start lerobot-train")

    monkeypatch.setattr(training_cli.subprocess, "run", fail_run)
    args = [*training_args(dataset, output, run=False), "--sync-manifest"]

    assert training_cli.main(args) == 0
    target = checkpoint / MANIFEST_FILENAME
    before = json.loads(target.read_text(encoding="utf-8"))["checkpoint_files"]["model.safetensors"]

    (checkpoint / "model.safetensors").write_bytes(b"model:after")
    assert training_cli.main(args) == 0

    after = json.loads(target.read_text(encoding="utf-8"))["checkpoint_files"]["model.safetensors"]
    assert before != after
    assert after == hashlib.sha256(b"model:after").hexdigest()
    assert json.loads((output / MANIFEST_FILENAME).read_text(encoding="utf-8"))["checkpoint_files"] is None
    assert "[SYNC]" in capsys.readouterr().out


def test_authorized_hub_upload_selects_last_bound_checkpoint(tmp_path, monkeypatch, capsys):
    dataset = tmp_path / "dataset"
    output = tmp_path / "output"
    numbered = output / "checkpoints" / "000100" / "pretrained_model"
    last = output / "checkpoints" / "last" / "pretrained_model"
    write_dataset(dataset)

    def fake_run(command, check):
        assert check is False
        write_checkpoint(numbered, marker="numbered")
        write_checkpoint(last, marker="last")
        return SimpleNamespace(returncode=0)

    visibility_checks = []
    uploads = []

    def fake_visibility(repo_id, *, public):
        visibility_checks.append((repo_id, public))

    def fake_upload(path, repo_id):
        uploads.append((path, repo_id))

    monkeypatch.setattr(training_cli.shutil, "which", lambda executable: executable)
    monkeypatch.setattr(training_cli.subprocess, "run", fake_run)
    monkeypatch.setattr(training_cli, "_validate_hub_visibility", fake_visibility)
    monkeypatch.setattr(training_cli, "_upload_bound_checkpoint_to_hub", fake_upload)
    args = [
        *training_args(dataset, output),
        "--push-to-hub",
        "--policy-repo-id",
        "FAFU-Robotics/act-demo",
    ]

    assert training_cli.main(args) == 0

    assert visibility_checks == [("FAFU-Robotics/act-demo", False)]
    assert uploads == [((last / MANIFEST_FILENAME).resolve(), "FAFU-Robotics/act-demo")]
    assert "[HUB] uploaded" in capsys.readouterr().out


def test_bound_checkpoint_upload_failure_keeps_local_output(tmp_path, monkeypatch, capsys):
    dataset = tmp_path / "dataset"
    output = tmp_path / "output"
    checkpoint = output / "checkpoints" / "last" / "pretrained_model"
    write_dataset(dataset)

    def fake_run(command, check):
        assert check is False
        write_checkpoint(checkpoint)
        return SimpleNamespace(returncode=0)

    def fail_upload(path, repo_id):
        raise training_cli.InferenceManifestError("checkpoint upload failed")

    monkeypatch.setattr(training_cli.shutil, "which", lambda executable: executable)
    monkeypatch.setattr(training_cli.subprocess, "run", fake_run)
    monkeypatch.setattr(training_cli, "_validate_hub_visibility", lambda repo_id, public: None)
    monkeypatch.setattr(training_cli, "_upload_bound_checkpoint_to_hub", fail_upload)
    args = [
        *training_args(dataset, output),
        "--push-to-hub",
        "--policy-repo-id",
        "FAFU-Robotics/act-demo",
    ]

    assert training_cli.main(args) == 2
    assert (output / MANIFEST_FILENAME).is_file()
    assert (checkpoint / MANIFEST_FILENAME).is_file()
    assert "checkpoint upload failed" in capsys.readouterr().err


def test_upload_bound_checkpoint_creates_one_atomic_commit_with_bound_files(tmp_path, monkeypatch):
    dataset = tmp_path / "dataset"
    output = tmp_path / "output"
    checkpoint = output / "checkpoints" / "last" / "pretrained_model"
    write_dataset(dataset)
    write_checkpoint(checkpoint)
    manifest = training_cli.build_manifest_from_dataset(dataset, "joint")
    training_cli.write_training_manifests(output, manifest=manifest)
    manifest_path = checkpoint / MANIFEST_FILENAME
    manifest_raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    commits = []

    class FakeApi:
        def create_commit(self, **kwargs):
            commits.append(kwargs)

    monkeypatch.setattr(training_cli, "HfApi", FakeApi)

    training_cli._upload_bound_checkpoint_to_hub(manifest_path, "FAFU-Robotics/act-demo")

    assert len(commits) == 1
    commit = commits[0]
    assert commit["repo_id"] == "FAFU-Robotics/act-demo"
    assert commit["repo_type"] == "model"
    assert commit["commit_message"] == "Upload bound FAFU ACT checkpoint"
    operations = commit["operations"]
    assert {operation.path_in_repo for operation in operations} == {
        *manifest_raw["checkpoint_files"],
        MANIFEST_FILENAME,
    }
    assert all(Path(operation.path_or_fileobj).parent == checkpoint for operation in operations)


@pytest.mark.parametrize(("public", "existing_private"), [(False, True), (True, False)])
def test_existing_hub_visibility_matching_request_is_accepted(monkeypatch, public, existing_private):
    calls = []

    class FakeApi:
        def repo_info(self, *, repo_id, repo_type):
            calls.append((repo_id, repo_type))
            return SimpleNamespace(private=existing_private)

    monkeypatch.setattr(training_cli, "HfApi", FakeApi)

    training_cli._validate_hub_visibility("FAFU-Robotics/act-demo", public=public)

    assert calls == [("FAFU-Robotics/act-demo", "model")]


@pytest.mark.parametrize(("public", "existing_private"), [(False, False), (True, True)])
def test_existing_hub_visibility_mismatch_is_rejected(monkeypatch, public, existing_private):
    class FakeApi:
        def repo_info(self, *, repo_id, repo_type):
            return SimpleNamespace(private=existing_private)

    monkeypatch.setattr(training_cli, "HfApi", FakeApi)

    with pytest.raises(ValueError, match="visibility|requests"):
        training_cli._validate_hub_visibility("FAFU-Robotics/act-demo", public=public)
