from __future__ import annotations

import hashlib
import json

import pytest

from lerobot_robot_fafu_arm.inference.manifest import (
    MANIFEST_FILENAME,
    InferenceManifest,
    InferenceManifestError,
    build_manifest_from_dataset,
    load_inference_manifest,
    verify_checkpoint_integrity,
    write_inference_manifest,
    write_training_manifests,
)
from lerobot_robot_fafu_arm.kinematics import default_urdf_path
from lerobot_robot_fafu_arm.representation import action_features


def write_dataset_info(
    root,
    *,
    action_mode="joint",
    action_names=None,
    state_names=None,
    cameras=None,
    robot_type="fafu_follower",
    fps=30,
):
    action_names = action_names or list(action_features(action_mode))
    if state_names is None:
        state_names = [f"joint{i}.pos" for i in range(1, 7)] + ["gripper.pos"]
    if cameras is None:
        cameras = {"front": [3, 480, 640]}
    features = {
        "action": {
            "dtype": "float32",
            "shape": [len(action_names)],
            "names": action_names,
        }
    }
    if state_names:
        features["observation.state"] = {
            "dtype": "float32",
            "shape": [len(state_names)],
            "names": state_names,
        }
    for name, shape in cameras.items():
        features[f"observation.images.{name}"] = {
            "dtype": "video",
            "shape": shape,
            "names": ["channels", "height", "width"],
        }
    info = {
        "robot_type": robot_type,
        "fps": fps,
        "total_episodes": 1,
        "features": features,
    }
    meta = root / "meta"
    meta.mkdir(parents=True)
    (meta / "info.json").write_text(json.dumps(info), encoding="utf-8")
    return features


def write_checkpoint_files(root, *, marker="checkpoint"):
    root.mkdir(parents=True, exist_ok=True)
    preprocessor_state = "policy_preprocessor_step_0_normalizer.safetensors"
    postprocessor_state = "policy_postprocessor_step_0_unnormalizer.safetensors"
    (root / "config.json").write_text(
        json.dumps({"type": "act", "marker": marker}),
        encoding="utf-8",
    )
    (root / "model.safetensors").write_bytes(f"model:{marker}".encode())
    (root / "policy_preprocessor.json").write_text(
        json.dumps(
            {
                "name": "policy_preprocessor",
                "steps": [{"registry_name": "normalizer", "config": {}, "state_file": preprocessor_state}],
            }
        ),
        encoding="utf-8",
    )
    (root / "policy_postprocessor.json").write_text(
        json.dumps(
            {
                "name": "policy_postprocessor",
                "steps": [{"registry_name": "unnormalizer", "config": {}, "state_file": postprocessor_state}],
            }
        ),
        encoding="utf-8",
    )
    (root / preprocessor_state).write_bytes(f"preprocessor:{marker}".encode())
    (root / postprocessor_state).write_bytes(f"postprocessor:{marker}".encode())
    return {
        "config.json",
        "model.safetensors",
        "policy_preprocessor.json",
        "policy_postprocessor.json",
        preprocessor_state,
        postprocessor_state,
    }


@pytest.mark.parametrize("action_mode", ["joint", "ee_delta", "ee_pose"])
def test_build_write_and_load_manifest_preserves_schema_without_local_path(tmp_path, action_mode):
    dataset = tmp_path / "private" / "dataset"
    features = write_dataset_info(dataset, action_mode=action_mode)

    manifest = build_manifest_from_dataset(dataset, action_mode)
    target = write_inference_manifest(manifest, tmp_path / "checkpoint")
    loaded = load_inference_manifest(tmp_path / "checkpoint")
    raw_text = target.read_text(encoding="utf-8")

    assert target.name == MANIFEST_FILENAME
    assert loaded.action_mode == action_mode
    assert loaded.robot_type == "fafu_follower"
    assert loaded.fps == 30.0
    assert loaded.features == features
    assert loaded.action_names == tuple(action_features(action_mode))
    assert loaded.state_names[-1] == "gripper.pos"
    assert tuple(loaded.camera_features) == ("observation.images.front",)
    assert loaded.kinematics == {
        "urdf_sha256": hashlib.sha256(default_urdf_path().read_bytes()).hexdigest(),
        "base_link": "base_link",
        "tool_link": "tool_link",
    }
    assert str(dataset.resolve()) not in raw_text


def test_manifest_rejects_missing_observation_state_required_by_act(tmp_path):
    dataset = tmp_path / "dataset"
    write_dataset_info(dataset, state_names=[])

    with pytest.raises(InferenceManifestError, match="observation.state is required"):
        build_manifest_from_dataset(dataset, "joint")


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ("action_order", "action names/order"),
        ("action_shape", "action.shape"),
        ("state_order", "observation.state names/order"),
        ("state_shape", "observation.state.shape"),
        ("camera_shape", "camera features must use the same shape"),
        ("camera_layout", "3 RGB channels"),
    ],
)
def test_manifest_rejects_inconsistent_action_state_and_camera_schema(tmp_path, change, message):
    dataset = tmp_path / change
    features = write_dataset_info(dataset, cameras={"front": [3, 480, 640], "wrist": [3, 480, 640]})
    if change == "action_order":
        features["action"]["names"][0], features["action"]["names"][1] = (
            features["action"]["names"][1],
            features["action"]["names"][0],
        )
    elif change == "action_shape":
        features["action"]["shape"] = [8]
    elif change == "state_order":
        features["observation.state"]["names"].reverse()
    elif change == "state_shape":
        features["observation.state"]["shape"] = [8]
    elif change == "camera_shape":
        features["observation.images.wrist"]["shape"] = [3, 240, 320]
    elif change == "camera_layout":
        features["observation.images.front"]["shape"] = [4, 480, 640]
    (dataset / "meta" / "info.json").write_text(
        json.dumps(
            {
                "robot_type": "fafu_follower",
                "fps": 30,
                "total_episodes": 1,
                "features": features,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(InferenceManifestError, match=message):
        build_manifest_from_dataset(dataset, "joint")


def test_manifest_rejects_single_channel_camera(tmp_path):
    dataset = tmp_path / "single-channel"
    write_dataset_info(dataset, cameras={"front": [1, 480, 640]})
    with pytest.raises(InferenceManifestError, match="3 RGB channels"):
        build_manifest_from_dataset(dataset, "joint")


def test_manifest_requires_a_camera_and_rejects_absolute_paths(tmp_path):
    no_camera = tmp_path / "no_camera"
    write_dataset_info(no_camera, cameras={})
    with pytest.raises(InferenceManifestError, match="at least one"):
        build_manifest_from_dataset(no_camera, "joint")

    dataset = tmp_path / "absolute"
    features = write_dataset_info(dataset)
    features["observation.images.front"]["info"] = {"cache": str(tmp_path.resolve())}
    raw = json.loads((dataset / "meta" / "info.json").read_text(encoding="utf-8"))
    raw["features"] = features
    (dataset / "meta" / "info.json").write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(InferenceManifestError, match="absolute path"):
        build_manifest_from_dataset(dataset, "joint")


def test_legacy_checkpoint_fallback_is_explicit_and_does_not_guess_action_mode(tmp_path):
    dataset = tmp_path / "dataset"
    checkpoint = tmp_path / "old_checkpoint"
    checkpoint.mkdir()
    write_dataset_info(dataset, action_mode="ee_delta")

    with pytest.raises(InferenceManifestError, match="require explicit"):
        load_inference_manifest(checkpoint)
    with pytest.raises(InferenceManifestError, match="require explicit"):
        load_inference_manifest(checkpoint, dataset_root=dataset)

    manifest = load_inference_manifest(
        checkpoint,
        dataset_root=dataset,
        action_mode="ee_delta",
        urdf_path=default_urdf_path(),
    )

    assert manifest.action_mode == "ee_delta"
    assert manifest.action_names == tuple(action_features("ee_delta"))
    assert not (checkpoint / MANIFEST_FILENAME).exists()
    assert verify_checkpoint_integrity(checkpoint, manifest, allow_legacy=True) is False
    with pytest.raises(InferenceManifestError, match="allow_legacy=True"):
        verify_checkpoint_integrity(checkpoint, manifest)


def test_manifest_load_can_cross_check_dataset_and_reject_action_mode_mismatch(tmp_path):
    dataset = tmp_path / "dataset"
    checkpoint = tmp_path / "checkpoint"
    write_dataset_info(dataset)
    write_inference_manifest(build_manifest_from_dataset(dataset, "joint"), checkpoint)

    with pytest.raises(InferenceManifestError, match="requested action_mode"):
        load_inference_manifest(checkpoint, action_mode="ee_pose")

    raw = json.loads((dataset / "meta" / "info.json").read_text(encoding="utf-8"))
    raw["fps"] = 20
    (dataset / "meta" / "info.json").write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(InferenceManifestError, match="metadata differ"):
        load_inference_manifest(checkpoint, dataset_root=dataset)


def test_write_training_manifests_covers_output_and_every_checkpoint(tmp_path):
    dataset = tmp_path / "dataset"
    output = tmp_path / "output"
    first = output / "checkpoints" / "000100" / "pretrained_model"
    last = output / "checkpoints" / "last" / "pretrained_model"
    first_files = write_checkpoint_files(first, marker="first")
    last_files = write_checkpoint_files(last, marker="last")
    write_dataset_info(dataset)
    manifest = build_manifest_from_dataset(dataset, "joint")

    paths = write_training_manifests(output, manifest=manifest)

    assert paths == (
        (output / MANIFEST_FILENAME).resolve(),
        (first / MANIFEST_FILENAME).resolve(),
        (last / MANIFEST_FILENAME).resolve(),
    )
    assert all(path.is_file() for path in paths)
    root_manifest = json.loads(paths[0].read_text(encoding="utf-8"))
    first_manifest = load_inference_manifest(first)
    last_manifest = load_inference_manifest(last)
    assert root_manifest["checkpoint_files"] is None
    assert set(first_manifest.checkpoint_files or {}) == first_files
    assert set(last_manifest.checkpoint_files or {}) == last_files
    assert first_manifest.checkpoint_files != last_manifest.checkpoint_files
    assert verify_checkpoint_integrity(first, first_manifest)
    assert verify_checkpoint_integrity(last, last_manifest)


def test_write_manifest_is_idempotent_but_refuses_inconsistent_existing_schema(tmp_path):
    joint_dataset = tmp_path / "joint"
    delta_dataset = tmp_path / "delta"
    checkpoint = tmp_path / "checkpoint"
    write_dataset_info(joint_dataset, action_mode="joint")
    write_dataset_info(delta_dataset, action_mode="ee_delta")
    joint = build_manifest_from_dataset(joint_dataset, "joint")
    delta = build_manifest_from_dataset(delta_dataset, "ee_delta")

    first = write_inference_manifest(joint, checkpoint)
    second = write_inference_manifest(joint, checkpoint)

    assert first == second
    with pytest.raises(InferenceManifestError, match="refusing to overwrite"):
        write_inference_manifest(delta, checkpoint)


def test_manifest_parser_rejects_unknown_version_and_fields(tmp_path):
    dataset = tmp_path / "dataset"
    features = write_dataset_info(dataset)
    base = {
        "schema_version": 1,
        "action_mode": "joint",
        "robot_type": "fafu_follower",
        "fps": 30,
        "features": features,
        "kinematics": build_manifest_from_dataset(dataset, "joint").kinematics,
        "checkpoint_files": None,
    }

    with pytest.raises(InferenceManifestError, match="schema_version"):
        InferenceManifest.from_dict({**base, "schema_version": 2})
    with pytest.raises(InferenceManifestError, match="unknown"):
        InferenceManifest.from_dict({**base, "dataset_root": str(dataset)})
    with pytest.raises(InferenceManifestError, match="missing: kinematics"):
        InferenceManifest.from_dict({key: value for key, value in base.items() if key != "kinematics"})


def test_default_urdf_fingerprint_is_stable_and_contains_no_path(tmp_path):
    dataset = tmp_path / "dataset"
    write_dataset_info(dataset)

    manifest = build_manifest_from_dataset(dataset, "joint")
    serialized = json.dumps(manifest.to_dict())

    assert manifest.kinematics["urdf_sha256"] == hashlib.sha256(default_urdf_path().read_bytes()).hexdigest()
    assert str(default_urdf_path().resolve()) not in serialized


def test_custom_urdf_fingerprint_round_trip_requires_same_selected_urdf(tmp_path):
    dataset = tmp_path / "dataset"
    checkpoint = tmp_path / "checkpoint"
    custom_urdf = tmp_path / "custom.urdf"
    different_urdf = tmp_path / "different.urdf"
    custom_urdf.write_text("<robot name='custom'/>", encoding="utf-8")
    different_urdf.write_text("<robot name='different'/>", encoding="utf-8")
    write_dataset_info(dataset)
    manifest = build_manifest_from_dataset(dataset, "joint", custom_urdf)
    write_inference_manifest(manifest, checkpoint)

    loaded = load_inference_manifest(checkpoint, urdf_path=custom_urdf)

    assert loaded == manifest
    assert loaded.kinematics["urdf_sha256"] == hashlib.sha256(custom_urdf.read_bytes()).hexdigest()
    assert str(custom_urdf.resolve()) not in json.dumps(loaded.to_dict())
    with pytest.raises(InferenceManifestError, match="URDF fingerprint"):
        load_inference_manifest(checkpoint, urdf_path=different_urdf)
    with pytest.raises(InferenceManifestError, match="URDF fingerprint"):
        load_inference_manifest(checkpoint)


def test_legacy_fallback_accepts_custom_urdf_path(tmp_path):
    dataset = tmp_path / "dataset"
    checkpoint = tmp_path / "old_checkpoint"
    custom_urdf = tmp_path / "legacy.urdf"
    checkpoint.mkdir()
    custom_urdf.write_bytes(b"legacy-kinematics")
    write_dataset_info(dataset)

    manifest = load_inference_manifest(
        checkpoint,
        dataset_root=dataset,
        action_mode="joint",
        urdf_path=custom_urdf,
    )

    assert manifest.kinematics["urdf_sha256"] == hashlib.sha256(custom_urdf.read_bytes()).hexdigest()


def test_manifest_load_detects_valid_but_tampered_urdf_fingerprint(tmp_path):
    dataset = tmp_path / "dataset"
    checkpoint = tmp_path / "checkpoint"
    write_dataset_info(dataset)
    target = write_inference_manifest(build_manifest_from_dataset(dataset, "joint"), checkpoint)
    raw = json.loads(target.read_text(encoding="utf-8"))
    raw["kinematics"]["urdf_sha256"] = "0" * 64
    target.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(InferenceManifestError, match="URDF fingerprint"):
        load_inference_manifest(checkpoint)


@pytest.mark.parametrize(
    "kinematics",
    [
        {"urdf_sha256": "ABC", "base_link": "base_link", "tool_link": "tool_link"},
        {"urdf_sha256": "0" * 64, "base_link": "C:\\robot", "tool_link": "tool_link"},
        {"urdf_sha256": "0" * 64, "base_link": "base_link", "tool_link": "base_link"},
        {
            "urdf_sha256": "0" * 64,
            "base_link": "base_link",
            "tool_link": "tool_link",
            "path": "relative.urdf",
        },
    ],
)
def test_manifest_parser_rejects_malformed_or_nonportable_kinematics(tmp_path, kinematics):
    dataset = tmp_path / "dataset"
    write_dataset_info(dataset)
    raw = build_manifest_from_dataset(dataset, "joint").to_dict()

    with pytest.raises(InferenceManifestError, match="kinematics"):
        InferenceManifest.from_dict({**raw, "kinematics": kinematics})


def bind_checkpoint(dataset, checkpoint, *, marker):
    files = write_checkpoint_files(checkpoint, marker=marker)
    target = write_inference_manifest(build_manifest_from_dataset(dataset, "joint"), checkpoint)
    return target, files


def test_pretrained_manifest_binds_all_inference_and_processor_state_files(tmp_path):
    dataset = tmp_path / "dataset"
    checkpoint = tmp_path / "run" / "pretrained_model"
    write_dataset_info(dataset)

    target, expected_files = bind_checkpoint(dataset, checkpoint, marker="bound")
    raw = json.loads(target.read_text(encoding="utf-8"))
    loaded = load_inference_manifest(checkpoint)

    assert set(raw["checkpoint_files"]) == expected_files
    assert loaded.checkpoint_files == raw["checkpoint_files"]
    for name, digest in raw["checkpoint_files"].items():
        assert digest == hashlib.sha256((checkpoint / name).read_bytes()).hexdigest()
    assert verify_checkpoint_integrity(checkpoint, loaded) is True
    semantic_manifest = build_manifest_from_dataset(dataset, "joint")
    assert semantic_manifest.checkpoint_files is None
    assert verify_checkpoint_integrity(checkpoint, semantic_manifest) is True


def test_checkpoint_rejects_manifest_replaced_from_another_checkpoint(tmp_path):
    dataset = tmp_path / "dataset"
    first = tmp_path / "first" / "pretrained_model"
    second = tmp_path / "second" / "pretrained_model"
    write_dataset_info(dataset)
    first_manifest, _ = bind_checkpoint(dataset, first, marker="first")
    second_manifest, _ = bind_checkpoint(dataset, second, marker="second")
    second_manifest.write_bytes(first_manifest.read_bytes())

    with pytest.raises(InferenceManifestError, match="SHA-256 mismatch"):
        load_inference_manifest(second)


def test_checkpoint_rejects_unbound_root_manifest_substitution(tmp_path):
    dataset = tmp_path / "dataset"
    output = tmp_path / "output"
    checkpoint = tmp_path / "run" / "pretrained_model"
    output.mkdir()
    write_dataset_info(dataset)
    manifest = build_manifest_from_dataset(dataset, "joint")
    root_manifest = write_inference_manifest(manifest, output)
    bound_manifest, _ = bind_checkpoint(dataset, checkpoint, marker="bound")
    bound_manifest.write_bytes(root_manifest.read_bytes())

    with pytest.raises(InferenceManifestError, match="no checkpoint_files binding"):
        load_inference_manifest(checkpoint)


@pytest.mark.parametrize(
    "filename",
    [
        "model.safetensors",
        "policy_preprocessor.json",
        "policy_preprocessor_step_0_normalizer.safetensors",
    ],
)
def test_checkpoint_rejects_modified_weight_or_processor_file(tmp_path, filename):
    dataset = tmp_path / "dataset"
    checkpoint = tmp_path / "run" / "pretrained_model"
    write_dataset_info(dataset)
    bind_checkpoint(dataset, checkpoint, marker="original")
    path = checkpoint / filename
    path.write_bytes(path.read_bytes() + b"\n")

    with pytest.raises(InferenceManifestError, match="SHA-256 mismatch"):
        load_inference_manifest(checkpoint)


@pytest.mark.parametrize(
    "filename",
    [
        "config.json",
        "policy_preprocessor_step_0_normalizer.safetensors",
    ],
)
def test_checkpoint_rejects_missing_required_or_processor_state_file(tmp_path, filename):
    dataset = tmp_path / "dataset"
    checkpoint = tmp_path / "run" / "pretrained_model"
    write_dataset_info(dataset)
    bind_checkpoint(dataset, checkpoint, marker="original")
    (checkpoint / filename).unlink()

    with pytest.raises(InferenceManifestError, match="missing"):
        load_inference_manifest(checkpoint)


def test_output_root_manifest_is_an_explicit_unbound_template(tmp_path):
    dataset = tmp_path / "dataset"
    output = tmp_path / "output"
    output.mkdir()
    write_dataset_info(dataset)
    target = write_inference_manifest(build_manifest_from_dataset(dataset, "joint"), output)

    raw = json.loads(target.read_text(encoding="utf-8"))
    loaded = load_inference_manifest(output)

    assert raw["checkpoint_files"] is None
    assert verify_checkpoint_integrity(output, loaded, allow_template=True) is False
    with pytest.raises(InferenceManifestError, match="output-root template"):
        verify_checkpoint_integrity(output, loaded)


@pytest.mark.parametrize(
    "checkpoint_files",
    [
        {},
        {
            "config.json": "0" * 64,
            "model.safetensors": "0" * 64,
            "policy_preprocessor.json": "0" * 64,
        },
        {
            "config.json": "0" * 64,
            "model.safetensors": "0" * 64,
            "policy_preprocessor.json": "0" * 64,
            "policy_postprocessor.json": "0" * 64,
            "../processor.safetensors": "0" * 64,
        },
    ],
)
def test_manifest_parser_rejects_incomplete_or_nonportable_checkpoint_binding(tmp_path, checkpoint_files):
    dataset = tmp_path / "dataset"
    write_dataset_info(dataset)
    raw = build_manifest_from_dataset(dataset, "joint").to_dict()

    with pytest.raises(InferenceManifestError, match="checkpoint_files"):
        InferenceManifest.from_dict({**raw, "checkpoint_files": checkpoint_files})
