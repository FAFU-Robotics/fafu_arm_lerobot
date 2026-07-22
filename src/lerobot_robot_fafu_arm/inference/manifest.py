"""Versioned dataset schema manifests used by FAFU policy inference."""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass, field, replace
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from ..kinematics import default_urdf_path
from ..local_dataset import DatasetReadError, load_dataset_info
from ..representation import EE_COMPONENTS, JOINT_NAMES, action_features

MANIFEST_FILENAME = "fafu_inference_manifest.json"
MANIFEST_SCHEMA_VERSION = 1
_INFERENCE_ACTION_MODES = frozenset({"joint", "ee_delta", "ee_pose"})
_CAMERA_DTYPES = frozenset({"image", "video"})
_DEFAULT_BASE_LINK = "base_link"
_DEFAULT_TOOL_LINK = "tool_link"
_REQUIRED_CHECKPOINT_FILES = (
    "config.json",
    "model.safetensors",
    "policy_preprocessor.json",
    "policy_postprocessor.json",
)
_PROCESSOR_CONFIG_FILES = _REQUIRED_CHECKPOINT_FILES[2:]


class InferenceManifestError(ValueError):
    """Raised when an inference manifest is missing, unsafe, or inconsistent."""


@dataclass(frozen=True)
class InferenceManifest:
    """Portable policy schema copied from a LeRobot dataset's metadata."""

    action_mode: str
    robot_type: str
    fps: float
    features: dict[str, dict[str, Any]]
    kinematics: dict[str, str] = field(
        default_factory=lambda: build_kinematics_identity(None, _DEFAULT_BASE_LINK, _DEFAULT_TOOL_LINK)
    )
    checkpoint_files: dict[str, str] | None = None
    schema_version: int = MANIFEST_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if (
            not isinstance(self.schema_version, int)
            or isinstance(self.schema_version, bool)
            or self.schema_version != MANIFEST_SCHEMA_VERSION
        ):
            raise InferenceManifestError(
                f"unsupported inference manifest schema_version {self.schema_version!r}; "
                f"expected {MANIFEST_SCHEMA_VERSION}"
            )
        if not isinstance(self.action_mode, str) or self.action_mode not in _INFERENCE_ACTION_MODES:
            raise InferenceManifestError("action_mode must be joint, ee_delta, or ee_pose")
        if not isinstance(self.robot_type, str) or not self.robot_type.strip():
            raise InferenceManifestError("robot_type must be a non-empty string")
        _reject_absolute_paths(self.robot_type, "robot_type")
        if (
            not isinstance(self.fps, (int, float))
            or isinstance(self.fps, bool)
            or not math.isfinite(float(self.fps))
            or float(self.fps) <= 0
        ):
            raise InferenceManifestError("fps must be a finite positive number")
        kinematics = _validate_kinematics_identity(self.kinematics)
        checkpoint_files = _validate_checkpoint_files(self.checkpoint_files)

        normalized = _json_copy(self.features, "features")
        if not isinstance(normalized, dict) or not normalized:
            raise InferenceManifestError("features must be a non-empty JSON object")
        if not all(isinstance(key, str) and isinstance(value, dict) for key, value in normalized.items()):
            raise InferenceManifestError("every feature must have a string key and an object value")
        _reject_absolute_paths(normalized, "features")
        _validate_action_feature(normalized, self.action_mode)
        _validate_state_feature(normalized)
        _validate_camera_features(normalized)

        object.__setattr__(self, "robot_type", self.robot_type.strip())
        object.__setattr__(self, "fps", float(self.fps))
        object.__setattr__(self, "features", normalized)
        object.__setattr__(self, "kinematics", kinematics)
        object.__setattr__(self, "checkpoint_files", checkpoint_files)

    @property
    def action_names(self) -> tuple[str, ...]:
        """Return the exact semantic order used to decode an action tensor."""

        return tuple(self.features["action"]["names"])

    @property
    def state_names(self) -> tuple[str, ...]:
        """Return the exact proprioceptive order required by ACT inference."""

        return tuple(self.features["observation.state"]["names"])

    @property
    def camera_features(self) -> dict[str, dict[str, Any]]:
        """Return camera feature metadata in its recorded order."""

        return {
            key: feature for key, feature in self.features.items() if key.startswith("observation.images.")
        }

    def to_dict(self) -> dict[str, Any]:
        """Return the stable JSON representation without local filesystem paths."""

        return {
            "schema_version": self.schema_version,
            "action_mode": self.action_mode,
            "robot_type": self.robot_type,
            "fps": self.fps,
            "features": _json_copy(self.features, "features"),
            "kinematics": dict(self.kinematics),
            "checkpoint_files": dict(self.checkpoint_files) if self.checkpoint_files is not None else None,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> InferenceManifest:
        """Parse and strictly validate a manifest JSON object."""

        if not isinstance(value, dict):
            raise InferenceManifestError("inference manifest must be a JSON object")
        required = {
            "schema_version",
            "action_mode",
            "robot_type",
            "fps",
            "features",
            "kinematics",
            "checkpoint_files",
        }
        missing = sorted(required - set(value))
        unknown = sorted(set(value) - required)
        if missing:
            raise InferenceManifestError(f"inference manifest is missing: {', '.join(missing)}")
        if unknown:
            raise InferenceManifestError(f"unknown inference manifest fields: {', '.join(unknown)}")
        return cls(
            schema_version=value["schema_version"],
            action_mode=value["action_mode"],
            robot_type=value["robot_type"],
            fps=value["fps"],
            features=value["features"],
            kinematics=value["kinematics"],
            checkpoint_files=value["checkpoint_files"],
        )


def build_manifest_from_dataset(
    dataset_root: str | Path,
    action_mode: str,
    urdf_path: str | Path | None = None,
    *,
    base_link: str = _DEFAULT_BASE_LINK,
    tool_link: str = _DEFAULT_TOOL_LINK,
) -> InferenceManifest:
    """Build a portable dataset manifest bound to one exact kinematic chain."""

    try:
        info = load_dataset_info(dataset_root)
        fps = info.fps
    except DatasetReadError as exc:
        raise InferenceManifestError(str(exc)) from exc
    robot_type = info.raw.get("robot_type")
    return InferenceManifest(
        action_mode=action_mode,
        robot_type=robot_type,
        fps=fps,
        features=info.features,
        kinematics=build_kinematics_identity(urdf_path, base_link, tool_link),
    )


def load_inference_manifest(
    checkpoint_or_file: str | Path,
    *,
    dataset_root: str | Path | None = None,
    action_mode: str | None = None,
    urdf_path: str | Path | None = None,
    base_link: str = _DEFAULT_BASE_LINK,
    tool_link: str = _DEFAULT_TOOL_LINK,
) -> InferenceManifest:
    """Load a checkpoint manifest, with an explicit dataset fallback for old checkpoints.

    A legacy checkpoint may omit ``MANIFEST_FILENAME``. In that case both ``dataset_root``
    and ``action_mode`` are required; no representation is guessed from tensor dimensions.
    The manifest is always checked against the selected URDF and links. ``urdf_path=None``
    selects the packaged FAFU URDF; custom-URDF checkpoints must pass their URDF explicitly.
    """

    source = Path(checkpoint_or_file).expanduser()
    manifest_path = source / MANIFEST_FILENAME if source.is_dir() else source
    if manifest_path.is_file():
        manifest = _read_manifest_file(manifest_path)
        selected_kinematics = build_kinematics_identity(urdf_path, base_link, tool_link)
        _require_same_kinematics(manifest.kinematics, selected_kinematics)
        if action_mode is not None and action_mode != manifest.action_mode:
            raise InferenceManifestError(
                f"requested action_mode {action_mode!r} does not match manifest {manifest.action_mode!r}"
            )
        if dataset_root is not None:
            rebuilt = build_manifest_from_dataset(
                dataset_root,
                action_mode or manifest.action_mode,
                urdf_path,
                base_link=base_link,
                tool_link=tool_link,
            )
            _require_same_manifest(
                replace(manifest, checkpoint_files=None),
                rebuilt,
                "checkpoint manifest and dataset metadata differ",
            )
        verify_checkpoint_integrity(source, manifest, allow_template=True)
        return manifest

    if not source.exists():
        raise InferenceManifestError(f"checkpoint or manifest path does not exist: {source.resolve()}")
    if dataset_root is None or action_mode is None or urdf_path is None:
        raise InferenceManifestError(
            f"inference manifest not found: {manifest_path}; legacy checkpoints require explicit "
            "dataset_root, action_mode, and urdf_path"
        )
    return build_manifest_from_dataset(
        dataset_root,
        action_mode,
        urdf_path,
        base_link=base_link,
        tool_link=tool_link,
    )


def write_inference_manifest(
    manifest: InferenceManifest,
    directory: str | Path,
    *,
    refresh_checkpoint_binding: bool = False,
) -> Path:
    """Atomically write one manifest, optionally refreshing only its checkpoint binding."""

    target_dir = Path(directory).expanduser()
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / MANIFEST_FILENAME
    target_manifest = (
        replace(manifest, checkpoint_files=_checkpoint_file_hashes(target_dir))
        if target_dir.name == "pretrained_model"
        else manifest
    )
    payload = _manifest_json(target_manifest)

    if target.exists():
        existing = _read_manifest_file(target)
        if refresh_checkpoint_binding and target_manifest.checkpoint_files is not None:
            _require_same_manifest(
                replace(existing, checkpoint_files=None),
                replace(target_manifest, checkpoint_files=None),
                f"refusing to rebind an inconsistent manifest: {target}",
            )
            if existing == target_manifest:
                return target.resolve()
        else:
            _require_same_manifest(
                existing, target_manifest, f"refusing to overwrite inconsistent manifest: {target}"
            )
            return target.resolve()

    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{MANIFEST_FILENAME}.",
            suffix=".tmp",
            dir=target_dir,
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(target)
    except OSError as exc:
        if temporary is not None:
            with contextlib.suppress(OSError):
                temporary.unlink(missing_ok=True)
        raise InferenceManifestError(f"could not write inference manifest {target}: {exc}") from exc
    return target.resolve()


def write_training_manifests(
    output_dir: str | Path,
    *,
    manifest: InferenceManifest | None = None,
    dataset_root: str | Path | None = None,
    action_mode: str | None = None,
    refresh_checkpoint_bindings: bool = False,
) -> tuple[Path, ...]:
    """Write the output manifest and copy it into every completed checkpoint."""

    output = Path(output_dir).expanduser()
    if not output.is_dir():
        raise InferenceManifestError(f"training output directory does not exist: {output.resolve()}")
    if manifest is None:
        if dataset_root is None or action_mode is None:
            raise InferenceManifestError("dataset_root and action_mode are required when manifest is omitted")
        manifest = build_manifest_from_dataset(dataset_root, action_mode)
    elif dataset_root is not None or action_mode is not None:
        raise InferenceManifestError("pass manifest or dataset_root/action_mode, not both")

    destinations = [output]
    checkpoint_root = output / "checkpoints"
    if checkpoint_root.is_dir():
        destinations.extend(
            path for path in sorted(checkpoint_root.glob("*/pretrained_model")) if path.is_dir()
        )
    paths = [write_inference_manifest(replace(manifest, checkpoint_files=None), output)]
    paths.extend(
        write_inference_manifest(
            manifest,
            destination,
            refresh_checkpoint_binding=refresh_checkpoint_bindings,
        )
        for destination in destinations[1:]
    )
    return tuple(paths)


def _validate_action_feature(features: dict[str, dict[str, Any]], action_mode: str) -> None:
    action = features.get("action")
    if action is None:
        raise InferenceManifestError("features.action is required")
    expected = tuple(action_features(action_mode))
    names = _validate_vector_feature(action, "action")
    if names != expected:
        raise InferenceManifestError(
            f"action names/order do not match {action_mode}: expected {list(expected)}, got {list(names)}"
        )


def _validate_state_feature(features: dict[str, dict[str, Any]]) -> None:
    state = features.get("observation.state")
    if state is None:
        raise InferenceManifestError("features.observation.state is required for ACT inference")
    names = _validate_vector_feature(state, "observation.state")
    if names not in _valid_state_name_orders():
        raise InferenceManifestError(
            "observation.state names/order do not match a supported FAFU joint, EE pose, or all schema"
        )


def _validate_vector_feature(feature: dict[str, Any], key: str) -> tuple[str, ...]:
    if feature.get("dtype") != "float32":
        raise InferenceManifestError(f"{key}.dtype must be 'float32'")
    names = feature.get("names")
    if not isinstance(names, list) or not names or not all(isinstance(name, str) and name for name in names):
        raise InferenceManifestError(f"{key}.names must be a non-empty string list")
    if len(set(names)) != len(names):
        raise InferenceManifestError(f"{key}.names contains duplicates")
    shape = feature.get("shape")
    if (
        not isinstance(shape, (list, tuple))
        or len(shape) != 1
        or isinstance(shape[0], bool)
        or not isinstance(shape[0], int)
        or shape[0] != len(names)
    ):
        raise InferenceManifestError(f"{key}.shape must be [{len(names)}], got {shape!r}")
    return tuple(names)


def _validate_camera_features(features: dict[str, dict[str, Any]]) -> None:
    cameras: list[tuple[str, tuple[int, int, int]]] = []
    for key, feature in features.items():
        dtype = feature.get("dtype")
        is_camera_key = key.startswith("observation.images.")
        if dtype in _CAMERA_DTYPES and not is_camera_key:
            raise InferenceManifestError(f"camera feature {key!r} must start with 'observation.images.'")
        if not is_camera_key:
            continue
        camera_name = key.removeprefix("observation.images.")
        if not camera_name:
            raise InferenceManifestError(f"invalid camera feature key: {key!r}")
        if dtype not in _CAMERA_DTYPES:
            raise InferenceManifestError(f"camera feature {key!r} must use image or video dtype")
        shape = feature.get("shape")
        if (
            not isinstance(shape, (list, tuple))
            or len(shape) != 3
            or not all(
                isinstance(value, int) and not isinstance(value, bool) and value > 0 for value in shape
            )
        ):
            raise InferenceManifestError(f"camera feature {key!r} must have a positive 3-D shape")
        shape_tuple = tuple(shape)
        _validate_camera_layout(key, shape_tuple, feature.get("names"))
        cameras.append((key, shape_tuple))

    if not cameras:
        raise InferenceManifestError("at least one observation.images.* camera feature is required")
    first_shape = cameras[0][1]
    if any(shape != first_shape for _, shape in cameras[1:]):
        details = ", ".join(f"{key}={list(shape)}" for key, shape in cameras)
        raise InferenceManifestError(f"camera features must use the same shape; found {details}")


def _validate_camera_layout(key: str, shape: tuple[int, int, int], names: Any) -> None:
    if names is not None:
        if names == ["height", "width", "channels"]:
            channel_index = 2
        elif names == ["channels", "height", "width"]:
            channel_index = 0
        else:
            raise InferenceManifestError(f"camera feature {key!r}.names must describe HWC or CHW dimensions")
        if shape[channel_index] != 3:
            raise InferenceManifestError(f"camera feature {key!r} must have 3 RGB channels")
        return

    possible_channels = [index for index in (0, 2) if shape[index] == 3]
    if len(possible_channels) != 1:
        raise InferenceManifestError(
            f"camera feature {key!r} must have one unambiguous 3-channel RGB HWC/CHW layout; add names"
        )


def _valid_state_name_orders() -> frozenset[tuple[str, ...]]:
    joint_position = tuple(f"{name}.pos" for name in JOINT_NAMES) + ("gripper.pos",)
    joint_velocity = tuple(f"{name}.vel" for name in JOINT_NAMES) + ("gripper.vel",)
    joint_effort = tuple(f"{name}.effort" for name in JOINT_NAMES) + ("gripper.effort",)
    ee_pose = tuple(f"ee.{name}" for name in EE_COMPONENTS) + ("gripper.pos",)
    ee_pose_without_duplicate_gripper = tuple(f"ee.{name}" for name in EE_COMPONENTS)
    ee_delta = tuple(f"ee_delta.{name}" for name in EE_COMPONENTS)

    schemas = {ee_pose}
    for include_velocity in (False, True):
        for include_effort in (False, True):
            optional = (joint_velocity if include_velocity else ()) + (joint_effort if include_effort else ())
            schemas.add(joint_position + optional)
            schemas.add(joint_position + optional + ee_pose_without_duplicate_gripper + ee_delta)
    return frozenset(schemas)


def _json_copy(value: Any, name: str) -> Any:
    try:
        return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise InferenceManifestError(f"{name} must contain only finite JSON values: {exc}") from exc


def build_kinematics_identity(
    urdf_path: str | Path | None,
    base_link: str,
    tool_link: str,
) -> dict[str, str]:
    path = Path(urdf_path).expanduser() if urdf_path is not None else default_urdf_path()
    if not path.is_file():
        raise InferenceManifestError(f"URDF does not exist or is not a file: {path.resolve()}")
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise InferenceManifestError(f"could not fingerprint URDF {path}: {exc}") from exc
    return _validate_kinematics_identity(
        {
            "urdf_sha256": digest.hexdigest(),
            "base_link": base_link,
            "tool_link": tool_link,
        }
    )


def _validate_kinematics_identity(value: Any) -> dict[str, str]:
    normalized = _json_copy(value, "kinematics")
    if not isinstance(normalized, dict):
        raise InferenceManifestError("kinematics must be a JSON object")
    required = {"urdf_sha256", "base_link", "tool_link"}
    missing = sorted(required - set(normalized))
    unknown = sorted(set(normalized) - required)
    if missing:
        raise InferenceManifestError(f"kinematics is missing: {', '.join(missing)}")
    if unknown:
        raise InferenceManifestError(f"unknown kinematics fields: {', '.join(unknown)}")
    digest = normalized["urdf_sha256"]
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise InferenceManifestError("kinematics.urdf_sha256 must be 64 lowercase hexadecimal characters")
    for key in ("base_link", "tool_link"):
        link = normalized[key]
        if not isinstance(link, str) or not link or link != link.strip():
            raise InferenceManifestError(f"kinematics.{key} must be a non-empty trimmed string")
        _reject_absolute_paths(link, f"kinematics.{key}")
    if normalized["base_link"] == normalized["tool_link"]:
        raise InferenceManifestError("kinematics.base_link and tool_link must differ")
    return {key: normalized[key] for key in ("urdf_sha256", "base_link", "tool_link")}


def verify_checkpoint_integrity(
    checkpoint_or_file: str | Path,
    manifest: InferenceManifest,
    *,
    allow_legacy: bool = False,
    allow_template: bool = False,
) -> bool:
    """Verify every inference-critical checkpoint file against ``manifest``.

    Returns ``True`` only for a cryptographically bound checkpoint. ``False`` is
    returned solely for an explicitly allowed manifest-free legacy checkpoint or
    an explicitly allowed unbound output-root template.
    """

    source = Path(checkpoint_or_file).expanduser()
    directory = source if source.is_dir() else source.parent
    manifest_path = directory / MANIFEST_FILENAME
    if manifest_path.is_file():
        on_disk = _read_manifest_file(manifest_path)
        if manifest.checkpoint_files is None and on_disk.checkpoint_files is not None:
            _require_same_manifest(
                replace(on_disk, checkpoint_files=None),
                manifest,
                "programmatic manifest does not match checkpoint manifest",
            )
            manifest = on_disk
        else:
            _require_same_manifest(
                on_disk, manifest, "programmatic manifest does not match checkpoint manifest"
            )
    if manifest.checkpoint_files is None:
        if allow_legacy and not manifest_path.is_file():
            return False
        if allow_template and manifest_path.is_file() and not _looks_like_checkpoint(directory):
            return False
        if manifest_path.is_file():
            raise InferenceManifestError(
                "checkpoint manifest has no checkpoint_files binding; only an output-root template may be unbound"
            )
        raise InferenceManifestError(
            "checkpoint has no cryptographic file binding; pass allow_legacy=True only for an explicit legacy fallback"
        )

    actual = _checkpoint_file_hashes(directory)
    expected_names = set(manifest.checkpoint_files)
    actual_names = set(actual)
    if actual_names != expected_names:
        missing = sorted(expected_names - actual_names)
        unexpected = sorted(actual_names - expected_names)
        details = []
        if missing:
            details.append(f"missing from checkpoint: {', '.join(missing)}")
        if unexpected:
            details.append(f"not bound by manifest: {', '.join(unexpected)}")
        raise InferenceManifestError(f"checkpoint file set does not match manifest ({'; '.join(details)})")
    for name, expected_digest in manifest.checkpoint_files.items():
        if actual[name] != expected_digest:
            raise InferenceManifestError(f"checkpoint file SHA-256 mismatch: {name}")
    return True


def _validate_checkpoint_files(value: Any) -> dict[str, str] | None:
    if value is None:
        return None
    normalized = _json_copy(value, "checkpoint_files")
    if not isinstance(normalized, dict) or not normalized:
        raise InferenceManifestError("checkpoint_files must be null or a non-empty JSON object")
    if not all(isinstance(name, str) and isinstance(digest, str) for name, digest in normalized.items()):
        raise InferenceManifestError("checkpoint_files must map string filenames to SHA-256 strings")
    missing_required = sorted(set(_REQUIRED_CHECKPOINT_FILES) - set(normalized))
    if missing_required:
        raise InferenceManifestError(
            f"checkpoint_files is missing required files: {', '.join(missing_required)}"
        )
    for name, digest in normalized.items():
        _validate_checkpoint_filename(name, context="checkpoint_files")
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise InferenceManifestError(
                f"checkpoint_files[{name!r}] must be 64 lowercase hexadecimal characters"
            )
    return {name: normalized[name] for name in sorted(normalized)}


def _checkpoint_file_hashes(directory: Path) -> dict[str, str]:
    if not directory.is_dir():
        raise InferenceManifestError(f"checkpoint directory does not exist: {directory.resolve()}")
    names = set(_REQUIRED_CHECKPOINT_FILES)
    missing_required = [name for name in _REQUIRED_CHECKPOINT_FILES if not (directory / name).is_file()]
    if missing_required:
        raise InferenceManifestError(
            f"checkpoint is missing inference-critical files: {', '.join(missing_required)}"
        )
    for config_name in _PROCESSOR_CONFIG_FILES:
        names.update(_processor_state_files(directory / config_name))
    return {name: _sha256_file(directory / name, name) for name in sorted(names)}


def _processor_state_files(config_path: Path) -> set[str]:
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise InferenceManifestError(f"could not read processor config {config_path.name}: {exc}") from exc
    if not isinstance(raw, dict) or not isinstance(raw.get("steps"), list):
        raise InferenceManifestError(f"processor config {config_path.name} must contain a steps list")
    result: set[str] = set()
    for index, step in enumerate(raw["steps"]):
        if not isinstance(step, dict):
            raise InferenceManifestError(
                f"processor config {config_path.name} step {index} must be an object"
            )
        state_file = step.get("state_file")
        if state_file is None:
            continue
        _validate_checkpoint_filename(
            state_file,
            context=f"processor config {config_path.name} step {index}.state_file",
        )
        if not state_file.endswith(".safetensors") or state_file in _REQUIRED_CHECKPOINT_FILES:
            raise InferenceManifestError(
                f"processor config {config_path.name} step {index}.state_file must name a dedicated safetensors file"
            )
        if not (config_path.parent / state_file).is_file():
            raise InferenceManifestError(f"processor state file is missing: {state_file}")
        result.add(state_file)
    return result


def _validate_checkpoint_filename(value: Any, *, context: str) -> None:
    if (
        not isinstance(value, str)
        or not value
        or PurePosixPath(value).name != value
        or PureWindowsPath(value).name != value
    ):
        raise InferenceManifestError(f"{context} must use portable checkpoint-local filenames")
    if value not in _REQUIRED_CHECKPOINT_FILES and not value.endswith(".safetensors"):
        raise InferenceManifestError(f"{context} contains a non-critical checkpoint filename: {value}")


def _sha256_file(path: Path, name: str) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise InferenceManifestError(f"could not fingerprint checkpoint file {name}: {exc}") from exc
    return digest.hexdigest()


def _looks_like_checkpoint(directory: Path) -> bool:
    return (
        directory.name == "pretrained_model"
        or any((directory / name).exists() for name in _REQUIRED_CHECKPOINT_FILES)
        or any(directory.glob("*.safetensors"))
    )


def _reject_absolute_paths(value: Any, location: str) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            _reject_absolute_paths(key, f"{location}.<key>")
            _reject_absolute_paths(child, f"{location}.{key}")
        return
    if isinstance(value, list):
        for index, child in enumerate(value):
            _reject_absolute_paths(child, f"{location}[{index}]")
        return
    if isinstance(value, str) and (
        PurePosixPath(value).is_absolute() or PureWindowsPath(value).is_absolute()
    ):
        raise InferenceManifestError(f"{location} must not contain an absolute path")


def _manifest_json(manifest: InferenceManifest) -> str:
    return json.dumps(manifest.to_dict(), ensure_ascii=False, allow_nan=False, indent=2) + "\n"


def _read_manifest_file(path: Path) -> InferenceManifest:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise InferenceManifestError(f"could not read inference manifest {path}: {exc}") from exc
    return InferenceManifest.from_dict(raw)


def _require_same_kinematics(
    manifest_identity: dict[str, str],
    selected_identity: dict[str, str],
) -> None:
    if manifest_identity != selected_identity:
        raise InferenceManifestError(
            "selected URDF fingerprint or base_link/tool_link does not match the inference manifest"
        )


def _require_same_manifest(
    left: InferenceManifest,
    right: InferenceManifest,
    message: str,
) -> None:
    if _manifest_json(left) != _manifest_json(right):
        raise InferenceManifestError(message)
