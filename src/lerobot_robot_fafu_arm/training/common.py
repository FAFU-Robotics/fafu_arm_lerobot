"""Dataset and runtime checks shared by policy training launchers."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from ..local_dataset import DatasetReadError, load_dataset_info, load_episode
from ..representation import action_features

TRAINING_ACTION_MODES = ("joint", "ee_delta", "ee_pose")
_SUPPORTED_LEROBOT_MIN = (0, 4, 3)
_SUPPORTED_LEROBOT_MAX = (0, 7, 0)


@dataclass(frozen=True)
class TrainingDatasetReport:
    """Machine-readable result of a training preflight."""

    root: str
    action_mode: str
    lerobot_version: str | None
    total_episodes: int | None
    total_frames: int | None
    fps: float | None
    action_names: tuple[str, ...]
    camera_features: tuple[str, ...]
    sampled_episode: int | None
    errors: tuple[str, ...]
    warnings: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["ok"] = self.ok
        return result


def check_training_dataset(root: str | Path, action_mode: str) -> TrainingDatasetReport:
    """Validate a local FAFU dataset before starting policy training."""

    dataset_root = Path(root).expanduser().resolve()
    errors: list[str] = []
    warnings: list[str] = []
    lerobot_version = _check_lerobot_runtime(errors, warnings)

    if action_mode not in TRAINING_ACTION_MODES:
        errors.append(
            "training action mode must be one of joint, ee_delta, or ee_pose; "
            "the redundant all mode must be filtered before training"
        )

    try:
        info = load_dataset_info(dataset_root)
    except DatasetReadError as exc:
        return TrainingDatasetReport(
            root=str(dataset_root),
            action_mode=action_mode,
            lerobot_version=lerobot_version,
            total_episodes=None,
            total_frames=None,
            fps=None,
            action_names=(),
            camera_features=(),
            sampled_episode=None,
            errors=tuple((*errors, str(exc))),
            warnings=tuple(warnings),
        )

    action_names = info.feature_names("action")
    if action_mode in TRAINING_ACTION_MODES:
        expected = tuple(action_features(action_mode))
        if action_names != expected:
            missing = [name for name in expected if name not in action_names]
            unexpected = [name for name in action_names if name not in expected]
            if missing:
                errors.append(f"action is missing fields for {action_mode}: {', '.join(missing)}")
            if unexpected:
                errors.append(f"action contains fields outside {action_mode}: {', '.join(unexpected)}")
            if not missing and not unexpected:
                errors.append(f"action field order does not match {action_mode}")

    action_feature = info.features.get("action", {})
    action_shape = action_feature.get("shape") if isinstance(action_feature, dict) else None
    if action_names and action_shape not in ([len(action_names)], (len(action_names),)):
        errors.append(f"features.action.shape must be [{len(action_names)}], got {action_shape!r}")

    camera_features = tuple(
        key
        for key, feature in info.features.items()
        if isinstance(feature, dict) and feature.get("dtype") in {"image", "video"}
    )
    if not camera_features:
        errors.append("ACT training requires at least one declared camera feature for this real-robot setup")
    _check_camera_shapes(info.features, camera_features, errors)

    if "observation.state" not in info.features:
        warnings.append(
            "observation.state is absent; ACT can be visual-only, but the recommended FAFU baseline uses proprioception"
        )

    total_frames_raw = info.raw.get("total_frames")
    total_frames = (
        total_frames_raw
        if isinstance(total_frames_raw, int) and not isinstance(total_frames_raw, bool)
        else None
    )
    if total_frames is None or total_frames <= 0:
        errors.append("metadata total_frames is missing or not positive")
    if info.total_episodes <= 0:
        errors.append("dataset contains no episodes")
    elif info.total_episodes < 50:
        warnings.append(
            f"dataset has {info.total_episodes} episode(s); 50 clean demonstrations is a useful first ACT target, not a hard minimum"
        )

    if info.raw.get("robot_type") not in {None, "fafu_follower"}:
        warnings.append(f"dataset robot_type is {info.raw.get('robot_type')!r}, not 'fafu_follower'")
    if not (dataset_root / "meta" / "stats.json").is_file():
        warnings.append("meta/stats.json is missing; LeRobot needs valid normalization statistics")

    sampled_episode: int | None = None
    if info.total_episodes > 0:
        try:
            episode = load_episode(dataset_root, 0)
            action = episode.feature_matrix("action")
            if action_names and action.shape[1] != len(action_names):
                errors.append(
                    f"sample action width is {action.shape[1]}, but metadata declares {len(action_names)} fields"
                )
            if "observation.state" in episode.columns:
                state = episode.feature_matrix("observation.state")
                state_names = info.feature_names("observation.state")
                if state_names and state.shape[1] != len(state_names):
                    errors.append(
                        f"sample state width is {state.shape[1]}, but metadata declares {len(state_names)} fields"
                    )
            sampled_episode = 0
        except DatasetReadError as exc:
            errors.append(f"could not read a finite sample episode: {exc}")

    return TrainingDatasetReport(
        root=str(dataset_root),
        action_mode=action_mode,
        lerobot_version=lerobot_version,
        total_episodes=info.total_episodes,
        total_frames=total_frames,
        fps=info.fps,
        action_names=action_names,
        camera_features=camera_features,
        sampled_episode=sampled_episode,
        errors=tuple(errors),
        warnings=tuple(warnings),
    )


def format_training_report(report: TrainingDatasetReport) -> str:
    """Render a compact human-readable preflight report."""

    lines = [f"[{'OK' if report.ok else 'FAIL'}] training dataset: {report.root}"]
    lines.append(f"[INFO] action mode: {report.action_mode}")
    lines.append(f"[INFO] LeRobot: {report.lerobot_version or 'not installed'}")
    if report.total_episodes is not None:
        lines.append(
            f"[INFO] episodes / frames / fps: {report.total_episodes} / {report.total_frames} / {report.fps:g}"
        )
    lines.append(f"[INFO] cameras: {', '.join(report.camera_features) or 'none'}")
    lines.extend(f"[WARN] {warning}" for warning in report.warnings)
    lines.extend(f"[FAIL] {error}" for error in report.errors)
    return "\n".join(lines)


def _check_lerobot_runtime(errors: list[str], warnings: list[str]) -> str | None:
    try:
        installed = version("lerobot")
    except PackageNotFoundError:
        errors.append("LeRobot is not installed; run `python -m pip install -e .` first")
        return None

    match = re.match(r"(\d+)\.(\d+)\.(\d+)", installed)
    if match is None:
        warnings.append(f"could not parse LeRobot version {installed!r}")
        return installed
    parsed = tuple(int(part) for part in match.groups())
    if not (_SUPPORTED_LEROBOT_MIN <= parsed < _SUPPORTED_LEROBOT_MAX):
        errors.append("training launcher supports LeRobot >=0.4.3,<0.7; install a compatible version")
    elif parsed < (0, 6, 0):
        warnings.append("the training guide and advanced evaluation flags target LeRobot 0.6")
    return installed


def _check_camera_shapes(
    features: dict[str, dict[str, Any]],
    camera_features: tuple[str, ...],
    errors: list[str],
) -> None:
    shapes: dict[str, tuple[int, ...]] = {}
    for key in camera_features:
        raw_shape = features[key].get("shape")
        if not isinstance(raw_shape, (list, tuple)) or not all(isinstance(value, int) for value in raw_shape):
            errors.append(f"camera feature {key!r} has an invalid shape: {raw_shape!r}")
            continue
        shapes[key] = tuple(raw_shape)
    if len(set(shapes.values())) > 1:
        details = ", ".join(f"{key}={shape}" for key, shape in shapes.items())
        errors.append(f"ACT camera features must use the same shape; found {details}")
