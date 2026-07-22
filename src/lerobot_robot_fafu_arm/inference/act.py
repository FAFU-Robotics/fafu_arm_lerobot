"""Strict ACT inference bridge for FAFU action and observation schemas."""

from __future__ import annotations

import math
import re
import time
from collections import deque
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np

from .manifest import InferenceManifest, build_kinematics_identity, verify_checkpoint_integrity

_ALLOWED_POLICY_TYPES = frozenset({"act", "fafu_act_demo"})
_MIN_LEROBOT = (0, 6, 0)
_MAX_LEROBOT = (0, 7, 0)


@dataclass(frozen=True)
class InferenceRunReport:
    """Timing summary returned after a finite hardware rollout."""

    steps: int
    elapsed_s: float
    overruns: int
    max_control_latency_s: float


class ActPolicyRuntime:
    """Loaded ACT policy plus its checkpoint-owned pre/post-processors."""

    def __init__(
        self,
        *,
        policy: Any,
        preprocessor: Any,
        postprocessor: Any,
        manifest: InferenceManifest,
        device: Any,
        task: str,
        torch_module: Any,
        build_inference_frame: Callable[..., dict[str, Any]],
        make_robot_action: Callable[[Any, dict[str, dict[str, Any]]], dict[str, Any]],
    ) -> None:
        self.policy = policy
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.manifest = manifest
        self.device = device
        self.task = task
        self._torch = torch_module
        self._build_inference_frame = build_inference_frame
        self._make_robot_action = make_robot_action

    @classmethod
    def load(
        cls,
        checkpoint: str | Path,
        manifest: InferenceManifest,
        *,
        device: str | None = None,
        task: str = "",
        allow_legacy_checkpoint: bool = False,
    ) -> ActPolicyRuntime:
        """Strictly load a LeRobot 0.6 ACT checkpoint and its processors."""

        verify_checkpoint_integrity(
            checkpoint,
            manifest,
            allow_legacy=allow_legacy_checkpoint,
        )
        _require_lerobot_06()

        import torch
        from lerobot.configs import PreTrainedConfig
        from lerobot.policies import get_policy_class, make_pre_post_processors
        from lerobot.policies.utils import build_inference_frame, make_robot_action
        from lerobot.utils.import_utils import register_third_party_plugins

        register_third_party_plugins()
        target_device = _resolve_device(torch, device)
        policy_config = PreTrainedConfig.from_pretrained(checkpoint)
        policy_type = getattr(policy_config, "type", None)
        if policy_type not in _ALLOWED_POLICY_TYPES:
            raise ValueError(
                f"checkpoint policy type must be one of {sorted(_ALLOWED_POLICY_TYPES)}, got {policy_type!r}"
            )
        validate_policy_schema(policy_config, manifest)

        policy_config.device = str(target_device)
        policy_config.pretrained_path = checkpoint
        policy_class = get_policy_class(policy_type)
        policy = policy_class.from_pretrained(checkpoint, config=policy_config, strict=True)
        policy = policy.to(target_device)
        policy.eval()
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=policy_config,
            pretrained_path=str(checkpoint),
            preprocessor_overrides={"device_processor": {"device": str(target_device)}},
        )
        return cls(
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            manifest=manifest,
            device=target_device,
            task=task,
            torch_module=torch,
            build_inference_frame=build_inference_frame,
            make_robot_action=make_robot_action,
        )

    def reset(self) -> None:
        """Clear ACT action queues and all stateful processor steps."""

        self.policy.reset()
        self.preprocessor.reset()
        self.postprocessor.reset()

    def predict(self, observation: dict[str, Any]) -> dict[str, float]:
        """Run one normalized ACT step and decode it with manifest field names."""

        validate_observation(observation, self.manifest)
        frame = self._build_inference_frame(
            observation=observation,
            device=self.device,
            ds_features=self.manifest.features,
            task=self.task,
            robot_type=self.manifest.robot_type,
        )
        autocast = (
            self._torch.autocast(device_type="cuda")
            if self.device.type == "cuda" and bool(getattr(self.policy.config, "use_amp", False))
            else nullcontext()
        )
        with self._torch.inference_mode(), autocast:
            processed = self.preprocessor(frame)
            action = self.policy.select_action(processed)
            action = self.postprocessor(action)

        expected_width = len(self.manifest.action_names)
        if not isinstance(action, self._torch.Tensor):
            raise RuntimeError(f"ACT postprocessor returned {type(action).__name__}, expected a tensor")
        if tuple(action.shape) != (1, expected_width):
            raise RuntimeError(f"ACT action must have shape (1, {expected_width}), got {tuple(action.shape)}")
        if not bool(self._torch.isfinite(action).all().item()):
            raise RuntimeError("ACT action contains NaN or infinity")

        decoded = self._make_robot_action(action, self.manifest.features)
        if tuple(decoded) != self.manifest.action_names:
            raise RuntimeError(
                "decoded action names/order differ from the training manifest: "
                f"expected {list(self.manifest.action_names)}, got {list(decoded)}"
            )
        result = {name: float(decoded[name]) for name in self.manifest.action_names}
        if not all(math.isfinite(value) for value in result.values()):
            raise RuntimeError("decoded ACT action contains NaN or infinity")
        return result

    def synthetic_warmup(self) -> float:
        """Run two no-motion forwards and return warmed full-chunk latency."""

        observation = build_synthetic_observation(self.manifest)
        self.reset()
        self.predict(observation)
        self.reset()
        started_at = time.perf_counter()
        self.predict(observation)
        latency = time.perf_counter() - started_at
        self.reset()
        return latency


def validate_policy_schema(policy_config: Any, manifest: InferenceManifest) -> None:
    """Require checkpoint tensor features to match the manifest exactly."""

    expected_inputs: dict[str, tuple[int, ...]] = {}
    for key, feature in manifest.features.items():
        if key == "observation.state" or key.startswith("observation.images."):
            expected_inputs[key] = _policy_shape(feature, key)
    expected_outputs = {"action": _policy_shape(manifest.features["action"], "action")}

    actual_inputs = _config_feature_shapes(getattr(policy_config, "input_features", None), "input")
    actual_outputs = _config_feature_shapes(getattr(policy_config, "output_features", None), "output")
    if tuple(actual_inputs) != tuple(expected_inputs) or actual_inputs != expected_inputs:
        raise ValueError(
            "checkpoint input features differ from the training manifest: "
            f"expected {expected_inputs}, got {actual_inputs}"
        )
    if tuple(actual_outputs) != tuple(expected_outputs) or actual_outputs != expected_outputs:
        raise ValueError(
            "checkpoint output features differ from the training manifest: "
            f"expected {expected_outputs}, got {actual_outputs}"
        )


def derive_observation_settings(manifest: InferenceManifest) -> tuple[str, bool, bool]:
    """Derive the narrowest FAFU observation configuration that supplies the schema."""

    names = manifest.state_names
    has_joint = any(name.startswith("joint") for name in names)
    has_ee = any(name.startswith("ee.") for name in names)
    has_delta = any(name.startswith("ee_delta.") for name in names)
    if has_delta or (has_joint and has_ee):
        mode = "all"
    elif has_ee:
        mode = "ee_pose"
    else:
        mode = "joint"
    record_velocity = any(name.endswith(".vel") for name in names)
    record_effort = any(name.endswith(".effort") for name in names)
    return mode, record_velocity, record_effort


def validate_robot_schema(robot: Any, manifest: InferenceManifest, *, fps: float) -> None:
    """Validate all policy-facing robot fields before hardware connection."""

    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("inference fps must be finite and positive")
    if not math.isclose(fps, manifest.fps, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(f"inference fps {fps:g} must equal training fps {manifest.fps:g}")

    robot_type = getattr(robot, "name", None)
    if robot_type != manifest.robot_type:
        raise ValueError(
            f"robot type differs from the training manifest: expected {manifest.robot_type!r}, got {robot_type!r}"
        )
    kinematics = getattr(robot, "kinematics", None)
    if kinematics is None:
        raise ValueError("robot does not expose its URDF kinematics identity")
    actual_kinematics = build_kinematics_identity(
        getattr(kinematics, "urdf_path", None),
        getattr(kinematics, "base_link", "base_link"),
        getattr(kinematics, "tip_link", "tool_link"),
    )
    if actual_kinematics != manifest.kinematics:
        raise ValueError("robot URDF fingerprint or base_link/tool_link differs from the manifest")
    action_names = tuple(robot.action_features)
    if action_names != manifest.action_names:
        raise ValueError(
            "robot action names/order differ from the training manifest: "
            f"expected {list(manifest.action_names)}, got {list(action_names)}"
        )

    observation_features = robot.observation_features
    missing_state = [name for name in manifest.state_names if observation_features.get(name) is not float]
    if missing_state:
        raise ValueError(f"robot cannot provide manifest state fields: {missing_state}")

    expected_cameras = {
        key.removeprefix("observation.images."): _raw_camera_shape(feature, key)
        for key, feature in manifest.camera_features.items()
    }
    actual_cameras = {
        key: tuple(value) for key, value in observation_features.items() if isinstance(value, tuple)
    }
    if tuple(actual_cameras) != tuple(expected_cameras) or actual_cameras != expected_cameras:
        raise ValueError(
            "robot camera keys/shapes differ from the training manifest: "
            f"expected {expected_cameras}, got {actual_cameras}"
        )

    config = getattr(robot, "config", None)
    if config is not None:
        if getattr(config, "action_mode", None) != manifest.action_mode:
            raise ValueError("robot action_mode differs from the training manifest")
        if not bool(getattr(config, "strict_action_features", False)):
            raise ValueError("strict_action_features must remain enabled for policy inference")
        if bool(getattr(config, "use_servo", False)) and not math.isclose(
            float(getattr(config, "servo_rate_hz", 0.0)), fps, rel_tol=0.0, abs_tol=1e-9
        ):
            raise ValueError("servo_rate_hz must equal the training/inference fps")
        for name, camera in getattr(config, "cameras", {}).items():
            camera_fps = getattr(camera, "fps", None)
            if (
                isinstance(camera_fps, bool)
                or not isinstance(camera_fps, (int, float))
                or not math.isfinite(float(camera_fps))
                or not math.isclose(float(camera_fps), fps, rel_tol=0.0, abs_tol=1e-9)
            ):
                raise ValueError(f"camera {name!r} fps must equal the training/inference fps")


def build_synthetic_observation(manifest: InferenceManifest) -> dict[str, Any]:
    """Build a zero-valued raw frame for no-hardware processor/model validation."""

    observation: dict[str, Any] = {name: 0.0 for name in manifest.state_names}
    for key, feature in manifest.camera_features.items():
        raw_shape = _raw_camera_shape(feature, key)
        observation[key.removeprefix("observation.images.")] = np.zeros(raw_shape, dtype=np.uint8)
    return observation


def validate_observation(observation: dict[str, Any], manifest: InferenceManifest) -> None:
    """Validate one raw policy frame before tensor conversion."""

    if not isinstance(observation, dict):
        raise TypeError(f"observation must be a dict, got {type(observation).__name__}")
    camera_names = tuple(key.removeprefix("observation.images.") for key in manifest.camera_features)
    expected = set(manifest.state_names) | set(camera_names)
    provided = set(observation)
    missing = sorted(expected - provided)
    unexpected = sorted(provided - expected)
    if missing or unexpected:
        raise RuntimeError(
            f"observation fields differ from the training manifest: missing={missing}, "
            f"unexpected={unexpected}"
        )

    for name in manifest.state_names:
        value = observation[name]
        if isinstance(value, (bool, np.bool_)) or np.asarray(value).shape != ():
            raise RuntimeError(f"observation state {name!r} must be a scalar")
        try:
            scalar = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise RuntimeError(f"observation state {name!r} is not numeric") from exc
        if not math.isfinite(scalar):
            raise RuntimeError(f"observation state {name!r} contains NaN or infinity")

    for key, feature in manifest.camera_features.items():
        name = key.removeprefix("observation.images.")
        image = observation[name]
        if not isinstance(image, np.ndarray):
            raise RuntimeError(f"camera {name!r} must return a NumPy array")
        expected_shape = _raw_camera_shape(feature, key)
        if image.shape != expected_shape:
            raise RuntimeError(f"camera {name!r} returned shape {image.shape}; expected {expected_shape}")
        if image.dtype == np.uint8:
            continue
        if not np.issubdtype(image.dtype, np.floating):
            raise RuntimeError(f"camera {name!r} dtype must be uint8 or floating point in [0, 1]")
        if not np.isfinite(image).all() or np.any(image < 0.0) or np.any(image > 1.0):
            raise RuntimeError(f"camera {name!r} floating values must be finite and in [0, 1]")


def run_control_loop(
    robot: Any,
    runtime: ActPolicyRuntime,
    *,
    fps: float,
    duration_s: float,
    max_consecutive_overruns: int = 3,
    max_overruns_per_second: int = 3,
    initial_observation_validator: Callable[[dict[str, Any]], None] | None = None,
    clock: Callable[[], float] = time.perf_counter,
    sleeper: Callable[[float], None] = time.sleep,
) -> InferenceRunReport:
    """Run a finite ACT rollout and always disconnect on exit or failure."""

    if not math.isfinite(duration_s) or duration_s <= 0:
        raise ValueError("duration_s must be finite and positive")
    if max_consecutive_overruns < 1:
        raise ValueError("max_consecutive_overruns must be positive")
    if max_overruns_per_second < 1:
        raise ValueError("max_overruns_per_second must be positive")
    if bool(robot.is_connected):
        raise RuntimeError("inference requires a disconnected robot")
    validate_robot_schema(robot, runtime.manifest, fps=fps)

    period = 1.0 / fps
    max_steps = max(1, math.ceil(duration_s * fps))
    steps = 0
    overruns = 0
    consecutive_overruns = 0
    recent_overruns: deque[float] = deque()
    max_latency = 0.0
    runtime.reset()
    started_at = 0.0
    deadline = 0.0
    try:
        robot.connect()
        started_at = clock()
        deadline = started_at
        while steps < max_steps and clock() - started_at < duration_s:
            step_started = clock()
            observation = robot.get_observation()
            if steps == 0 and initial_observation_validator is not None:
                validate_observation(observation, runtime.manifest)
                initial_observation_validator(observation)
            predicted = runtime.predict(observation)
            robot.send_action(dict(predicted))
            step_finished = clock()
            latency = step_finished - step_started
            max_latency = max(max_latency, latency)
            steps += 1

            if latency > period:
                overruns += 1
                consecutive_overruns += 1
                recent_overruns.append(step_finished)
                window_start = step_finished - 1.0
                while recent_overruns and recent_overruns[0] < window_start:
                    recent_overruns.popleft()
                deadline = step_finished
                if consecutive_overruns >= max_consecutive_overruns:
                    raise RuntimeError(
                        f"control loop exceeded the {period * 1e3:.1f} ms deadline "
                        f"{consecutive_overruns} consecutive times"
                    )
                if len(recent_overruns) >= max_overruns_per_second:
                    raise RuntimeError(
                        f"control loop missed {len(recent_overruns)} deadlines within one second"
                    )
            else:
                consecutive_overruns = 0
            deadline += period
            remaining = deadline - clock()
            if remaining > 0:
                sleeper(remaining)
    finally:
        try:
            runtime.reset()
        finally:
            robot.disconnect()

    return InferenceRunReport(
        steps=steps,
        elapsed_s=clock() - started_at,
        overruns=overruns,
        max_control_latency_s=max_latency,
    )


def _require_lerobot_06() -> str:
    try:
        installed = version("lerobot")
    except PackageNotFoundError as exc:
        raise RuntimeError("LeRobot is not installed") from exc
    match = re.match(r"(\d+)\.(\d+)\.(\d+)", installed)
    if match is None:
        raise RuntimeError(f"could not parse LeRobot version {installed!r}")
    parsed = tuple(int(part) for part in match.groups())
    if not (_MIN_LEROBOT <= parsed < _MAX_LEROBOT):
        raise RuntimeError("ACT inference requires LeRobot >=0.6,<0.7")
    return installed


def _resolve_device(torch_module: Any, requested: str | None) -> Any:
    if requested is None:
        if torch_module.cuda.is_available():
            requested = "cuda"
        elif hasattr(torch_module.backends, "mps") and torch_module.backends.mps.is_available():
            requested = "mps"
        else:
            requested = "cpu"
    device = torch_module.device(requested)
    if device.type == "cuda" and not torch_module.cuda.is_available():
        raise ValueError(f"CUDA device {requested!r} was requested but CUDA is unavailable")
    if (
        device.type == "cuda"
        and device.index is not None
        and device.index >= torch_module.cuda.device_count()
    ):
        raise ValueError(f"CUDA device index {device.index} is unavailable")
    if device.type == "mps" and not (
        hasattr(torch_module.backends, "mps") and torch_module.backends.mps.is_available()
    ):
        raise ValueError("MPS was requested but is unavailable")
    if device.type not in {"cpu", "cuda", "mps"}:
        raise ValueError("device must be cpu, cuda, cuda:N, or mps")
    return device


def _config_feature_shapes(features: Any, name: str) -> dict[str, tuple[int, ...]]:
    if not isinstance(features, dict):
        raise ValueError(f"checkpoint {name}_features must be a mapping")
    result: dict[str, tuple[int, ...]] = {}
    for key, feature in features.items():
        shape = getattr(feature, "shape", None)
        if not isinstance(key, str) or not isinstance(shape, (list, tuple)):
            raise ValueError(f"checkpoint {name} feature {key!r} has an invalid shape")
        result[key] = tuple(int(value) for value in shape)
    return result


def _policy_shape(feature: dict[str, Any], key: str) -> tuple[int, ...]:
    shape = tuple(int(value) for value in feature["shape"])
    if feature.get("dtype") not in {"image", "video"}:
        return shape
    names = feature.get("names")
    if names == ["height", "width", "channels"]:
        return (shape[2], shape[0], shape[1])
    if names == ["channels", "height", "width"]:
        return shape
    if shape[2] in {1, 3} and shape[0] not in {1, 3}:
        return (shape[2], shape[0], shape[1])
    if shape[0] in {1, 3} and shape[2] not in {1, 3}:
        return shape
    raise ValueError(f"camera feature {key!r} has an ambiguous channel layout")


def _raw_camera_shape(feature: dict[str, Any], key: str) -> tuple[int, int, int]:
    shape = tuple(int(value) for value in feature["shape"])
    names = feature.get("names")
    if names == ["height", "width", "channels"]:
        return shape
    if names == ["channels", "height", "width"]:
        return (shape[1], shape[2], shape[0])
    if shape[2] in {1, 3} and shape[0] not in {1, 3}:
        return shape
    if shape[0] in {1, 3} and shape[2] not in {1, 3}:
        return (shape[1], shape[2], shape[0])
    raise ValueError(f"camera feature {key!r} has an ambiguous channel layout")
