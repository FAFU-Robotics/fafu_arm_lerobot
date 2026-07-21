"""Runtime discovery for the separately maintained FAFU arm SDK."""

from __future__ import annotations

import importlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any


class FafuSDKError(ImportError):
    """Raised when the FAFU SDK cannot be found or loaded."""


@dataclass(frozen=True)
class SDKBindings:
    controller_class: type[Any]
    servo_options_class: type[Any]
    python_dir: Path | None


def _as_python_dir(path: Path) -> Path | None:
    path = path.expanduser().resolve()
    candidates = (path, path / "fafu_robot_python")
    for candidate in candidates:
        if (candidate / "fafu_robot_controller.py").is_file():
            return candidate
    return None


def sdk_search_paths(explicit_path: str | os.PathLike[str] | None = None) -> list[Path]:
    """Return candidate SDK Python directories in priority order."""

    candidates: list[Path] = []
    if explicit_path:
        candidates.append(Path(explicit_path))
    if env_path := os.environ.get("FAFU_ARM_SDK_PATH"):
        candidates.append(Path(env_path))

    project_root = Path(__file__).resolve().parents[2]
    cwd = Path.cwd()
    candidates.extend(
        [
            project_root / "third_party" / "fafu_arm_sdk",
            project_root.parent / "fafu_arm_sdk",
            cwd / "third_party" / "fafu_arm_sdk",
            cwd / "fafu_arm_sdk",
        ]
    )

    resolved: list[Path] = []
    for candidate in candidates:
        python_dir = _as_python_dir(candidate)
        if python_dir is not None and python_dir not in resolved:
            resolved.append(python_dir)
    return resolved


def _import_sdk_modules() -> tuple[ModuleType, ModuleType]:
    package = importlib.import_module("fafu_robot_python")
    implementation = importlib.import_module("fafu_robot_controller")
    return package, implementation


def load_sdk(explicit_path: str | os.PathLike[str] | None = None) -> SDKBindings:
    """Load SDK classes without requiring the SDK to be a pip package."""

    first_error: Exception | None = None
    try:
        package, implementation = _import_sdk_modules()
        return SDKBindings(package.FafuRobotController, implementation.ServoOpts, None)
    except (ImportError, OSError, AttributeError) as exc:
        first_error = exc

    searched = sdk_search_paths(explicit_path)
    for python_dir in searched:
        path_text = os.fspath(python_dir)
        if path_text not in sys.path:
            sys.path.insert(0, path_text)
        try:
            implementation = importlib.import_module("fafu_robot_controller")
            return SDKBindings(
                implementation.FafuRobotController,
                implementation.ServoOpts,
                python_dir,
            )
        except (ImportError, OSError, AttributeError) as exc:
            first_error = exc

    details = f" Last import error: {first_error}" if first_error else ""
    raise FafuSDKError(
        "FAFU arm SDK could not be loaded. Clone this repository with "
        "--recurse-submodules, or set FAFU_ARM_SDK_PATH to the fafu_arm_sdk checkout. "
        "On Windows, rebuild fafu_motor.pyd for the active Python ABI if needed."
        f" Searched: {[str(path) for path in searched]}.{details}"
    ) from first_error


def default_sdk_config_path() -> Path:
    """Return the packaged, hardware-ready default controller config."""

    return Path(__file__).resolve().parent / "resources" / "fafu_arm.cfg"
