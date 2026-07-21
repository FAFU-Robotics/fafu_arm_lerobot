"""Small compatibility layer for LeRobot 0.4 through 0.6."""

from __future__ import annotations

from typing import Any


def make_cameras(configs: dict[str, Any]) -> dict[str, Any]:
    try:
        from lerobot.cameras import make_cameras_from_configs
    except ImportError:  # LeRobot 0.4
        from lerobot.cameras.utils import make_cameras_from_configs

    return make_cameras_from_configs(configs)


def read_camera_rgb(camera: Any) -> Any:
    if hasattr(camera, "read_latest"):
        return camera.read_latest()
    if hasattr(camera, "async_read"):
        return camera.async_read()
    return camera.read()


def read_camera_depth(camera: Any) -> Any:
    if hasattr(camera, "read_latest_depth"):
        depth = camera.read_latest_depth()
    elif hasattr(camera, "read_depth"):
        depth = camera.read_depth()
    else:
        raise RuntimeError(f"Camera {camera!r} is configured for depth but has no depth read method")

    if getattr(depth, "ndim", None) == 2:
        depth = depth[..., None]
    return depth
