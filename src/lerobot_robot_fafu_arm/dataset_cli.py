"""Inspect a local LeRobot dataset before replaying it on a FAFU arm."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .representation import ACTION_MODES, action_features


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate local LeRobot dataset metadata before FAFU arm replay"
    )
    parser.add_argument("--root", type=Path, required=True, help="Local LeRobot dataset root")
    parser.add_argument(
        "--action-mode",
        choices=sorted(ACTION_MODES),
        required=True,
        help="FAFU action mode that will be used for replay",
    )
    parser.add_argument("--episode", type=int, default=0, help="Episode index to validate")
    parser.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="Print a machine-readable report",
    )
    return parser


def inspect_dataset(root: Path, action_mode: str, episode: int) -> dict[str, Any]:
    info_path = root.expanduser().resolve() / "meta" / "info.json"
    errors: list[str] = []
    warnings: list[str] = []

    if episode < 0:
        errors.append("episode must be non-negative")

    try:
        info = json.loads(info_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {
            "ok": False,
            "info_path": str(info_path),
            "errors": [f"metadata file not found: {info_path}"],
            "warnings": [],
        }
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "ok": False,
            "info_path": str(info_path),
            "errors": [f"could not read dataset metadata: {exc}"],
            "warnings": [],
        }

    features = info.get("features")
    action_feature = features.get("action") if isinstance(features, dict) else None
    actual_names = action_feature.get("names") if isinstance(action_feature, dict) else None
    expected_names = list(action_features(action_mode))

    if not isinstance(actual_names, list) or not all(isinstance(name, str) for name in actual_names):
        errors.append("metadata does not contain a valid features.action.names list")
        actual_names = []
    elif actual_names != expected_names:
        missing = [name for name in expected_names if name not in actual_names]
        unexpected = [name for name in actual_names if name not in expected_names]
        if missing:
            errors.append(f"missing action fields: {', '.join(missing)}")
        if unexpected:
            errors.append(f"unexpected action fields: {', '.join(unexpected)}")
        if not missing and not unexpected:
            errors.append("action field order does not match the selected FAFU action mode")

    total_episodes = info.get("total_episodes")
    if not isinstance(total_episodes, int) or total_episodes < 0:
        errors.append("metadata total_episodes is missing or invalid")
    elif episode >= total_episodes:
        errors.append(f"episode {episode} is out of range; dataset contains {total_episodes} episode(s)")

    fps = info.get("fps")
    if not isinstance(fps, (int, float)) or isinstance(fps, bool) or fps <= 0:
        errors.append("metadata fps is missing or invalid")

    robot_type = info.get("robot_type")
    if robot_type not in {None, "fafu_follower"}:
        warnings.append(f"dataset robot_type is {robot_type!r}, not 'fafu_follower'")

    camera_features = []
    if isinstance(features, dict):
        camera_features = [
            name
            for name, feature in features.items()
            if isinstance(feature, dict) and feature.get("dtype") in {"image", "video"}
        ]
    if not camera_features:
        warnings.append("dataset declares no RGB/depth camera features")

    return {
        "ok": not errors,
        "info_path": str(info_path),
        "action_mode": action_mode,
        "action_names": actual_names,
        "episode": episode,
        "total_episodes": total_episodes,
        "fps": fps,
        "robot_type": robot_type,
        "camera_features": camera_features,
        "errors": errors,
        "warnings": warnings,
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = inspect_dataset(args.root, args.action_mode, args.episode)
    if args.as_json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        status = "OK" if report["ok"] else "FAIL"
        print(f"[{status}] dataset metadata: {report['info_path']}")
        if "action_mode" in report:
            print(f"[INFO] action mode: {report['action_mode']}")
            total_episodes = report["total_episodes"]
            last_episode = max(total_episodes - 1, 0) if isinstance(total_episodes, int) else "unknown"
            print(f"[INFO] episode: {report['episode']} / {last_episode}")
            print(f"[INFO] fps: {report['fps']}")
            cameras = ", ".join(report["camera_features"]) or "none"
            print(f"[INFO] cameras: {cameras}")
        for warning in report["warnings"]:
            print(f"[WARN] {warning}")
        for error in report["errors"]:
            print(f"[FAIL] {error}", file=sys.stderr)
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
