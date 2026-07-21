"""Inspect a local LeRobot dataset before replaying it on a FAFU arm."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .local_dataset import (
    DatasetReadError,
    dataset_summary,
    export_episode_csv,
    load_dataset_info,
    load_episode,
)
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


def build_collection_parser() -> argparse.ArgumentParser:
    """Build the unified read/check/preview/export command parser."""

    parser = argparse.ArgumentParser(description="Inspect and export a local FAFU LeRobot dataset")
    subparsers = parser.add_subparsers(dest="command", required=True)

    info = subparsers.add_parser("info", help="Show dataset metadata and available features")
    info.add_argument("--root", type=Path, required=True, help="Local LeRobot dataset root")
    info.add_argument("--json", action="store_true", dest="as_json")

    check = subparsers.add_parser("check", help="Validate metadata before replay")
    check.add_argument("--root", type=Path, required=True, help="Local LeRobot dataset root")
    check.add_argument("--action-mode", choices=sorted(ACTION_MODES), required=True)
    check.add_argument("--episode", type=int, default=0)
    check.add_argument("--json", action="store_true", dest="as_json")

    preview = subparsers.add_parser("preview", help="Print low-dimensional rows from one episode")
    preview.add_argument("--root", type=Path, required=True, help="Local LeRobot dataset root")
    preview.add_argument("--episode", type=int, default=0)
    preview.add_argument("--rows", type=int, default=5, help="Number of rows to print")
    preview.add_argument("--json", action="store_true", dest="as_json")

    export = subparsers.add_parser("export", help="Export low-dimensional episode columns to CSV")
    export.add_argument("--root", type=Path, required=True, help="Local LeRobot dataset root")
    export.add_argument("--episode", type=int, default=0)
    export.add_argument("--output", type=Path, required=True)
    export.add_argument("--force", action="store_true", help="Overwrite an existing CSV file")
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


def collection_main(argv: list[str] | None = None) -> int:
    """Entry point for the unified local dataset command."""

    args = build_collection_parser().parse_args(argv)
    if args.command == "check":
        forwarded = [
            "--root",
            str(args.root),
            "--action-mode",
            args.action_mode,
            "--episode",
            str(args.episode),
        ]
        if args.as_json:
            forwarded.append("--json")
        return main(forwarded)

    try:
        if args.command == "info":
            report = dataset_summary(load_dataset_info(args.root))
            if args.as_json:
                print(json.dumps(report, ensure_ascii=False, indent=2))
            else:
                print(f"[OK] dataset: {report['root']}")
                print(f"[INFO] robot: {report['robot_type']}")
                print(f"[INFO] LeRobot format: {report['codebase_version']}")
                print(f"[INFO] episodes / frames: {report['total_episodes']} / {report['total_frames']}")
                print(f"[INFO] fps: {report['fps']:g}")
                print(f"[INFO] features: {', '.join(report['features'])}")
                print(f"[INFO] cameras: {', '.join(report['camera_features']) or 'none'}")
            return 0

        if args.command == "preview" and args.rows < 0:
            raise DatasetReadError("rows must be non-negative")
        episode = load_episode(args.root, args.episode)
        if args.command == "preview":
            records = episode.records(args.rows)
            if args.as_json:
                print(json.dumps(records, ensure_ascii=False, indent=2))
            else:
                print(f"[OK] episode {args.episode}: {len(episode)} frame(s), showing {len(records)} row(s)")
                for index, record in enumerate(records):
                    print(f"[ROW {index}] {json.dumps(record, ensure_ascii=False, separators=(',', ':'))}")
            return 0

        output = export_episode_csv(episode, args.output, overwrite=args.force)
        print(f"[OK] exported {len(episode)} frame(s) to {output}")
        return 0
    except (DatasetReadError, FileExistsError, OSError) as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
