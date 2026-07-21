"""Play a local FAFU LeRobot joint trajectory as a WRS kinematic skeleton."""

from __future__ import annotations

import argparse
import math
import os
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .kinematics import default_urdf_path, rotation_vector_to_matrix
from .local_dataset import DatasetReadError, load_episode


@dataclass(frozen=True)
class _Joint:
    name: str
    kind: str
    parent: str
    child: str
    origin_position: NDArray[np.float64]
    origin_rotation: NDArray[np.float64]
    axis: NDArray[np.float64]


class UrdfSkeleton:
    """Minimal serial-chain FK used only for WRS stick visualization."""

    def __init__(self, urdf_path: str | Path | None = None) -> None:
        self.urdf_path = Path(urdf_path or default_urdf_path()).expanduser().resolve()
        self.joints = self._load_chain(self.urdf_path)
        self.dof = sum(joint.kind != "fixed" for joint in self.joints)

    def frames(
        self, joint_values: NDArray[np.float64] | list[float]
    ) -> list[tuple[NDArray[np.float64], NDArray[np.float64]]]:
        values = np.asarray(joint_values, dtype=np.float64)
        if values.shape != (self.dof,):
            raise ValueError(f"joint_values must have shape ({self.dof},), got {values.shape}")
        if not np.all(np.isfinite(values)):
            raise ValueError("joint_values contains NaN or infinity")

        transform = np.eye(4, dtype=np.float64)
        frames = [(transform[:3, 3].copy(), transform[:3, :3].copy())]
        value_index = 0
        for joint in self.joints:
            origin = np.eye(4, dtype=np.float64)
            origin[:3, 3] = joint.origin_position
            origin[:3, :3] = joint.origin_rotation
            transform = transform @ origin
            if joint.kind != "fixed":
                motion = np.eye(4, dtype=np.float64)
                motion[:3, :3] = rotation_vector_to_matrix(joint.axis * values[value_index])
                transform = transform @ motion
                value_index += 1
            frames.append((transform[:3, 3].copy(), transform[:3, :3].copy()))
        return frames

    @staticmethod
    def _load_chain(urdf_path: Path) -> tuple[_Joint, ...]:
        try:
            root = ET.parse(urdf_path).getroot()
        except (OSError, ET.ParseError) as exc:
            raise ValueError(f"could not parse URDF {urdf_path}: {exc}") from exc

        children: dict[str, ET.Element] = {}
        for element in root.findall("joint"):
            parent = element.find("parent")
            if parent is None or "link" not in parent.attrib:
                raise ValueError("URDF joint is missing its parent link")
            parent_name = parent.attrib["link"]
            if parent_name in children:
                raise ValueError(f"URDF is not a serial chain at parent link {parent_name!r}")
            children[parent_name] = element

        chain: list[_Joint] = []
        link = "base_link"
        visited: set[str] = set()
        while link != "tool_link":
            if link in visited or link not in children:
                raise ValueError("URDF does not contain a serial base_link -> tool_link chain")
            visited.add(link)
            element = children[link]
            child = element.find("child")
            if child is None or "link" not in child.attrib:
                raise ValueError(f"URDF joint {element.attrib.get('name')!r} is missing its child link")
            origin = element.find("origin")
            axis = element.find("axis")
            kind = element.attrib.get("type", "fixed")
            chain.append(
                _Joint(
                    name=element.attrib.get("name", "unnamed"),
                    kind=kind,
                    parent=link,
                    child=child.attrib["link"],
                    origin_position=_vector_attribute(origin, "xyz", default="0 0 0"),
                    origin_rotation=_rpy_matrix(_vector_attribute(origin, "rpy", default="0 0 0")),
                    axis=_normalized_axis(axis, kind),
                )
            )
            link = child.attrib["link"]
        return tuple(chain)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="View a local FAFU LeRobot episode in WRS")
    parser.add_argument("--root", type=Path, required=True, help="Local LeRobot dataset root")
    parser.add_argument("--episode", type=int, default=0, help="Episode index")
    parser.add_argument(
        "--source",
        choices=("observation", "action"),
        default="observation",
        help="Use measured joint observations or joint action targets",
    )
    parser.add_argument("--wrs-path", type=Path, help="WRS checkout containing the wrs Python package")
    parser.add_argument("--urdf", type=Path, help="Custom FAFU URDF used for visualization")
    parser.add_argument("--stride", type=int, default=1, help="Render every Nth dataset frame")
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier")
    parser.add_argument("--max-frames", type=int, help="Limit playback after applying stride")
    parser.add_argument("--no-loop", action="store_true", help="Stop on the final frame")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and summarize the trajectory without opening a WRS window",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.stride <= 0:
        print("[FAIL] stride must be positive", file=sys.stderr)
        return 2
    if not math.isfinite(args.speed) or args.speed <= 0:
        print("[FAIL] speed must be a positive finite number", file=sys.stderr)
        return 2
    if args.max_frames is not None and args.max_frames <= 0:
        print("[FAIL] max-frames must be positive", file=sys.stderr)
        return 2

    try:
        episode = load_episode(args.root, args.episode)
        trajectory = episode.joint_trajectory(args.source)[:: args.stride]
        if args.max_frames is not None:
            trajectory = trajectory[: args.max_frames]
        if len(trajectory) == 0:
            raise DatasetReadError("selected trajectory contains no frames")
        skeleton = UrdfSkeleton(args.urdf)
        tool_positions = np.asarray([skeleton.frames(joints)[-1][0] for joints in trajectory])
    except (DatasetReadError, OSError, ValueError) as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 2

    print(
        f"[OK] episode {args.episode}: {len(trajectory)} rendered frame(s), "
        f"source={args.source}, fps={episode.info.fps:g}, stride={args.stride}"
    )
    print(
        "[INFO] TCP bounds (m): "
        f"min={np.array2string(tool_positions.min(axis=0), precision=4)}, "
        f"max={np.array2string(tool_positions.max(axis=0), precision=4)}"
    )
    if args.dry_run:
        return 0

    try:
        _open_wrs_window(
            skeleton,
            trajectory,
            tool_positions,
            fps=episode.info.fps,
            stride=args.stride,
            speed=args.speed,
            loop=not args.no_loop,
            wrs_path=args.wrs_path,
            episode_index=args.episode,
        )
    except (ImportError, RuntimeError) as exc:
        print(f"[FAIL] WRS viewer: {exc}", file=sys.stderr)
        return 3
    return 0


def _open_wrs_window(
    skeleton: UrdfSkeleton,
    trajectory: NDArray[np.float64],
    tool_positions: NDArray[np.float64],
    *,
    fps: float,
    stride: int,
    speed: float,
    loop: bool,
    wrs_path: Path | None,
    episode_index: int,
) -> None:
    checkout = wrs_path or (Path(os.environ["WRS_PATH"]) if os.environ.get("WRS_PATH") else None)
    if checkout is not None:
        checkout = checkout.expanduser().resolve()
        if not (checkout / "wrs").is_dir():
            raise RuntimeError(f"--wrs-path must contain the wrs package: {checkout}")
        if str(checkout) not in sys.path:
            sys.path.insert(0, str(checkout))

    try:
        import wrs.modeling.geometric_model as gm
        import wrs.visualization.panda.world as wd
    except ImportError as exc:
        raise ImportError(
            "could not import WRS; pass --wrs-path or set WRS_PATH to the WRS checkout"
        ) from exc

    base = wd.World(cam_pos=[1.1, 1.1, 0.8], lookat_pos=[0.0, 0.0, 0.2])
    gm.gen_frame(ax_length=0.10).attach_to(base)
    for start, end in zip(tool_positions[:-1], tool_positions[1:], strict=False):
        gm.gen_stick(
            spos=start,
            epos=end,
            radius=0.0012,
            rgb=np.array([0.35, 0.55, 0.95]),
            alpha=0.65,
        ).attach_to(base)

    current_nodes: list[Any] = []
    counter = 0
    label = base.show_text("", pos=(-1.25, 0.90), scale=0.04)

    def draw(index: int) -> None:
        nonlocal current_nodes
        for node in current_nodes:
            node.detach()
        current_nodes = []
        frames = skeleton.frames(trajectory[index])
        for (start, _), (end, _) in zip(frames[:-1], frames[1:], strict=True):
            node = gm.gen_stick(
                spos=start,
                epos=end,
                radius=0.008,
                rgb=np.array([0.18, 0.55, 0.30]),
            )
            node.attach_to(base)
            current_nodes.append(node)
        for position, _ in frames[:-1]:
            node = gm.gen_sphere(pos=position, radius=0.012, rgb=np.array([0.12, 0.28, 0.18]))
            node.attach_to(base)
            current_nodes.append(node)
        tcp_position, tcp_rotation = frames[-1]
        tcp_frame = gm.gen_frame(pos=tcp_position, rotmat=tcp_rotation, ax_length=0.06)
        tcp_frame.attach_to(base)
        current_nodes.append(tcp_frame)
        label.setText(f"FAFU dataset episode {episode_index} | frame {index + 1}/{len(trajectory)}")

    draw(0)
    interval = max(stride / fps / speed, 0.001)

    def update(task: Any) -> Any:
        nonlocal counter
        if counter + 1 >= len(trajectory):
            if not loop:
                return task.done
            counter = 0
        else:
            counter += 1
        draw(counter)
        return task.again

    base.taskMgr.doMethodLater(interval, update, "fafu_dataset_playback", appendTask=True)
    base.run()


def _vector_attribute(element: ET.Element | None, name: str, *, default: str) -> NDArray[np.float64]:
    raw = default if element is None else element.attrib.get(name, default)
    values = np.fromstring(raw, sep=" ", dtype=np.float64)
    if values.shape != (3,) or not np.all(np.isfinite(values)):
        raise ValueError(f"URDF attribute {name!r} must contain three finite numbers")
    return values


def _normalized_axis(element: ET.Element | None, kind: str) -> NDArray[np.float64]:
    if kind == "fixed":
        return np.zeros(3, dtype=np.float64)
    axis = _vector_attribute(element, "xyz", default="1 0 0")
    norm = float(np.linalg.norm(axis))
    if norm < 1e-12:
        raise ValueError("URDF joint axis must be non-zero")
    return axis / norm


def _rpy_matrix(rpy: NDArray[np.float64]) -> NDArray[np.float64]:
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rotation_x = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
    rotation_y = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
    rotation_z = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
    return rotation_z @ rotation_y @ rotation_x


if __name__ == "__main__":
    raise SystemExit(main())
