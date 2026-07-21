"""Read and export local LeRobot datasets without connecting to a robot."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .representation import JOINT_NAMES


class DatasetReadError(RuntimeError):
    """Raised when a local LeRobot dataset cannot be read safely."""


@dataclass(frozen=True)
class LocalDatasetInfo:
    """Canonical metadata loaded from ``meta/info.json``."""

    root: Path
    raw: dict[str, Any]

    @property
    def features(self) -> dict[str, dict[str, Any]]:
        features = self.raw.get("features")
        return features if isinstance(features, dict) else {}

    @property
    def fps(self) -> float:
        value = self.raw.get("fps")
        if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
            raise DatasetReadError("metadata fps is missing or invalid")
        return float(value)

    @property
    def total_episodes(self) -> int:
        value = self.raw.get("total_episodes")
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise DatasetReadError("metadata total_episodes is missing or invalid")
        return value

    def feature_names(self, key: str) -> tuple[str, ...]:
        feature = self.features.get(key)
        names = feature.get("names") if isinstance(feature, dict) else None
        if not isinstance(names, list) or not all(isinstance(name, str) for name in names):
            return ()
        return tuple(names)


@dataclass(frozen=True)
class EpisodeData:
    """Low-dimensional columns belonging to one local dataset episode."""

    info: LocalDatasetInfo
    episode_index: int
    columns: dict[str, NDArray[Any]]

    def __len__(self) -> int:
        if not self.columns:
            return 0
        return len(next(iter(self.columns.values())))

    def column(self, key: str) -> NDArray[Any]:
        try:
            return self.columns[key]
        except KeyError as exc:
            available = ", ".join(self.columns)
            raise DatasetReadError(f"column {key!r} is unavailable; loaded columns: {available}") from exc

    def feature_matrix(self, key: str) -> NDArray[np.float64]:
        values = np.asarray(self.column(key), dtype=np.float64)
        if values.ndim != 2:
            raise DatasetReadError(f"column {key!r} must be a 2-D feature matrix, got shape {values.shape}")
        if not np.all(np.isfinite(values)):
            raise DatasetReadError(f"column {key!r} contains NaN or infinity")
        return values

    def joint_trajectory(self, source: str = "observation") -> NDArray[np.float64]:
        """Return six FAFU joint positions from observation state or joint actions."""

        if source == "observation":
            key = "observation.state"
        elif source == "action":
            key = "action"
        else:
            raise ValueError("source must be 'observation' or 'action'")

        names = self.info.feature_names(key)
        if not names:
            raise DatasetReadError(f"metadata does not declare semantic names for {key!r}")
        expected = tuple(f"{name}.pos" for name in JOINT_NAMES)
        missing = [name for name in expected if name not in names]
        if missing:
            raise DatasetReadError(
                f"{key!r} does not contain a joint trajectory; missing fields: {', '.join(missing)}"
            )
        matrix = self.feature_matrix(key)
        if matrix.shape[1] != len(names):
            raise DatasetReadError(
                f"metadata declares {len(names)} names for {key!r}, but data width is {matrix.shape[1]}"
            )
        indices = [names.index(name) for name in expected]
        return matrix[:, indices]

    def flattened_columns(self) -> dict[str, NDArray[Any]]:
        """Flatten scalar and vector columns into CSV-friendly one-dimensional columns."""

        flattened: dict[str, NDArray[Any]] = {}
        for key, raw_values in self.columns.items():
            values = np.asarray(raw_values)
            if values.ndim == 1:
                flattened[key] = values
                continue
            if values.ndim != 2:
                continue
            names = self.info.feature_names(key)
            if names and len(names) != values.shape[1]:
                raise DatasetReadError(
                    f"metadata declares {len(names)} names for {key!r}, but data width is {values.shape[1]}"
                )
            labels = names or tuple(str(index) for index in range(values.shape[1]))
            for index, name in enumerate(labels):
                flattened[f"{key}.{name}"] = values[:, index]
        return flattened

    def records(self, limit: int | None = None) -> list[dict[str, Any]]:
        """Return flattened rows suitable for JSON output or notebook inspection."""

        columns = self.flattened_columns()
        count = len(self) if limit is None else min(max(limit, 0), len(self))
        return [
            {name: _json_value(values[index]) for name, values in columns.items()} for index in range(count)
        ]


def load_dataset_info(root: str | Path) -> LocalDatasetInfo:
    """Load and minimally validate a local LeRobot ``meta/info.json`` file."""

    dataset_root = Path(root).expanduser().resolve()
    info_path = dataset_root / "meta" / "info.json"
    try:
        raw = json.loads(info_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise DatasetReadError(f"metadata file not found: {info_path}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise DatasetReadError(f"could not read dataset metadata {info_path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise DatasetReadError(f"dataset metadata must be a JSON object: {info_path}")
    info = LocalDatasetInfo(dataset_root, raw)
    _ = info.fps
    _ = info.total_episodes
    if not info.features:
        raise DatasetReadError("metadata features are missing or invalid")
    return info


def load_episode(
    root: str | Path,
    episode_index: int,
    *,
    columns: list[str] | tuple[str, ...] | None = None,
) -> EpisodeData:
    """Load one episode's tabular data from LeRobot v2.1 or v3 Parquet shards."""

    info = load_dataset_info(root)
    if episode_index < 0 or episode_index >= info.total_episodes:
        raise DatasetReadError(
            f"episode {episode_index} is out of range; dataset contains {info.total_episodes} episode(s)"
        )

    try:
        import pyarrow as pa
        import pyarrow.dataset as pads
    except ModuleNotFoundError as exc:
        raise DatasetReadError(
            'reading frame data requires PyArrow; install LeRobot dataset support with "pip install lerobot[dataset]"'
        ) from exc

    data_dir = info.root / "data"
    parquet_files = sorted(data_dir.rglob("*.parquet")) if data_dir.is_dir() else []
    if not parquet_files:
        raise DatasetReadError(f"no Parquet frame files found under: {data_dir}")

    try:
        parquet = pads.dataset([str(path) for path in parquet_files], format="parquet")
    except Exception as exc:
        raise DatasetReadError(f"could not open Parquet frame files: {exc}") from exc

    available = tuple(parquet.schema.names)
    if "episode_index" not in available:
        raise DatasetReadError("Parquet frame data does not contain episode_index")
    selected = _selected_columns(info, available, columns)
    try:
        table = parquet.to_table(
            columns=selected,
            filter=pads.field("episode_index") == episode_index,
        )
    except Exception as exc:
        raise DatasetReadError(f"could not read episode {episode_index}: {exc}") from exc
    if table.num_rows == 0:
        raise DatasetReadError(f"episode {episode_index} has no frame rows")

    sort_key = next((key for key in ("frame_index", "index", "timestamp") if key in table.column_names), None)
    if sort_key is not None:
        order = np.argsort(np.asarray(table[sort_key].to_pylist()), kind="stable")
        table = table.take(pa.array(order))

    loaded = {name: _column_to_numpy(table[name]) for name in table.column_names}
    return EpisodeData(info=info, episode_index=episode_index, columns=loaded)


def export_episode_csv(
    episode: EpisodeData,
    output: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Export loaded low-dimensional episode columns to a flat UTF-8 CSV file."""

    output_path = Path(output).expanduser().resolve()
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite existing file: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    columns = episode.flattened_columns()
    with output_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        for index in range(len(episode)):
            writer.writerow(_csv_value(values[index]) for values in columns.values())
    return output_path


def dataset_summary(info: LocalDatasetInfo) -> dict[str, Any]:
    """Return a JSON-serializable high-level dataset summary."""

    camera_features = [
        key
        for key, feature in info.features.items()
        if isinstance(feature, dict) and feature.get("dtype") in {"image", "video"}
    ]
    return {
        "root": str(info.root),
        "robot_type": info.raw.get("robot_type"),
        "codebase_version": info.raw.get("codebase_version"),
        "fps": info.fps,
        "total_episodes": info.total_episodes,
        "total_frames": info.raw.get("total_frames"),
        "features": list(info.features),
        "camera_features": camera_features,
    }


def _selected_columns(
    info: LocalDatasetInfo,
    available: tuple[str, ...],
    requested: list[str] | tuple[str, ...] | None,
) -> list[str]:
    if requested is not None:
        missing = [name for name in requested if name not in available]
        if missing:
            raise DatasetReadError(f"requested columns are unavailable: {', '.join(missing)}")
        selected = list(requested)
    else:
        metadata_columns = ["timestamp", "frame_index", "episode_index", "index", "task_index"]
        data_columns = []
        for key, feature in info.features.items():
            dtype = feature.get("dtype") if isinstance(feature, dict) else None
            if dtype not in {"image", "video"} and key in available:
                data_columns.append(key)
        selected = [name for name in (*metadata_columns, *data_columns) if name in available]
    if "episode_index" not in selected:
        selected.append("episode_index")
    return list(dict.fromkeys(selected))


def _column_to_numpy(column: Any) -> NDArray[Any]:
    values = column.to_pylist()
    try:
        return np.asarray(values)
    except ValueError:
        return np.asarray(values, dtype=object)


def _json_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def _csv_value(value: Any) -> Any:
    value = _json_value(value)
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return value
