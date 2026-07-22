"""Strict YAML configuration loader for reproducible FAFU training runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class TrainingConfigError(ValueError):
    """Raised when a training YAML file is missing fields or has unsafe values."""


_TOP_LEVEL_KEYS = {
    "schema_version",
    "algorithm",
    "dataset",
    "run",
    "policy",
    "evaluation",
    "tracking",
    "hub",
}

_GROUP_FIELDS: dict[str, dict[str, tuple[str, type | tuple[type, ...]]]] = {
    "dataset": {
        "repo_id": ("dataset_repo_id", str),
        "urdf_path": ("urdf_path", (str, type(None))),
        "root": ("dataset_root", str),
        "action_mode": ("action_mode", str),
    },
    "run": {
        "output_dir": ("output_dir", str),
        "device": ("device", (str, type(None))),
        "steps": ("steps", int),
        "batch_size": ("batch_size", int),
        "num_workers": ("num_workers", int),
        "seed": ("seed", int),
        "save_freq": ("save_freq", int),
        "log_freq": ("log_freq", int),
    },
    "evaluation": {
        "eval_split": ("eval_split", (int, float)),
        "eval_steps": ("eval_steps", int),
    },
    "tracking": {
        "wandb": ("wandb", bool),
    },
    "hub": {
        "push_to_hub": ("push_to_hub", bool),
        "policy_repo_id": ("policy_repo_id", (str, type(None))),
        "public": ("public", bool),
    },
}

_POLICY_DIRECT_FIELDS: dict[str, tuple[str, type | tuple[type, ...]]] = {
    "type": ("policy_type", str),
    "chunk_size": ("chunk_size", int),
    "n_action_steps": ("n_action_steps", int),
    "temporal_ensemble_coeff": ("temporal_ensemble_coeff", (int, float, type(None))),
}

_POLICY_TUNING_FIELDS: dict[str, type | tuple[type, ...]] = {
    "vision_backbone": str,
    "pretrained_backbone_weights": (str, type(None)),
    "replace_final_stride_with_dilation": bool,
    "pre_norm": bool,
    "dim_model": int,
    "n_heads": int,
    "dim_feedforward": int,
    "feedforward_activation": str,
    "n_encoder_layers": int,
    "n_decoder_layers": int,
    "use_vae": bool,
    "latent_dim": int,
    "n_vae_encoder_layers": int,
    "dropout": (int, float),
    "kl_weight": (int, float),
    "optimizer_lr": (int, float),
    "optimizer_weight_decay": (int, float),
    "optimizer_lr_backbone": (int, float),
}


def load_act_yaml(path: str | Path) -> dict[str, Any]:
    """Load a versioned ACT YAML file into ``ActTrainConfig`` keyword arguments."""

    config_path = Path(path).expanduser().resolve()
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise TrainingConfigError(f"training config not found: {config_path}") from exc
    except (OSError, yaml.YAMLError) as exc:
        raise TrainingConfigError(f"could not read training config {config_path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise TrainingConfigError("training config must be a YAML mapping")

    unknown = sorted(set(raw) - _TOP_LEVEL_KEYS)
    if unknown:
        raise TrainingConfigError(f"unknown top-level training keys: {', '.join(unknown)}")
    if raw.get("schema_version") != 1:
        raise TrainingConfigError("schema_version must be 1")
    if raw.get("algorithm") != "act":
        raise TrainingConfigError("algorithm must be 'act'")

    values: dict[str, Any] = {}
    for group_name, fields in _GROUP_FIELDS.items():
        group = _mapping(raw.get(group_name, {}), group_name)
        _reject_unknown(group, set(fields), group_name)
        for yaml_name, (target_name, expected_type) in fields.items():
            if yaml_name in group:
                value = group[yaml_name]
                _check_type(value, expected_type, f"{group_name}.{yaml_name}")
                values[target_name] = value

    policy = _mapping(raw.get("policy", {}), "policy")
    allowed_policy = set(_POLICY_DIRECT_FIELDS) | set(_POLICY_TUNING_FIELDS) | {"extra"}
    _reject_unknown(policy, allowed_policy, "policy")
    for yaml_name, (target_name, expected_type) in _POLICY_DIRECT_FIELDS.items():
        if yaml_name in policy:
            value = policy[yaml_name]
            _check_type(value, expected_type, f"policy.{yaml_name}")
            values[target_name] = value

    extra_args = []
    for name, expected_type in sorted(_POLICY_TUNING_FIELDS.items()):
        if name not in policy:
            continue
        value = policy[name]
        _check_type(value, expected_type, f"policy.{name}")
        extra_args.append(f"policy.{name}={_format_scalar(value, f'policy.{name}')}")
    policy_extra = _mapping(policy.get("extra", {}), "policy.extra")
    for name, value in sorted(policy_extra.items()):
        if not isinstance(name, str) or not name or "." in name:
            raise TrainingConfigError(f"invalid custom policy option name: {name!r}")
        extra_args.append(f"policy.{name}={_format_scalar(value, f'policy.extra.{name}')}")

    for key in ("dataset_root", "output_dir", "urdf_path"):
        if key in values and values[key] is not None:
            values[key] = Path(values[key])
    if "eval_split" in values:
        values["eval_split"] = float(values["eval_split"])
    if "temporal_ensemble_coeff" in values and values["temporal_ensemble_coeff"] is not None:
        values["temporal_ensemble_coeff"] = float(values["temporal_ensemble_coeff"])
    values["extra_args"] = tuple(extra_args)
    return values


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TrainingConfigError(f"{name} must be a YAML mapping")
    if not all(isinstance(key, str) for key in value):
        raise TrainingConfigError(f"{name} contains a non-string key")
    return value


def _reject_unknown(group: dict[str, Any], allowed: set[str], name: str) -> None:
    unknown = sorted(set(group) - allowed)
    if unknown:
        raise TrainingConfigError(f"unknown {name} keys: {', '.join(unknown)}")


def _check_type(value: Any, expected: type | tuple[type, ...], name: str) -> None:
    if isinstance(value, bool) and expected is not bool:
        raise TrainingConfigError(f"{name} has invalid boolean value")
    if not isinstance(value, expected):
        raise TrainingConfigError(f"{name} has invalid type {type(value).__name__}")


def _format_scalar(value: Any, name: str) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (str, int, float)) and not isinstance(value, bool):
        return str(value)
    raise TrainingConfigError(f"{name} must be a scalar YAML value")
