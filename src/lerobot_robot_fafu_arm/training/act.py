"""ACT-specific launch configuration using LeRobot's official implementation."""

from __future__ import annotations

import os
import shlex
import subprocess
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from .common import TRAINING_ACTION_MODES

_PROTECTED_OVERRIDES = {
    "dataset.repo_id",
    "dataset.root",
    "job.target",
    "output_dir",
    "policy.type",
    "policy.push_to_hub",
    "policy.private",
    "policy.repo_id",
    "save_checkpoint_to_hub",
    "wandb.enable",
}


@dataclass(frozen=True)
class ActTrainConfig:
    """Reproducible subset of LeRobot ACT training options used by FAFU Arm."""

    dataset_repo_id: str
    dataset_root: Path
    output_dir: Path
    action_mode: str
    device: str | None = None
    steps: int = 100_000
    batch_size: int = 8
    num_workers: int = 4
    seed: int = 1000
    save_freq: int = 20_000
    log_freq: int = 200
    chunk_size: int = 100
    n_action_steps: int = 10
    eval_split: float = 0.0
    eval_steps: int = 0
    temporal_ensemble_coeff: float | None = None
    use_amp: bool = False
    wandb: bool = False
    push_to_hub: bool = False
    public: bool = False
    policy_repo_id: str | None = None
    extra_args: tuple[str, ...] = ()

    def validate(self) -> None:
        if self.action_mode not in TRAINING_ACTION_MODES:
            raise ValueError("action_mode must be joint, ee_delta, or ee_pose")
        if not self.dataset_repo_id.strip():
            raise ValueError("dataset_repo_id must not be empty")
        for name in ("steps", "batch_size", "save_freq", "log_freq", "chunk_size", "n_action_steps"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.num_workers < 0:
            raise ValueError("num_workers must be non-negative")
        if self.n_action_steps > self.chunk_size:
            raise ValueError("n_action_steps cannot exceed chunk_size")
        if not 0.0 <= self.eval_split < 1.0:
            raise ValueError("eval_split must be in [0, 1)")
        if self.eval_steps < 0:
            raise ValueError("eval_steps must be non-negative")
        if self.eval_steps > 0 and self.eval_split == 0.0:
            raise ValueError("eval_steps > 0 requires eval_split > 0")
        if self.temporal_ensemble_coeff is not None:
            if self.temporal_ensemble_coeff <= 0:
                raise ValueError("temporal_ensemble_coeff must be positive")
            if self.n_action_steps != 1:
                raise ValueError("temporal ensembling requires n_action_steps=1")
        if self.push_to_hub and not self.policy_repo_id:
            raise ValueError("push_to_hub requires policy_repo_id")
        if self.public and not self.push_to_hub:
            raise ValueError("public is only valid together with push_to_hub")
        for override in self.extra_args:
            key, separator, _ = override.partition("=")
            if not separator or not key:
                raise ValueError(f"extra override must use KEY=VALUE syntax: {override!r}")
            if key.removeprefix("--") in _PROTECTED_OVERRIDES:
                raise ValueError(f"use the dedicated option instead of overriding {key!r}")


def build_act_command(config: ActTrainConfig, executable: str = "lerobot-train") -> list[str]:
    """Build a shell-free command for LeRobot's official ACT trainer."""

    config.validate()
    command = [
        executable,
        f"--dataset.repo_id={config.dataset_repo_id}",
        f"--dataset.root={config.dataset_root.expanduser().resolve()}",
        "--policy.type=act",
        f"--output_dir={config.output_dir.expanduser().resolve()}",
        f"--job_name=act_fafu_{config.action_mode}",
        f"--steps={config.steps}",
        f"--batch_size={config.batch_size}",
        f"--num_workers={config.num_workers}",
        f"--seed={config.seed}",
        f"--save_freq={config.save_freq}",
        f"--log_freq={config.log_freq}",
        f"--policy.chunk_size={config.chunk_size}",
        f"--policy.n_action_steps={config.n_action_steps}",
        f"--policy.use_amp={str(config.use_amp).lower()}",
        f"--wandb.enable={str(config.wandb).lower()}",
        f"--policy.push_to_hub={str(config.push_to_hub).lower()}",
    ]
    if _lerobot_at_least_0_6():
        command.append("--save_checkpoint_to_hub=false")
    if config.device:
        command.append(f"--policy.device={config.device}")
    if config.eval_split > 0.0:
        command.extend(
            (
                f"--dataset.eval_split={config.eval_split}",
                f"--eval_steps={config.eval_steps}",
            )
        )
    if config.temporal_ensemble_coeff is not None:
        command.append(f"--policy.temporal_ensemble_coeff={config.temporal_ensemble_coeff}")
    if config.push_to_hub:
        command.extend(
            (
                f"--policy.repo_id={config.policy_repo_id}",
                f"--policy.private={str(not config.public).lower()}",
            )
        )
    command.extend(
        override if override.startswith("--") else f"--{override}" for override in config.extra_args
    )
    return command


def format_command(command: list[str]) -> str:
    """Render a command for the current operating system."""

    return subprocess.list2cmdline(command) if os.name == "nt" else shlex.join(command)


def _lerobot_at_least_0_6() -> bool:
    try:
        installed = version("lerobot")
    except PackageNotFoundError:
        return False
    parts = installed.split(".", maxsplit=2)
    try:
        return tuple(int(part.split("+", maxsplit=1)[0]) for part in parts[:2]) >= (0, 6)
    except ValueError:
        return False
