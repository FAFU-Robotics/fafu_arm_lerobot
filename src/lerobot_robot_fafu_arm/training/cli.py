"""Command line entry point for FAFU policy training."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from .act import ActTrainConfig, build_act_command, format_command
from .common import TRAINING_ACTION_MODES, check_training_dataset, format_training_report
from .config_file import load_act_yaml


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Preflight and launch LeRobot policies on FAFU datasets")
    subparsers = parser.add_subparsers(dest="algorithm", required=True)
    act = subparsers.add_parser("act", help="Check a dataset and prepare official LeRobot ACT training")
    act.add_argument("--config", type=Path, help="Versioned FAFU training YAML")
    act.add_argument("--dataset-root", type=Path)
    act.add_argument("--dataset-repo-id")
    act.add_argument("--action-mode", choices=TRAINING_ACTION_MODES)
    act.add_argument("--output-dir", type=Path)
    act.add_argument("--policy-type", help="LeRobot policy type; defaults to act")
    act.add_argument("--device", help="cuda, cuda:0, mps, or cpu; omit for LeRobot auto-selection")
    act.add_argument("--steps", type=int)
    act.add_argument("--batch-size", type=int)
    act.add_argument("--num-workers", type=int)
    act.add_argument("--seed", type=int)
    act.add_argument("--save-freq", type=int)
    act.add_argument("--log-freq", type=int)
    act.add_argument("--chunk-size", type=int)
    act.add_argument(
        "--n-action-steps",
        type=int,
        help="Actions executed before observing again; FAFU safety-oriented default: 10",
    )
    act.add_argument("--eval-split", type=float)
    act.add_argument("--eval-steps", type=int)
    act.add_argument("--temporal-ensemble-coeff", type=float)
    act.add_argument("--amp", action="store_true", default=None, help="Enable automatic mixed precision")
    act.add_argument(
        "--wandb", action="store_true", default=None, help="Explicitly enable external W&B logging"
    )
    act.add_argument(
        "--push-to-hub", action="store_true", default=None, help="Explicitly upload the trained policy"
    )
    act.add_argument("--policy-repo-id")
    act.add_argument(
        "--public",
        action="store_true",
        default=None,
        help="Make an uploaded policy public; uploads are private when this flag is absent",
    )
    act.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Forward an advanced LeRobot option, e.g. policy.dropout=0.2",
    )
    act.add_argument("--json", action="store_true", dest="as_json")
    act.add_argument(
        "--run",
        action="store_true",
        help="Start training after checks; without this flag only print the validated command",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        values = load_act_yaml(args.config) if args.config else {}
        cli_fields = {
            "dataset_repo_id": "dataset_repo_id",
            "dataset_root": "dataset_root",
            "output_dir": "output_dir",
            "action_mode": "action_mode",
            "policy_type": "policy_type",
            "device": "device",
            "steps": "steps",
            "batch_size": "batch_size",
            "num_workers": "num_workers",
            "seed": "seed",
            "save_freq": "save_freq",
            "log_freq": "log_freq",
            "chunk_size": "chunk_size",
            "n_action_steps": "n_action_steps",
            "eval_split": "eval_split",
            "eval_steps": "eval_steps",
            "temporal_ensemble_coeff": "temporal_ensemble_coeff",
            "amp": "use_amp",
            "wandb": "wandb",
            "push_to_hub": "push_to_hub",
            "public": "public",
            "policy_repo_id": "policy_repo_id",
        }
        for argument_name, config_name in cli_fields.items():
            value = getattr(args, argument_name)
            if value is not None:
                values[config_name] = value
        values["extra_args"] = tuple(values.get("extra_args", ())) + tuple(args.set)
        missing = [
            name
            for name in ("dataset_repo_id", "dataset_root", "output_dir", "action_mode")
            if name not in values
        ]
        if missing:
            flags = ", ".join(f"--{name.replace('_', '-')}" for name in missing)
            raise ValueError(f"missing training options: {flags}; provide them directly or with --config")
        config = ActTrainConfig(**values)
        command = build_act_command(config)
    except ValueError as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 2
    report = check_training_dataset(config.dataset_root, config.action_mode)

    if args.as_json:
        if args.run:
            print("[FAIL] --json cannot be combined with --run", file=sys.stderr)
            return 2
        result = report.to_dict()
        result["command"] = command
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(format_training_report(report))
        print(f"[COMMAND] {format_command(command)}")
    if not report.ok:
        return 2
    if not args.run:
        if not args.as_json:
            print("[DRY-RUN] checks passed; add --run to start training")
        return 0

    output_dir = config.output_dir.expanduser().resolve()
    if output_dir.exists():
        print(
            f"[FAIL] output directory already exists: {output_dir}; choose a new directory or use LeRobot resume",
            file=sys.stderr,
        )
        return 2
    executable = shutil.which(command[0])
    if executable is None:
        print("[FAIL] lerobot-train is not available in the active environment", file=sys.stderr)
        return 2
    command[0] = executable
    try:
        return subprocess.run(command, check=False).returncode
    except OSError as exc:
        print(f"[FAIL] could not start LeRobot training: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
