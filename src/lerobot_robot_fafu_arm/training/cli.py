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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Preflight and launch LeRobot policies on FAFU datasets")
    subparsers = parser.add_subparsers(dest="algorithm", required=True)
    act = subparsers.add_parser("act", help="Check a dataset and prepare official LeRobot ACT training")
    act.add_argument("--dataset-root", type=Path, required=True)
    act.add_argument("--dataset-repo-id", required=True)
    act.add_argument("--action-mode", choices=TRAINING_ACTION_MODES, required=True)
    act.add_argument("--output-dir", type=Path, required=True)
    act.add_argument("--device", help="cuda, cuda:0, mps, or cpu; omit for LeRobot auto-selection")
    act.add_argument("--steps", type=int, default=100_000)
    act.add_argument("--batch-size", type=int, default=8)
    act.add_argument("--num-workers", type=int, default=4)
    act.add_argument("--seed", type=int, default=1000)
    act.add_argument("--save-freq", type=int, default=20_000)
    act.add_argument("--log-freq", type=int, default=200)
    act.add_argument("--chunk-size", type=int, default=100)
    act.add_argument(
        "--n-action-steps",
        type=int,
        default=10,
        help="Actions executed before observing again; FAFU safety-oriented default: 10",
    )
    act.add_argument("--eval-split", type=float, default=0.0)
    act.add_argument("--eval-steps", type=int, default=0)
    act.add_argument("--temporal-ensemble-coeff", type=float)
    act.add_argument("--amp", action="store_true", help="Enable automatic mixed precision")
    act.add_argument("--wandb", action="store_true", help="Explicitly enable external W&B logging")
    act.add_argument("--push-to-hub", action="store_true", help="Explicitly upload the trained policy")
    act.add_argument("--policy-repo-id")
    act.add_argument(
        "--public",
        action="store_true",
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
    report = check_training_dataset(args.dataset_root, args.action_mode)
    try:
        config = ActTrainConfig(
            dataset_repo_id=args.dataset_repo_id,
            dataset_root=args.dataset_root,
            output_dir=args.output_dir,
            action_mode=args.action_mode,
            device=args.device,
            steps=args.steps,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            seed=args.seed,
            save_freq=args.save_freq,
            log_freq=args.log_freq,
            chunk_size=args.chunk_size,
            n_action_steps=args.n_action_steps,
            eval_split=args.eval_split,
            eval_steps=args.eval_steps,
            temporal_ensemble_coeff=args.temporal_ensemble_coeff,
            use_amp=args.amp,
            wandb=args.wandb,
            push_to_hub=args.push_to_hub,
            public=args.public,
            policy_repo_id=args.policy_repo_id,
            extra_args=tuple(args.set),
        )
        command = build_act_command(config)
    except ValueError as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 2

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
