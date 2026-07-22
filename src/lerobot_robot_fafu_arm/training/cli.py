"""Command line entry point for FAFU policy training."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi
from huggingface_hub.errors import HfHubHTTPError, RepositoryNotFoundError

from ..inference.manifest import (
    MANIFEST_FILENAME,
    InferenceManifest,
    InferenceManifestError,
    build_manifest_from_dataset,
    verify_checkpoint_integrity,
    write_training_manifests,
)
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
    act.add_argument("--urdf-path", type=Path, help="URDF used while collecting this dataset")
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
        "--sync-manifest",
        action="store_true",
        help="Validate and synchronize manifests into an existing output after official resume",
    )

    act.add_argument(
        "--run",
        action="store_true",
        help="Start training after checks; without this flag only print the validated command",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.sync_manifest and (args.run or args.as_json):
        print("[FAIL] --sync-manifest cannot be combined with --run or --json", file=sys.stderr)
        return 2

    try:
        values = load_act_yaml(args.config) if args.config else {}
        cli_fields = {
            "dataset_repo_id": "dataset_repo_id",
            "dataset_root": "dataset_root",
            "output_dir": "output_dir",
            "action_mode": "action_mode",
            "urdf_path": "urdf_path",
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
    try:
        manifest = build_manifest_from_dataset(
            config.dataset_root, config.action_mode, urdf_path=config.urdf_path
        )
    except InferenceManifestError as exc:
        if report.ok:
            print(f"[FAIL] inference manifest: {exc}", file=sys.stderr)
            return 2

    upload_to_hub = config.push_to_hub if not args.sync_manifest else bool(args.push_to_hub)
    if report.ok and upload_to_hub:
        try:
            _validate_hub_visibility(config.policy_repo_id, public=config.public)
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
        if args.sync_manifest:
            print(f"[SYNC] output: {config.output_dir.expanduser().resolve()}")
        else:
            print(f"[COMMAND] {format_command(command)}")
    if not report.ok:
        return 2
    if args.sync_manifest:
        return _finalize_manifests(
            config,
            manifest,
            upload_to_hub=upload_to_hub,
            refresh_checkpoint_bindings=True,
        )
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
        return_code = subprocess.run(command, check=False).returncode
    except OSError as exc:
        print(f"[FAIL] could not start LeRobot training: {exc}", file=sys.stderr)
        return 2
    if return_code != 0:
        return return_code

    return _finalize_manifests(config, manifest, upload_to_hub=config.push_to_hub)


def _finalize_manifests(
    config: ActTrainConfig,
    manifest: InferenceManifest,
    *,
    upload_to_hub: bool,
    refresh_checkpoint_bindings: bool = False,
) -> int:
    output_dir = config.output_dir.expanduser().resolve()
    try:
        manifest_paths = write_training_manifests(
            output_dir,
            manifest=manifest,
            refresh_checkpoint_bindings=refresh_checkpoint_bindings,
        )
    except InferenceManifestError as exc:
        print(f"[FAIL] inference manifest could not be written: {exc}", file=sys.stderr)
        return 2
    for path in manifest_paths:
        print(f"[MANIFEST] {path}")
    if upload_to_hub:
        if config.policy_repo_id is None:
            print("[FAIL] push_to_hub requires policy_repo_id", file=sys.stderr)
            return 2
        try:
            bound_manifest = _select_bound_checkpoint_manifest(output_dir)
            _upload_bound_checkpoint_to_hub(bound_manifest, config.policy_repo_id)
        except InferenceManifestError as exc:
            print(f"[FAIL] {exc}", file=sys.stderr)
            return 2
        print(f"[HUB] uploaded the bound ACT checkpoint and {MANIFEST_FILENAME} to {config.policy_repo_id}")
    return 0


def _select_bound_checkpoint_manifest(output_dir: Path) -> Path:
    """Select and verify the final checkpoint-owned manifest."""

    checkpoint_root = output_dir / "checkpoints"
    preferred = checkpoint_root / "last" / "pretrained_model" / MANIFEST_FILENAME
    if preferred.is_file():
        _read_bound_checkpoint_manifest(preferred)
        return preferred.resolve()

    candidates = [
        path for path in checkpoint_root.glob("*/pretrained_model/" + MANIFEST_FILENAME) if path.is_file()
    ]
    if not candidates:
        raise InferenceManifestError("no completed pretrained_model checkpoint is available for Hub upload")

    def checkpoint_order(path: Path) -> tuple[int, int, str]:
        name = path.parent.parent.name
        if name.isdigit():
            return (1, int(name), name)
        return (0, 0, name)

    selected = max(candidates, key=checkpoint_order)
    _read_bound_checkpoint_manifest(selected)
    return selected.resolve()


def _read_bound_checkpoint_manifest(manifest_path: Path) -> InferenceManifest:
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise InferenceManifestError(f"could not read checkpoint manifest {manifest_path}: {exc}") from exc
    manifest = InferenceManifest.from_dict(raw)
    if manifest.checkpoint_files is None:
        raise InferenceManifestError(
            f"checkpoint manifest is not bound to model and processor files: {manifest_path}"
        )
    verify_checkpoint_integrity(manifest_path.parent, manifest)
    return manifest


def _upload_bound_checkpoint_to_hub(manifest_path: Path, repo_id: str) -> None:
    """Atomically upload the files covered by the checkpoint manifest."""

    manifest = _read_bound_checkpoint_manifest(manifest_path)
    assert manifest.checkpoint_files is not None
    try:
        operations = [
            CommitOperationAdd(
                path_in_repo=name,
                path_or_fileobj=manifest_path.parent / name,
            )
            for name in manifest.checkpoint_files
        ]
        operations.append(
            CommitOperationAdd(
                path_in_repo=MANIFEST_FILENAME,
                path_or_fileobj=manifest_path,
            )
        )
        HfApi().create_commit(
            repo_id=repo_id,
            repo_type="model",
            operations=operations,
            commit_message="Upload bound FAFU ACT checkpoint",
        )
    except Exception as exc:
        raise InferenceManifestError(
            f"training succeeded, but the bound checkpoint could not be uploaded to {repo_id}: {exc}"
        ) from exc


def _validate_hub_visibility(repo_id: str | None, *, public: bool) -> None:
    """Refuse an existing Hub repository whose visibility differs from the request."""

    if repo_id is None:
        raise ValueError("push_to_hub requires policy_repo_id")
    try:
        info = HfApi().repo_info(repo_id=repo_id, repo_type="model")
    except RepositoryNotFoundError:
        return
    except HfHubHTTPError as exc:
        raise ValueError(f"could not verify Hugging Face repository visibility for {repo_id}: {exc}") from exc

    private = getattr(info, "private", None)
    if not isinstance(private, bool):
        raise ValueError(f"Hugging Face did not return repository visibility for {repo_id}")
    requested_private = not public
    if private != requested_private:
        actual = "private" if private else "public"
        requested = "public" if public else "private"
        raise ValueError(
            f"existing model repository {repo_id} is {actual}, but this run requests {requested}; "
            "use a repository with matching visibility instead of relying on create_repo to change it"
        )


if __name__ == "__main__":
    raise SystemExit(main())
