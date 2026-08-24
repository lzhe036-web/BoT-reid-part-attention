#!/usr/bin/env python
# encoding: utf-8
"""Unified formal/smoke entry point: verify, train, finalize, archive."""

from __future__ import absolute_import

import argparse
import codecs
import contextlib
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import cfg
from data.datasets import init_dataset
from utils.experiment_recording import (
    NOT_RECORDED,
    SCHEMA_VERSION,
    atomic_write_json,
    build_dataset_manifest,
    collect_environment,
    config_protocol_signature,
    experiment_identity,
    finalize_run,
    generate_run_id,
    initialize_run,
    git_implementation_signature,
    record_console_log_evidence,
    record_run_failure,
    record_training_exit,
    sha256_bytes,
    sha256_file,
    validate_git_preflight,
    validate_parent_lineage,
)
from utils.multigranular_signature import (
    FEATURE_REFERENCE_CONFIG,
    FIXED_HARD_FEATURE_REFERENCE_COMMIT,
    build_feature_compatibility_evidence,
    canonical_multigranular_feature_signature,
    git_show_source,
    require_feature_compatibility,
)
from utils.reproducibility import (
    RUNNER_SEED_ENV,
    validate_seed,
    validate_seed_evidence_chain,
    resolved_config_text,
)
from utils.smoke_gate import validate_formal_smoke_gate


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run one recorded Re-ID experiment")
    parser.add_argument("--config", required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--experiment-family", required=True)
    parser.add_argument(
        "--run-kind", choices=("formal", "smoke"), default="formal"
    )
    parser.add_argument("--expected-branch", required=True)
    parser.add_argument("--expected-commit")
    parser.add_argument("--run-id")
    parser.add_argument("--notes", default="")
    parser.add_argument("--method")
    parser.add_argument("--baseline-method", default=NOT_RECORDED)
    parser.add_argument("--baseline-commit", default=NOT_RECORDED)
    parser.add_argument("--parent-branch", default=NOT_RECORDED)
    parser.add_argument("--parent-commit", default=NOT_RECORDED)
    parser.add_argument("--feature-reference-commit", default=NOT_RECORDED)
    parser.add_argument(
        "--required-smoke-experiment-id",
        help="Require matching successful one-epoch smoke evidence before formal",
    )
    parser.add_argument(
        "--feature-reference-config", default=FEATURE_REFERENCE_CONFIG
    )
    parser.add_argument(
        "--reference-config",
        help="Resolved-config comparison reference used by strict protocols",
    )
    parser.add_argument(
        "--expected-config-difference", action="append", default=[],
        help="Exact dotted leaf expected to differ from --reference-config",
    )
    parser.add_argument(
        "--records-root",
        default=str(REPO_ROOT / "experiment_records"),
    )
    parser.add_argument(
        "opts",
        help="YACS config overrides, used by isolated smoke runs",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser.parse_args(argv)


def _load_config(config_path, opts=None):
    local_cfg = cfg.clone()
    local_cfg.merge_from_file(str(config_path))
    local_cfg.merge_from_list(opts or [])
    local_cfg.freeze()
    return local_cfg


def _effective_run_opts(run_kind, source_output_dir, opts=None):
    """Derive the standard one-epoch smoke without shell hyperparameter overrides."""
    overrides = list(opts or [])
    if run_kind != "smoke" or overrides:
        return overrides
    return [
        "SOLVER.MAX_EPOCHS", "1",
        "SOLVER.CHECKPOINT_PERIOD", "1",
        "SOLVER.EVAL_PERIOD", "1",
        "OUTPUT_DIR", "{}_smoke".format(str(source_output_dir).rstrip("/")),
    ]


def _validate_run_overrides(run_kind, opts):
    overrides = list(opts or [])
    if run_kind == "formal" and overrides:
        raise RuntimeError("Formal runs forbid command-line config overrides")
    if len(overrides) % 2:
        raise RuntimeError("Config overrides must contain key/value pairs")
    allowed_smoke = {
        "SOLVER.MAX_EPOCHS",
        "SOLVER.CHECKPOINT_PERIOD",
        "SOLVER.EVAL_PERIOD",
        "OUTPUT_DIR",
    }
    keys = set(overrides[::2])
    unexpected = sorted(keys - allowed_smoke)
    if run_kind == "smoke" and unexpected:
        raise RuntimeError(
            "Smoke overrides contain non-isolated fields: {}".format(
                unexpected
            )
        )


def _load_explicit_source_seed(config_path):
    import yaml

    with Path(config_path).open("r", encoding="utf-8") as handle:
        source = yaml.safe_load(handle) or {}
    if "SEED" not in source:
        raise RuntimeError("Formal config must explicitly declare SEED")
    return validate_seed(source["SEED"])


def _load_source_output_dir(config_path):
    import yaml

    with Path(config_path).open("r", encoding="utf-8") as handle:
        source = yaml.safe_load(handle) or {}
    return source.get("OUTPUT_DIR", NOT_RECORDED)


def _load_revision_config(repo_root, revision, repo_relative_path):
    import yaml

    source = git_show_source(repo_root, revision, repo_relative_path)
    configuration = yaml.safe_load(source)
    if not isinstance(configuration, dict):
        raise RuntimeError(
            "Feature reference config is not a YAML mapping"
        )
    return configuration


def _validate_resolved_run_config(configuration, run_kind,
                                  source_output_dir):
    """Validate effective smoke isolation after every override is applied."""
    solver = configuration.get("SOLVER", {})
    output_dir = configuration.get("OUTPUT_DIR", NOT_RECORDED)
    if output_dir in (None, "", NOT_RECORDED):
        raise RuntimeError("Resolved OUTPUT_DIR is missing")
    if run_kind == "smoke":
        expected = {
            "MAX_EPOCHS": 1,
            "CHECKPOINT_PERIOD": 1,
            "EVAL_PERIOD": 1,
        }
        conflicts = {
            key: solver.get(key, NOT_RECORDED)
            for key, value in expected.items()
            if solver.get(key, NOT_RECORDED) != value
        }
        if conflicts:
            raise RuntimeError(
                "Resolved smoke config is not isolated: {}".format(conflicts)
            )
        if str(output_dir) == str(source_output_dir):
            raise RuntimeError(
                "Smoke OUTPUT_DIR must differ from the formal OUTPUT_DIR"
            )
    return configuration


def _build_training_environment(resolved_seed, base_environment=None):
    seed = validate_seed(resolved_seed)
    training_env = dict(
        os.environ if base_environment is None else base_environment
    )
    training_env["PYTHONHASHSEED"] = str(seed)
    training_env[RUNNER_SEED_ENV] = str(seed)
    return training_env


class TrainingInterrupted(KeyboardInterrupt):
    """Raised after a child is terminated and its console tee is flushed."""

    def __init__(self, message, returncode=None):
        super(TrainingInterrupted, self).__init__(message)
        self.returncode = returncode


@contextlib.contextmanager
def _formal_gpu_lock(run_kind):
    """Reject concurrent formal runs targeting the same visible GPU set."""
    if run_kind != "formal":
        yield NOT_RECORDED
        return
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "all")
    safe_visible = re.sub(r"[^A-Za-z0-9_.-]+", "_", visible) or "all"
    lock_path = Path(tempfile.gettempdir()) / (
        "bot-reid-formal-gpu-{}.lock".format(safe_visible)
    )
    handle = lock_path.open("a+")
    acquired = False
    try:
        try:
            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write("0")
                handle.flush()
            handle.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(
                    handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB
                )
        except (IOError, OSError):
            raise RuntimeError(
                "Another formal experiment already holds the GPU lock: {}"
                .format(lock_path)
            )
        acquired = True
        yield str(lock_path)
    finally:
        try:
            if acquired:
                handle.seek(0)
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def _write_console_chunk(decoder, chunk, console_handle, terminal_stream,
                         final=False):
    text = decoder.decode(chunk, final=final)
    if text:
        console_handle.write(text)
        console_handle.flush()
        terminal_stream.write(text)
        terminal_stream.flush()


def _terminate_and_drain(process, decoder, console_handle, terminal_stream):
    if process.poll() is None:
        process.terminate()
    try:
        remaining, _ = process.communicate(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        remaining, _ = process.communicate()
    if remaining:
        _write_console_chunk(
            decoder, remaining, console_handle, terminal_stream
        )
    _write_console_chunk(
        decoder, b"", console_handle, terminal_stream, final=True
    )
    return process.returncode


def _launch_training_subprocess(train_command, training_env,
                                console_log_path=None,
                                terminal_stream=None):
    # Retain the callable's legacy diagnostic behavior for old callers; every
    # unified runner invocation below always supplies an independent log path.
    if console_log_path is None:
        return subprocess.run(
            train_command, cwd=str(REPO_ROOT), check=False, env=training_env
        )
    console_path = Path(console_log_path)
    console_path.parent.mkdir(parents=True, exist_ok=True)
    terminal_stream = terminal_stream or sys.stdout
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    child_environment = dict(training_env)
    child_environment["PYTHONIOENCODING"] = "utf-8:replace"
    child_environment["PYTHONUTF8"] = "1"
    process = subprocess.Popen(
        train_command, cwd=str(REPO_ROOT), env=child_environment,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=0,
    )
    previous_sigterm = None
    can_handle_sigterm = hasattr(signal, "SIGTERM")

    def interrupt_handler(signum, frame):
        raise TrainingInterrupted("Training interrupted by signal {}".format(signum))

    if can_handle_sigterm:
        try:
            previous_sigterm = signal.signal(signal.SIGTERM, interrupt_handler)
        except (ValueError, OSError):
            can_handle_sigterm = False
    try:
        with console_path.open(
                "w", encoding="utf-8", errors="replace", newline="") as handle:
            try:
                while True:
                    chunk = process.stdout.read(4096)
                    if not chunk:
                        break
                    _write_console_chunk(
                        decoder, chunk, handle, terminal_stream
                    )
                _write_console_chunk(
                    decoder, b"", handle, terminal_stream, final=True
                )
                returncode = process.wait()
            except (KeyboardInterrupt, TrainingInterrupted):
                _terminate_and_drain(
                    process, decoder, handle, terminal_stream
                )
                raise TrainingInterrupted(
                    "Training subprocess interrupted (return code {})".format(
                        process.returncode
                    ), returncode=process.returncode,
                )
            finally:
                handle.flush()
    finally:
        if process.stdout is not None:
            process.stdout.close()
        if can_handle_sigterm:
            signal.signal(signal.SIGTERM, previous_sigterm)
    return subprocess.CompletedProcess(train_command, returncode)


def _plain_config(local_cfg):
    import yaml
    from utils.reproducibility import resolved_config_text

    return yaml.safe_load(resolved_config_text(local_cfg))


def _changed_leaf_paths(left, right, prefix=""):
    paths = set()
    for key in set(left) | set(right):
        path = "{}.{}".format(prefix, key) if prefix else str(key)
        if key not in left or key not in right:
            paths.add(path)
        elif isinstance(left[key], dict) and isinstance(right[key], dict):
            paths.update(_changed_leaf_paths(left[key], right[key], path))
        elif left[key] != right[key]:
            paths.add(path)
    return paths


def _build_config_comparison(reference_path, reference_cfg, candidate_path,
                             candidate_cfg, expected_differences):
    expected = list(expected_differences or [])
    if len(expected) != len(set(expected)):
        raise RuntimeError("Expected config differences contain duplicates")
    reference_configuration = _plain_config(reference_cfg)
    candidate_configuration = _plain_config(candidate_cfg)
    observed = sorted(_changed_leaf_paths(
        reference_configuration, candidate_configuration
    ))
    if set(observed) != set(expected):
        raise RuntimeError(
            "Resolved-config differences are not isolated: expected={} "
            "observed={}".format(sorted(expected), observed)
        )
    reference_text = resolved_config_text(reference_cfg).encode("utf-8")
    candidate_text = resolved_config_text(candidate_cfg).encode("utf-8")
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "pass",
        "reference_config_path": str(Path(reference_path).resolve()).replace(
            "\\", "/"
        ),
        "reference_config_size_bytes": Path(reference_path).stat().st_size,
        "reference_config_sha256": sha256_file(reference_path),
        "reference_resolved_size_bytes": len(reference_text),
        "reference_resolved_sha256": sha256_bytes(reference_text),
        "candidate_config_path": str(Path(candidate_path).resolve()).replace(
            "\\", "/"
        ),
        "candidate_config_size_bytes": Path(candidate_path).stat().st_size,
        "candidate_config_sha256": sha256_file(candidate_path),
        "candidate_resolved_size_bytes": len(candidate_text),
        "candidate_resolved_sha256": sha256_bytes(candidate_text),
        "expected_differences": sorted(expected),
        "observed_differences": observed,
    }


def _require_new_output_dir(path):
    output = Path(path)
    if output.exists():
        if not output.is_dir():
            raise RuntimeError("OUTPUT_DIR exists and is not a directory: {}".format(output))
        if next(output.iterdir(), None) is not None:
            raise RuntimeError(
                "OUTPUT_DIR is not empty; refusing to overwrite evidence: {}"
                .format(output)
            )
    return output


def _model_manifest(configuration, method=None,
                    baseline_method=NOT_RECORDED,
                    baseline_commit=NOT_RECORDED,
                    parent_branch=NOT_RECORDED,
                    parent_commit=NOT_RECORDED,
                    feature_compatibility=None):
    identity = experiment_identity(configuration)
    signature, signature_sha256 = (
        canonical_multigranular_feature_signature(configuration)
    )
    feature_compatibility = feature_compatibility or {}
    return {
        "schema_version": SCHEMA_VERSION,
        "backbone": configuration.get("MODEL", {}).get("NAME", NOT_RECORDED),
        "neck": configuration.get("MODEL", {}).get("NECK", NOT_RECORDED),
        "method": method or identity["method"],
        "method_family": identity["method_family"],
        "method_variant": identity["method_variant"],
        "baseline": identity["baseline"],
        "baseline_method": baseline_method,
        "baseline_commit": baseline_commit,
        "parent_branch": parent_branch,
        "parent_commit": parent_commit,
        "feature_reference_commit": feature_compatibility.get(
            "feature_reference_commit", NOT_RECORDED
        ),
        "feature_reference_signature_sha256": feature_compatibility.get(
            "feature_reference_signature_sha256", NOT_RECORDED
        ),
        "current_feature_signature_sha256": feature_compatibility.get(
            "current_feature_signature_sha256", NOT_RECORDED
        ),
        "feature_compatibility_status": feature_compatibility.get(
            "feature_compatibility_status", NOT_RECORDED
        ),
        "feature_compatibility_evidence_path": (
            "feature_compatibility.json" if feature_compatibility
            else NOT_RECORDED
        ),
        "multigranular_feature_signature": signature,
        "multigranular_feature_signature_sha256": signature_sha256,
        "modules": identity["modules"],
        "part_correspondence_consistency": {
            "enabled": identity["pcc_enabled"],
            "parts": identity["pcc_parts"],
            "mode": identity["pcc_mode"],
            "alignment_strategy": identity["alignment_strategy"],
            "alignment_mode": identity["alignment_mode"],
            "alignment_temperature": identity["alignment_temperature"],
            "gating_mode": identity["gating_mode"],
            "gating_temperature": identity["gating_temperature"],
            "lambda": identity["pcc_lambda"],
            "warmup_epochs": identity["local_alignment_warmup_epochs"],
        },
        "cross_camera_positive_lambda": identity[
            "cross_camera_positive_lambda"
        ],
        "total_params": NOT_RECORDED,
        "trainable_params": NOT_RECORDED,
        "FLOPs": NOT_RECORDED,
        "source": "resolved_config; efficiency values populated post-training",
    }


def run(args):
    config_path = Path(args.config).resolve()
    _validate_run_overrides(args.run_kind, args.opts)
    git_info = validate_git_preflight(
        REPO_ROOT, args.expected_branch, expected_commit=args.expected_commit
    )
    source_local_cfg = _load_config(config_path)
    source_configuration = _plain_config(source_local_cfg)
    if bool(args.reference_config) != bool(args.expected_config_difference):
        raise RuntimeError(
            "--reference-config and --expected-config-difference must be "
            "provided together"
        )
    config_comparison = None
    if args.reference_config:
        reference_path = Path(args.reference_config).resolve()
        reference_cfg = _load_config(reference_path)
        config_comparison = _build_config_comparison(
            reference_path, reference_cfg, config_path, source_local_cfg,
            args.expected_config_difference,
        )
    effective_opts = _effective_run_opts(
        args.run_kind, source_local_cfg.OUTPUT_DIR, args.opts
    )
    _validate_run_overrides(args.run_kind, effective_opts)
    local_cfg = _load_config(config_path, effective_opts)
    output_dir = _require_new_output_dir(local_cfg.OUTPUT_DIR)
    configuration = _plain_config(local_cfg)
    _validate_resolved_run_config(
        configuration, args.run_kind, _load_source_output_dir(config_path)
    )
    if ((args.parent_branch == NOT_RECORDED)
            != (args.parent_commit == NOT_RECORDED)):
        raise RuntimeError(
            "Parent branch and parent commit must be recorded together"
        )
    if args.parent_branch != NOT_RECORDED:
        lineage = validate_parent_lineage(
            REPO_ROOT, args.parent_branch, args.parent_commit,
            child_commit=git_info["commit"],
        )
        git_info["merge_base"] = lineage["merge_base"]
    feature_compatibility = None
    alignment_mode = configuration.get("MODEL", {}).get("PCC_MODE")
    if (alignment_mode == "soft_min"
            and args.feature_reference_commit.lower()
            != FIXED_HARD_FEATURE_REFERENCE_COMMIT):
        raise RuntimeError(
            "Soft-Min feature reference must be the fixed Hard commit {}"
            .format(FIXED_HARD_FEATURE_REFERENCE_COMMIT)
        )
    if args.feature_reference_commit != NOT_RECORDED:
        reference_configuration = _load_revision_config(
            REPO_ROOT, args.feature_reference_commit,
            args.feature_reference_config,
        )
        feature_compatibility = require_feature_compatibility(
            build_feature_compatibility_evidence(
                REPO_ROOT, args.feature_reference_commit,
                git_info["commit"], reference_configuration, configuration,
            )
        )
    elif alignment_mode == "soft_min":
        raise RuntimeError(
            "Soft-Min requires --feature-reference-commit"
        )
    source_seed = _load_explicit_source_seed(config_path)
    resolved_seed = validate_seed(configuration.get("SEED", NOT_RECORDED))
    validate_seed_evidence_chain(source_seed, resolved_seed, resolved_seed)
    protocol_signature_sha256 = config_protocol_signature(
        source_configuration
    )
    implementation_signature_sha256 = git_implementation_signature(
        REPO_ROOT, git_info["commit"]
    )
    dataset = init_dataset(
        local_cfg.DATASETS.NAMES, root=local_cfg.DATASETS.ROOT_DIR
    )
    dataset_manifest = build_dataset_manifest(
        dataset, configuration, local_cfg.DATASETS.ROOT_DIR
    )
    smoke_gate_evidence = None
    if args.required_smoke_experiment_id and args.run_kind != "formal":
        raise RuntimeError(
            "--required-smoke-experiment-id is valid only for formal runs"
        )
    if args.run_kind == "formal" and args.required_smoke_experiment_id:
        smoke_gate_evidence = validate_formal_smoke_gate(
            repo_root=REPO_ROOT,
            records_root=args.records_root,
            formal_config_path=config_path,
            formal_configuration=configuration,
            current_commit=git_info["commit"],
            expected_branch=args.expected_branch,
            expected_experiment_id=args.required_smoke_experiment_id,
            expected_experiment_family=args.experiment_family,
            feature_compatibility=feature_compatibility,
            expected_parent_branch=args.parent_branch,
            expected_parent_commit=args.parent_commit,
            expected_merge_base=git_info.get("merge_base", NOT_RECORDED),
            expected_protocol_signature=protocol_signature_sha256,
            expected_implementation_signature=(
                implementation_signature_sha256
            ),
            expected_dataset_manifest=dataset_manifest,
        )
    training_env = _build_training_environment(resolved_seed)
    run_id = args.run_id or generate_run_id(
        args.experiment_id, git_info["commit"], resolved_seed
    )
    train_command = [
        sys.executable,
        "tools/train.py",
        "--config_file",
        str(config_path),
    ] + list(effective_opts)
    run_dir, manifest = initialize_run(
        records_root=args.records_root,
        experiment_id=args.experiment_id,
        experiment_family=args.experiment_family,
        run_id=run_id,
        config_file=str(config_path),
        resolved_cfg=local_cfg,
        output_dir=str(output_dir),
        git_info=git_info,
        notes=args.notes,
        command=sys.argv,
        expected_branch=args.expected_branch,
        method=args.method,
        baseline_method=args.baseline_method,
        baseline_commit=args.baseline_commit,
        run_kind=args.run_kind,
        parent_branch=args.parent_branch,
        parent_commit=args.parent_commit,
        feature_compatibility=feature_compatibility,
        experiments_path=REPO_ROOT / "EXPERIMENTS.md",
        protocol_signature_sha256=protocol_signature_sha256,
        implementation_signature_sha256=implementation_signature_sha256,
        dataset_manifest=dataset_manifest,
    )
    manifest["runner_command"] = list(sys.argv)
    manifest["training_command"] = list(train_command)
    if config_comparison is not None:
        comparison_path = atomic_write_json(
            run_dir / "config_comparison.json", config_comparison
        )
        manifest.update({
            "config_comparison_path": str(comparison_path.resolve()).replace(
                "\\", "/"
            ),
            "config_comparison_size_bytes": comparison_path.stat().st_size,
            "config_comparison_sha256": sha256_file(comparison_path),
        })
    atomic_write_json(run_dir / "run_manifest.json", manifest)
    if smoke_gate_evidence is not None:
        manifest["smoke_gate"] = smoke_gate_evidence
        atomic_write_json(run_dir / "run_manifest.json", manifest)
    try:
        environment = collect_environment(
            run_dir,
            REPO_ROOT,
            training_pythonhashseed=training_env["PYTHONHASHSEED"],
            expected_branch=git_info["branch"],
            expected_commit=git_info["commit"],
        )
        atomic_write_json(run_dir / "environment.json", environment)
        atomic_write_json(run_dir / "dataset_manifest.json", dataset_manifest)
        model_manifest = _model_manifest(
                configuration,
                method=args.method,
                baseline_method=args.baseline_method,
                baseline_commit=args.baseline_commit,
                parent_branch=args.parent_branch,
                parent_commit=args.parent_commit,
                feature_compatibility=feature_compatibility,
            )
        model_manifest.update({
            "feature_compatibility_evidence_path": manifest.get(
                "feature_compatibility_evidence_path", NOT_RECORDED
            ),
            "feature_compatibility_evidence_size_bytes": (
                (run_dir / "feature_compatibility.json").stat().st_size
                if (run_dir / "feature_compatibility.json").is_file()
                else NOT_RECORDED
            ),
            "feature_compatibility_evidence_sha256": manifest.get(
                "feature_compatibility_evidence_sha256", NOT_RECORDED
            ),
        })
        atomic_write_json(run_dir / "model_manifest.json", model_manifest)
        started = time.monotonic()
        completed = _launch_training_subprocess(
            train_command, training_env, run_dir / "console.log"
        )
        runtime = time.monotonic() - started
        record_training_exit(run_dir, completed.returncode, runtime)
        record_console_log_evidence(run_dir)
        if completed.returncode != 0:
            raise RuntimeError(
                "Training exited with code {}".format(completed.returncode)
            )
        result = finalize_run(
            run_dir=run_dir,
            records_root=args.records_root,
            repo_root=REPO_ROOT,
            experiments_path=REPO_ROOT / "EXPERIMENTS.md",
            run_analyses=True,
            verify_git=True,
        )
        print(json.dumps(result["metrics"], ensure_ascii=False, indent=2))
        return 0
    except BaseException as error:
        current_status = (
            "interrupted" if isinstance(error, (KeyboardInterrupt, TrainingInterrupted))
            else "failed"
        )
        try:
            console_path = run_dir / "console.log"
            if console_path.is_file() and console_path.stat().st_size > 0:
                record_console_log_evidence(run_dir)
        except Exception:
            pass
        if isinstance(error, TrainingInterrupted) and error.returncode is not None:
            try:
                record_training_exit(
                    run_dir, error.returncode, time.monotonic() - started
                )
            except Exception:
                pass
        try:
            status_path = run_dir / "run_status.json"
            status = json.loads(status_path.read_text(encoding="utf-8"))
            if (current_status != "interrupted"
                    and status.get("training_exit_code") == 0):
                current_status = "incomplete"
        except Exception:
            pass
        record_run_failure(run_dir, error, status=current_status)
        raise


def main(argv=None):
    args = parse_args(argv)
    try:
        with _formal_gpu_lock(args.run_kind):
            return run(args)
    except BaseException as error:
        print("Experiment failed closed: {}".format(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
