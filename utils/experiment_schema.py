# encoding: utf-8
"""Project-wide experiment evidence schema.

The current writer always emits schema v5. Readers retain the original
schema_version on historical v1-v4 rows; migration only adds explicit sentinel
values and never relabels old evidence as v5.
"""

SCHEMA_VERSION = 5
LEGACY_STATIC_SCHEMA_VERSION = 1
SUPPORTED_SCHEMA_VERSIONS = (1, 2, 3, 4, 5)

NOT_RECORDED = "not_recorded"
NOT_APPLICABLE = "not_applicable"
MISSING_EVIDENCE = "missing_evidence"

AUTO_RUNS_START = "<!-- AUTO-EXPERIMENT-RUNS:START -->"
AUTO_RUNS_END = "<!-- AUTO-EXPERIMENT-RUNS:END -->"
AUTO_RESULTS_START = "<!-- AUTO-EXPERIMENT-RESULTS:START -->"
AUTO_RESULTS_END = "<!-- AUTO-EXPERIMENT-RESULTS:END -->"
AUTO_CHECKPOINTS_START = "<!-- AUTO-CHECKPOINT-EVIDENCE:START -->"
AUTO_CHECKPOINTS_END = "<!-- AUTO-CHECKPOINT-EVIDENCE:END -->"

LEGACY_DYNAMIC_MARKERS = (
    ("<!-- AUTO-DYNAMIC-GATING-RUNS:START -->",
     "<!-- AUTO-DYNAMIC-GATING-RUNS:END -->"),
    ("<!-- AUTO-DYNAMIC-GATING-FORMAL:START -->",
     "<!-- AUTO-DYNAMIC-GATING-FORMAL:END -->"),
    ("<!-- AUTO-DYNAMIC-GATING-CHECKPOINTS:START -->",
     "<!-- AUTO-DYNAMIC-GATING-CHECKPOINTS:END -->"),
)

GATING_STAT_FIELDS = (
    "gating_temperature", "gating_sample_count",
    "p2_mean", "p2_std", "p2_min", "p2_max",
    "p4_mean", "p4_std", "p4_min", "p4_max",
    "p6_mean", "p6_std", "p6_min", "p6_max",
    "applied_w2_mean", "applied_w2_std",
    "applied_w4_mean", "applied_w4_std",
    "applied_w6_mean", "applied_w6_std",
    "mean_gate_entropy", "dominant_k2_ratio", "dominant_k4_ratio",
    "dominant_k6_ratio",
)

BASE_RUN_FIELDS = (
    "schema_version", "experiment_id", "experiment_family", "evidence_id",
    "run_id", "run_kind", "status", "method_family", "method_variant",
    "branch", "commit", "parent_branch", "parent_commit", "merge_base",
    "candidate_protocol_signature_sha256", "implementation_signature_sha256",
    "feature_reference_commit", "feature_reference_signature_sha256",
    "current_feature_signature_sha256", "feature_compatibility_status",
    "feature_compatibility_evidence_path",
    "feature_compatibility_evidence_size_bytes",
    "feature_compatibility_evidence_sha256", "gating_signature_sha256",
    "gating_signature_path", "gating_signature_size_bytes",
    "gating_signature_evidence_sha256",
    "seed", "source_config_origin_path", "source_config_origin_size_bytes",
    "source_config_origin_sha256", "source_config_path", "source_config_size_bytes",
    "source_config_sha256", "resolved_config_path",
    "resolved_config_size_bytes", "resolved_config_sha256",
    "training_log_path", "training_log_size_bytes", "training_log_sha256",
    "console_log_path", "console_log_size_bytes", "console_log_sha256",
    "output_dir", "selected_checkpoint_path", "selected_checkpoint_size_bytes",
    "selected_checkpoint_sha256", "checkpoint_manifest_path",
    "checkpoint_manifest_size_bytes", "checkpoint_manifest_sha256",
    "artifact_manifest_path", "artifact_manifest_size_bytes",
    "artifact_manifest_sha256", "run_evidence_manifest_path",
    "run_evidence_manifest_size_bytes", "run_evidence_manifest_sha256",
    "run_manifest_path", "run_manifest_size_bytes", "run_manifest_sha256",
    "run_status_path", "run_status_size_bytes", "run_status_sha256",
    "reproducibility_path", "reproducibility_size_bytes",
    "reproducibility_sha256", "dataset_manifest_path",
    "dataset_manifest_sha256", "model_manifest_path", "model_manifest_sha256",
    "environment_path", "environment_sha256", "gpu", "start_time", "end_time",
    "runtime_seconds", "return_code", "alignment_mode",
    "alignment_temperature", "gating_mode", "gating_input",
    "gating_temperature", "gating_normalization", "scale_order",
    "dynamic_gating_summary_path", "dynamic_gating_summary_size_bytes",
    "dynamic_gating_summary_sha256",
    "dynamic_gating_summary_source_checkpoint_sha256", "gating_samples_path",
    "gating_samples_size_bytes", "gating_samples_sha256",
    "gating_samples_source_checkpoint_sha256", "gating_sample_selection_rule",
)

RUN_FIELDS = BASE_RUN_FIELDS + tuple(
    field for field in GATING_STAT_FIELDS if field not in BASE_RUN_FIELDS
) + (
    "rank1_percent", "rank5_percent", "rank10_percent", "map_percent",
    "best_epoch", "selected_epoch", "notes",
)
FORMAL_FIELDS = RUN_FIELDS
EVIDENCE_FIELDS = (
    "schema_version", "run_id", "experiment_id", "run_kind", "status",
    "artifact_type", "path", "size_bytes", "sha256",
    "source_checkpoint_sha256", "selection_rule",
)
CHECKPOINT_FIELDS = (
    "run_id", "experiment_id", "run_kind", "checkpoint_path", "size_bytes",
    "ignite_epoch", "global_iteration", "sha256", "selected",
)


def validate_schema_version(value):
    try:
        version = int(value)
    except (TypeError, ValueError):
        raise ValueError("Invalid experiment schema version {!r}".format(value))
    if version not in SUPPORTED_SCHEMA_VERSIONS:
        raise ValueError("Unsupported experiment schema version {}".format(version))
    return version


def migrate_row(row, fields, assumed_version=LEGACY_STATIC_SCHEMA_VERSION):
    """Losslessly project a v1-v5 row onto the current field set."""
    migrated = dict(row)
    raw_version = migrated.get("schema_version")
    if raw_version in (None, "", NOT_RECORDED):
        raw_version = assumed_version
    version = validate_schema_version(raw_version)
    migrated["schema_version"] = str(version)
    aliases = {
        "commit": ("commit_id",),
        "source_config_path": ("config_file",),
        "training_log_path": ("log_path",),
        "training_log_sha256": ("log_sha256",),
        "rank1_percent": ("Rank-1",),
        "rank5_percent": ("Rank-5",),
        "rank10_percent": ("Rank-10",),
        "map_percent": ("mAP",),
        "selected_checkpoint_path": ("checkpoint",),
        "selected_checkpoint_sha256": ("checkpoint_sha256",),
    }
    for target, sources in aliases.items():
        if migrated.get(target) not in (None, "", NOT_RECORDED):
            continue
        for source in sources:
            if migrated.get(source) not in (None, ""):
                migrated[target] = migrated[source]
                break
    for field in fields:
        if migrated.get(field) in (None, ""):
            migrated[field] = NOT_RECORDED
    return migrated
