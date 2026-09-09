"""Checkpoint sidecars bind same-shaped alpha variants to training provenance."""
import json
from pathlib import Path
from utils.config_serialization import deserialize_cfg_node_yaml
from utils.experiment_recording import (atomic_write_json, sha256_file,
                                        build_dynamic_checkpoint_manifest, read_validation_history, select_dynamic_checkpoint)
from utils.multigranularity_signatures import _fusion_signature


def seal(output_dir):
    output_dir = Path(output_dir)
    provenance_path = output_dir/"reproducibility.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    resolved = output_dir/"config_resolved.yml"
    configuration = deserialize_cfg_node_yaml(resolved.read_text(encoding="utf-8"))
    signature, digest = _fusion_signature(configuration, True)
    history = read_validation_history(output_dir/"validation_history.jsonl")
    rows = build_dynamic_checkpoint_manifest(output_dir, history)
    select_dynamic_checkpoint(rows, history)
    for row in rows:
        checkpoint = output_dir/row["relative_path"]
        payload = {"schema_version": 1, "checkpoint_sha256": row["sha256"], "epoch": int(row["epoch"]),
                   "training_commit": provenance["code"]["commit"], "training_branch": provenance["code"]["branch"],
                   "reproducibility_sha256": sha256_file(provenance_path),
                   "resolved_config_sha256": sha256_file(resolved),
                   "feature_signature": signature, "feature_signature_sha256": digest}
        target = Path(str(checkpoint)+".metadata.json")
        if target.exists() and json.loads(target.read_text(encoding="utf-8")) != payload:
            raise ValueError("Refusing to replace checkpoint identity: {}".format(target))
        atomic_write_json(target, payload)


def validate(checkpoint_path, configuration):
    checkpoint = Path(checkpoint_path)
    sidecar = Path(str(checkpoint)+".metadata.json")
    metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    signature, digest = _fusion_signature(configuration, True)
    if (metadata["checkpoint_sha256"] != sha256_file(checkpoint)
            or metadata["feature_signature"] != signature
            or metadata["feature_signature_sha256"] != digest):
        raise ValueError("Checkpoint alpha/tau/fusion signature mismatch")
    provenance_path, resolved_path = checkpoint.parent/"reproducibility.json", checkpoint.parent/"config_resolved.yml"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if (metadata["training_commit"] != provenance["code"]["commit"]
            or metadata["training_branch"] != provenance["code"]["branch"]
            or metadata["reproducibility_sha256"] != sha256_file(provenance_path)
            or metadata["resolved_config_sha256"] != sha256_file(resolved_path)):
        raise ValueError("Checkpoint training identity mismatch")
    return metadata
