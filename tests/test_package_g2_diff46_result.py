import csv
import hashlib
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from yacs.config import CfgNode

from tools.package_g2_diff46_result import PACKAGE_NAME, package
from utils.config_serialization import serialize_cfg_node_yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


class G2Diff46PackageTest(unittest.TestCase):
    def test_synthetic_machine_evidence_creates_hashed_delivery_layout(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "output"; output.mkdir()
            source = output / "source.yml"; source.write_text("SEED: 42\n", encoding="utf-8")
            (output / "reproducibility.json").write_text(
                json.dumps({"command": ["python", "tools/train.py"], "environment": {"python": "synthetic"}}),
                encoding="utf-8",
            )
            (output / "config_resolved.yml").write_text(
                serialize_cfg_node_yaml(CfgNode({"DATASETS": {"ROOT_DIR": str(output)} })), encoding="utf-8")
            checkpoint_sha = "a" * 64
            samples = output / "gating_samples.tsv"
            fields = ("stable_sample_key", "dataset_split", "pid", "camid", "p2", "p4", "p6", "w2", "w4", "w6", "entropy", "dominant_k", "checkpoint_sha256")
            rows = [
                {"stable_sample_key": "query|market1501/query/a.jpg|1|0", "dataset_split": "query", "pid": 1, "camid": 0, "p2": .2, "p4": .6, "p6": .2, "w2": .6, "w4": 1.8, "w6": .6, "entropy": .9, "dominant_k": 4, "checkpoint_sha256": checkpoint_sha},
                {"stable_sample_key": "gallery|market1501/bounding_box_test/b.jpg|2|1", "dataset_split": "gallery", "pid": 2, "camid": 1, "p2": .1, "p4": .2, "p6": .7, "w2": .3, "w4": .6, "w6": 2.1, "entropy": .8, "dominant_k": 6, "checkpoint_sha256": checkpoint_sha},
            ]
            with samples.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t"); writer.writeheader(); writer.writerows(rows)
            analysis = output / "analysis.json"
            analysis.write_text(json.dumps({"files": {"test_gate_samples_tsv": {"path": str(samples)}}, "test_weight_protocol": "synthetic fixed evidence"}), encoding="utf-8")
            commit = subprocess.check_output(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"]).decode().strip()
            result = {"commit": commit, "branch": "codex/g2-d1", "gating_input": "concat([g, z2, z4, z6, abs(z4-z6)])", "gating_input_mode": "concat_global_local_diff46", "controller_input_dim": 3072, "controller_parameter_count": 9219, "retrieval_feature_dim": 2816, "delta46_definition": "torch.abs(z4-z6); unweighted; graph-connected; controller input only", "gating_temperature": .5, "selected_checkpoint": {"epoch": 120, "sha256": checkpoint_sha, "path": "/synthetic.pt"}, "metrics": {"rank1_percent": 90.0, "rank5_percent": 95.0, "rank10_percent": 97.0, "map_percent": 80.0}, "evidence": {"config": str(source), "analysis_manifest": str(analysis)}}
            (output / "g2_d1_formal_result.json").write_text(json.dumps(result), encoding="utf-8")
            package_dir = package(output)
            self.assertEqual(package_dir.name, PACKAGE_NAME)
            required = ("README.md", "retrieval_metrics.csv", "gate_statistics.csv", "per_sample_gating.tsv", "sample_manifest.tsv", "sample_selection.csv", "config_source.yml", "config_resolved.yml", "run_manifest.json", "code_diff_from_g2a_tau0p5.patch", "SHA256SUMS", "scripts/package_g2_diff46_result.py", "scripts/compare_g1_g2_g2a_diff46_gating.py", "scripts/analyze_g2_global_local_gating.py", "scripts/export_g2_diff46_result_autodl.sh")
            for name in required: self.assertTrue((package_dir / name).is_file(), name)
            for line in (package_dir / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
                digest, relative = line.split("  ", 1)
                self.assertEqual(digest, hashlib.sha256((package_dir / relative).read_bytes()).hexdigest())


if __name__ == "__main__":
    unittest.main()
