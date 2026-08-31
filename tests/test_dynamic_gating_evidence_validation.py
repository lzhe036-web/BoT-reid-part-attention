import csv
import hashlib
import io
import json
import math
import tempfile
import unittest
from pathlib import Path

from utils.dynamic_gating_evidence import (
    DYNAMIC_GATING_SAMPLE_FIELDS,
    DYNAMIC_GATING_SELECTION_RULE,
    DynamicGatingEvidenceError,
    GatingEpochAccumulator,
    dynamic_gating_sample_fields,
    validate_dynamic_gating_evidence,
)
from utils.experiment_schema import SCHEMA_VERSION


class DynamicGatingEvidenceValidationTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.samples_path = self.root / "gating_samples.tsv"
        self.summary_path = self.root / "dynamic_gating_summary.json"
        self.checkpoint_sha = "a" * 64
        keys = ["stable-a", "stable-b"]
        keys.sort(key=lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest())
        metadata = {
            "stable-a": ("query", 1, 2),
            "stable-b": ("gallery", 3, 4),
        }
        probabilities = {
            "stable-a": (0.2, 0.3, 0.5),
            "stable-b": (0.4, 0.4, 0.2),
        }
        self.rows = []
        self.expected = []
        for key in keys:
            split, pid, camid = metadata[key]
            p = probabilities[key]
            entropy = -sum(value * math.log(value) for value in p)
            dominant = (2, 4, 6)[max(range(3), key=lambda index: p[index])]
            self.rows.append({
                "stable_sample_key": key, "dataset_split": split,
                "pid": pid, "camid": camid,
                "p2": p[0], "p4": p[1], "p6": p[2],
                "w2": 3 * p[0], "w4": 3 * p[1], "w6": 3 * p[2],
                "entropy": entropy, "dominant_k": dominant,
                "checkpoint_sha256": self.checkpoint_sha,
            })
            self.expected.append((key, split, "unused.jpg", pid, camid))
        accumulator = GatingEpochAccumulator(1.0)
        accumulator.update([
            [row["p2"], row["p4"], row["p6"]] for row in self.rows
        ])
        self.statistics = accumulator.summary()
        self.resolved = {"MODEL": {"MULTI_GRANULARITY_GATING_TAU": 1.0}}
        self.write_samples(update_summary=False)
        self.summary = {
            "schema_version": SCHEMA_VERSION,
            "source_checkpoint_sha256": self.checkpoint_sha,
            "selection_rule": DYNAMIC_GATING_SELECTION_RULE,
            "selected_sample_count": len(self.rows),
            "training_epoch_statistics": dict(self.statistics),
            "deterministic_sample_statistics": dict(self.statistics),
            "gating_samples": self.samples_evidence(),
        }
        self.write_summary()

    def tearDown(self):
        self.temporary.cleanup()

    def samples_evidence(self):
        return {
            "path": str(self.samples_path.resolve()),
            "size_bytes": self.samples_path.stat().st_size,
            "sha256": hashlib.sha256(self.samples_path.read_bytes()).hexdigest(),
            "source_checkpoint_sha256": self.checkpoint_sha,
            "selection_rule": DYNAMIC_GATING_SELECTION_RULE,
        }

    def write_samples(self, update_summary=True):
        buffer = io.StringIO(newline="")
        writer = csv.DictWriter(
            buffer, fieldnames=DYNAMIC_GATING_SAMPLE_FIELDS,
            delimiter="\t", lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(self.rows)
        self.samples_path.write_text(buffer.getvalue(), encoding="utf-8", newline="\n")
        if update_summary:
            self.summary["gating_samples"] = self.samples_evidence()
            self.write_summary()

    def write_summary(self):
        self.summary_path.write_text(
            json.dumps(self.summary, sort_keys=True), encoding="utf-8"
        )

    def validate(self):
        return validate_dynamic_gating_evidence(
            self.summary_path, self.samples_path, self.checkpoint_sha,
            self.resolved, self.statistics, {},
            selection_resolver=lambda _cfg: list(self.expected),
            dataset_validator=lambda _cfg, _manifest: None,
        )

    def assert_tamper_fails(self):
        with self.assertRaises(DynamicGatingEvidenceError):
            self.validate()

    def test_valid_summary_and_tsv_pass(self):
        self.assertEqual(self.validate()["sample_count"], 2)

    def test_replaced_tsv_fails(self):
        self.samples_path.write_text("replaced\n", encoding="utf-8")
        self.assert_tamper_fails()

    def test_summary_tsv_sha_fails(self):
        self.summary["gating_samples"]["sha256"] = "b" * 64
        self.write_summary()
        self.assert_tamper_fails()

    def test_row_checkpoint_sha_fails(self):
        self.rows[0]["checkpoint_sha256"] = "b" * 64
        self.write_samples()
        self.assert_tamper_fails()

    def test_probability_sum_fails(self):
        self.rows[0]["p2"] = 0.9
        self.write_samples()
        self.assert_tamper_fails()

    def test_weight_relation_fails(self):
        self.rows[0]["w2"] = 0.0
        self.write_samples()
        self.assert_tamper_fails()

    def test_entropy_fails(self):
        self.rows[0]["entropy"] = 99.0
        self.write_samples()
        self.assert_tamper_fails()

    def test_dominant_k_fails(self):
        self.rows[0]["dominant_k"] = 2 if self.rows[0]["dominant_k"] != 2 else 4
        self.write_samples()
        self.assert_tamper_fails()

    def test_duplicate_sample_key_fails(self):
        self.rows[1]["stable_sample_key"] = self.rows[0]["stable_sample_key"]
        self.write_samples()
        self.assert_tamper_fails()

    def test_selection_order_fails(self):
        self.rows.reverse()
        self.write_samples()
        self.assert_tamper_fails()

    def test_summary_sample_count_fails(self):
        self.summary["selected_sample_count"] = 3
        self.write_summary()
        self.assert_tamper_fails()

    def test_summary_statistics_fails(self):
        self.summary["deterministic_sample_statistics"]["p2_mean"] += 0.1
        self.write_summary()
        self.assert_tamper_fails()

    def test_extra_column_fails(self):
        content = self.samples_path.read_text(encoding="utf-8")
        lines = content.splitlines()
        lines[0] += "\textra"
        lines[1] += "\tvalue"
        lines[2] += "\tvalue"
        self.samples_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        self.summary["gating_samples"] = self.samples_evidence()
        self.write_summary()
        self.assert_tamper_fails()

    def test_two_way_schema_excludes_z2_and_uses_unscaled_weights(self):
        checkpoint_sha = "b" * 64
        samples_path = self.root / "two_way_samples.tsv"
        summary_path = self.root / "two_way_summary.json"
        keys = ["two-way-a", "two-way-b"]
        keys.sort(key=lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest())
        probabilities = ((0.25, 0.75), (0.6, 0.4))
        rows = []
        expected = []
        for index, key in enumerate(keys):
            p4, p6 = probabilities[index]
            rows.append({
                "stable_sample_key": key,
                "dataset_split": "query" if index == 0 else "gallery",
                "pid": index + 1,
                "camid": index + 2,
                "p4": p4,
                "p6": p6,
                "w4": p4,
                "w6": p6,
                "entropy": -(p4 * math.log(p4) + p6 * math.log(p6)),
                "dominant_k": 4 if p4 > p6 else 6,
                "checkpoint_sha256": checkpoint_sha,
            })
            expected.append((key, rows[-1]["dataset_split"], "unused.jpg", index + 1, index + 2))
        buffer = io.StringIO(newline="")
        writer = csv.DictWriter(
            buffer, fieldnames=dynamic_gating_sample_fields((4, 6)),
            delimiter="\t", lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
        samples_path.write_text(buffer.getvalue(), encoding="utf-8", newline="\n")
        accumulator = GatingEpochAccumulator(1.0, scales=(4, 6), weight_sum=1.0)
        accumulator.update(probabilities)
        statistics = accumulator.summary()
        summary = {
            "schema_version": SCHEMA_VERSION,
            "source_checkpoint_sha256": checkpoint_sha,
            "selection_rule": DYNAMIC_GATING_SELECTION_RULE,
            "selected_sample_count": len(rows),
            "training_epoch_statistics": statistics,
            "deterministic_sample_statistics": statistics,
            "gating_samples": {
                "path": str(samples_path.resolve()),
                "size_bytes": samples_path.stat().st_size,
                "sha256": hashlib.sha256(samples_path.read_bytes()).hexdigest(),
                "source_checkpoint_sha256": checkpoint_sha,
                "selection_rule": DYNAMIC_GATING_SELECTION_RULE,
            },
        }
        summary_path.write_text(json.dumps(summary, sort_keys=True), encoding="utf-8")
        resolved = {
            "MODEL": {
                "MULTI_GRANULARITY_GATING_TAU": 1.0,
                "MULTI_GRANULARITY_GATING_INPUT": "concat_z4_z6",
            }
        }
        validated = validate_dynamic_gating_evidence(
            summary_path, samples_path, checkpoint_sha, resolved, statistics, {},
            selection_resolver=lambda _cfg: expected,
            dataset_validator=lambda _cfg, _manifest: None,
        )
        self.assertEqual(validated["sample_count"], 2)
        self.assertNotIn("p2", samples_path.read_text(encoding="utf-8").splitlines()[0])

        # A legacy z2 column is not accepted in the two-way protocol.
        content = samples_path.read_text(encoding="utf-8")
        lines = content.splitlines()
        lines[0] = lines[0].replace("\tp4", "\tp2\tp4")
        lines[1] = lines[1].replace("\t0.25\t0.75", "\t0.0\t0.25\t0.75")
        lines[2] = lines[2].replace("\t0.6\t0.4", "\t0.0\t0.6\t0.4")
        samples_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        summary["gating_samples"]["size_bytes"] = samples_path.stat().st_size
        summary["gating_samples"]["sha256"] = hashlib.sha256(samples_path.read_bytes()).hexdigest()
        summary_path.write_text(json.dumps(summary, sort_keys=True), encoding="utf-8")
        with self.assertRaisesRegex(DynamicGatingEvidenceError, "header mismatch"):
            validate_dynamic_gating_evidence(
                summary_path, samples_path, checkpoint_sha, resolved, statistics, {},
                selection_resolver=lambda _cfg: expected,
                dataset_validator=lambda _cfg, _manifest: None,
            )


if __name__ == "__main__":
    unittest.main()
