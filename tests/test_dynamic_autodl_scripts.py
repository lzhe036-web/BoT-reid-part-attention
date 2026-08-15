import tempfile
import unittest
from pathlib import Path

from utils.dynamic_experiment_registry import (
    DynamicExperimentEvidenceError,
    initialize_dynamic_run,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
PREPARE_SCRIPT = (
    REPO_ROOT / "scripts" / "prepare_c2_l03_dynamic_gating_autodl.sh"
)
SMOKE_SCRIPT = (
    REPO_ROOT / "scripts" /
    "test_c2_l03_multi_granularity_dynamic_gating_1epoch.sh"
)
STATIC_SHA = "9cd7dbcee07b255803c8c21f4d9c5ee67a30930e"


class DynamicAutoDLPreflightContractTest(unittest.TestCase):
    def test_preflight_uses_canonical_remote_and_fail_closed_lineage(self):
        source = PREPARE_SCRIPT.read_text(encoding="utf-8")
        required = (
            'REMOTE_URL="https://github.com/lzhe036-web/BoT-reid-part-attention.git"',
            'DYNAMIC_BRANCH="exp/c2-l03-multi-granularity-dynamic-gating"',
            'STATIC_BRANCH="exp/c2-l03-multi-granularity-local-feature"',
            'STATIC_SHA="{}"'.format(STATIC_SHA),
            'git config --local http.version HTTP/1.1',
            'git config --local core.compression 0',
            'git config --local http.lowSpeedLimit 1',
            'git config --local http.lowSpeedTime 30',
            'git remote set-url origin "$REMOTE_URL"',
            'git ls-remote --heads "$remote" "$STATIC_BRANCH"',
            'git cat-file -e "${STATIC_SHA}^{commit}"',
            'git branch -f "$STATIC_BRANCH" "$STATIC_SHA"',
            'git update-ref "refs/remotes/origin/${STATIC_BRANCH}" "$STATIC_SHA"',
            'git merge-base "$STATIC_BRANCH" HEAD',
            'verify_remote_static origin',
        )
        for value in required:
            self.assertIn(value, source)
        self.assertLess(
            source.index('verify_remote_static "$REMOTE_URL"'),
            source.index('git cat-file -e "${STATIC_SHA}^{commit}"'),
        )
        self.assertLess(
            source.index('git cat-file -e "${STATIC_SHA}^{commit}"'),
            source.index('git branch -f "$STATIC_BRANCH" "$STATIC_SHA"'),
        )
        self.assertNotIn("tools/train.py", source)
        self.assertNotIn("git reset", source)

    def test_smoke_runner_requires_explicit_safe_output_contract(self):
        source = SMOKE_SCRIPT.read_text(encoding="utf-8")
        self.assertIn("DEFAULT_SMOKE_OUTPUT_DIR=", source)
        self.assertIn(
            'SMOKE_OUTPUT_DIR="${SMOKE_OUTPUT_DIR:-${DEFAULT_SMOKE_OUTPUT_DIR}}"',
            source,
        )
        self.assertIn('[[ -e "$SMOKE_OUTPUT_DIR" && ! -d "$SMOKE_OUTPUT_DIR" ]]', source)
        self.assertIn('find "$SMOKE_OUTPUT_DIR" -mindepth 1 -maxdepth 1', source)
        self.assertIn('--output-dir "${SMOKE_OUTPUT_DIR}"', source)
        self.assertNotIn("rm -", source)

    def test_nonempty_output_is_also_rejected_by_unified_recorder(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = root / "experiment_records"
            experiments = root / "EXPERIMENTS.md"
            experiments.write_text("# Experiments\n", encoding="utf-8")
            config = root / "config.yml"
            config.write_text("SEED: 42\n", encoding="utf-8")
            output = root / "existing-output"
            output.mkdir()
            (output / "old-evidence.txt").write_text(
                "preserve", encoding="utf-8"
            )
            lineage = {
                "branch": "exp/c2-l03-multi-granularity-dynamic-gating",
                "commit": "a" * 40,
                "parent_branch": "exp/c2-l03-multi-granularity-local-feature",
                "parent_commit": STATIC_SHA, "merge_base": STATIC_SHA,
            }
            feature = {
                "feature_reference_commit": STATIC_SHA,
                "feature_reference_signature_sha256": "b" * 64,
                "current_feature_signature_sha256": "b" * 64,
                "feature_compatibility_status": "compatible",
                "fusion_gating_signature": {
                    "current_sha256": "c" * 64,
                },
            }
            with self.assertRaisesRegex(
                    DynamicExperimentEvidenceError, "non-empty"):
                initialize_dynamic_run(
                    records, experiments, "SMOKE", "smoke", config,
                    "SEED: 42\n", output, lineage, feature,
                    ["python", "tools/train.py"],
                )
            self.assertEqual(
                (output / "old-evidence.txt").read_text(encoding="utf-8"),
                "preserve",
            )


if __name__ == "__main__":
    unittest.main()
