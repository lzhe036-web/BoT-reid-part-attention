import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class G2EAutoDLScriptsTest(unittest.TestCase):
    def test_smoke_and_formal_are_isolated_and_gate_on_own_smoke_evidence(self):
        smoke = (REPO_ROOT / "scripts" / "test_g2_e_static_dynamic_alpha0p3_tau0p5_1epoch_autodl.sh").read_text(encoding="utf-8")
        formal = (REPO_ROOT / "scripts" / "train_g2_e_static_dynamic_alpha0p3_tau0p5_seed42_autodl.sh").read_text(encoding="utf-8")
        self.assertIn("codex/g2-e-static-dynamic-alpha0p3-tau0p5", smoke)
        self.assertIn("G2-E smoke requires a clean worktree", smoke)
        self.assertIn("G2_A_BASE_COMMIT=\"d724a6536e4a819c5d2932412e90b7dea224041b\"", formal)
        self.assertIn("g2_e_smoke_analysis/g2_e_static_dynamic_alpha0p3_tau0p5_analysis_manifest.json", formal)
        self.assertIn("test ! -e \"${OUTPUT_DIR}\"", formal)
        self.assertIn("finalize_g2_e_static_dynamic_alpha0p3_tau0p5_experiment.py", formal)
        self.assertIn("package_g2_e_static_dynamic_alpha0p3_tau0p5_result.py", formal)
        self.assertIn("recover_g2_e_static_dynamic_alpha0p3_tau0p5_experiment.py", formal)

    def test_export_verifies_existing_package_before_archiving(self):
        export = (REPO_ROOT / "scripts" / "export_g2_e_static_dynamic_alpha0p3_tau0p5_result_autodl.sh").read_text(encoding="utf-8")
        self.assertIn("sha256sum -c SHA256SUMS.txt", export)
        self.assertIn("Refusing to overwrite archive", export)
        self.assertIn("tar -C", export)


if __name__ == "__main__":
    unittest.main()
