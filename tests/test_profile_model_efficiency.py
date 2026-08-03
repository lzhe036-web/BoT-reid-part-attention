import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


TOOL = Path(__file__).resolve().parents[1] / "tools" / "profile_model_efficiency.py"
SPEC = importlib.util.spec_from_file_location("profile_model_efficiency", TOOL)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class ProfileModelEfficiencyTests(unittest.TestCase):
    def test_formal_id_set_is_exactly_twelve_and_excludes_historical_rows(self):
        specs = MODULE.experiment_specs()
        self.assertEqual(tuple(specs), MODULE.FORMAL_IDS)
        self.assertEqual(len(specs), 12)
        self.assertNotIn("CAT001", specs)
        self.assertNotIn("C2-CCPO-Market", specs)

    def test_log_parser_prefers_explicit_runtime(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "log.txt"
            log.write_text("Training runtime: 12.5 minutes\n", encoding="utf-8")
            self.assertEqual(MODULE.parse_training_time(log)[:2], ("12.5 minutes", "explicit_recorded_runtime"))

    def test_log_parser_marks_timestamp_span_as_derived(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "log.txt"
            log.write_text("2026-01-01 00:00:00 start\n2026-01-01 00:01:30 end\n", encoding="utf-8")
            self.assertEqual(MODULE.parse_training_time(log)[:2], ("90 s", "derived_log_span"))


if __name__ == "__main__":
    unittest.main()
