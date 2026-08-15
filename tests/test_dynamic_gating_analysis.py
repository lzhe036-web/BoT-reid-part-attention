import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
from torch.utils.data import Dataset

from config import cfg
from tools.analyze_dynamic_gating import generate_dynamic_gating_evidence
from utils.dynamic_gating_evidence import GatingEpochAccumulator
from utils.experiment_recording import sha256_file


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = REPO_ROOT / "configs" / "softmax_triplet_c2_l03_multi_granularity_dynamic_gating_autodl.yml"


class SyntheticImages(Dataset):
    def __init__(self, entries, _transform):
        self.entries = entries

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        _path, pid, camid = self.entries[index]
        return torch.full((3, 8, 4), float(index)), pid, camid, str(_path)


class SyntheticModel(object):
    def __init__(self):
        self._last_dynamic_gating = None

    def load_state_dict(self, state, strict=True):
        self.state = state
        self.strict = strict

    def to(self, _device):
        return self

    def eval(self):
        return self

    def __call__(self, images):
        count = int(images.size(0))
        base = torch.arange(1, count + 1, device=images.device, dtype=torch.float64)
        probabilities = torch.stack((base, base + 1, base + 2), dim=1)
        probabilities = probabilities / probabilities.sum(dim=1, keepdim=True)
        self._last_dynamic_gating = {
            "probabilities": probabilities,
            "weights": 3.0 * probabilities,
        }
        return torch.zeros(count, 2816, device=images.device)


class DynamicGatingAnalysisTest(unittest.TestCase):
    def test_summary_and_bounded_tsv_are_checkpoint_bound_and_hashed(self):
        configuration = cfg.clone()
        configuration.merge_from_file(str(CONFIG))
        configuration.defrost()
        configuration.MODEL.DEVICE = "cpu"
        configuration.DATALOADER.NUM_WORKERS = 0
        configuration.freeze()
        selected = [
            ("key{:02d}".format(index), "query" if index < 2 else "gallery",
             "image{}.jpg".format(index), index, index % 2)
            for index in range(4)
        ]
        accumulator = GatingEpochAccumulator(1.0)
        accumulator.update([[1.0 / 3.0] * 3] * 4)
        epoch_statistics = accumulator.summary()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "selected.pt"
            torch.save({"weight": torch.tensor([1.0])}, checkpoint)
            with mock.patch(
                    "tools.analyze_dynamic_gating.select_samples",
                    return_value=(selected, 2)), mock.patch(
                    "tools.analyze_dynamic_gating.ImageDataset", SyntheticImages), mock.patch(
                    "tools.analyze_dynamic_gating.build_transforms", return_value=None), mock.patch(
                    "tools.analyze_dynamic_gating.build_model", return_value=SyntheticModel()):
                summary_path, samples_path, summary = generate_dynamic_gating_evidence(
                    configuration, checkpoint, root / "evidence", epoch_statistics,
                    device="cpu",
                )
            self.assertEqual(summary["source_checkpoint_sha256"], sha256_file(checkpoint))
            self.assertEqual(summary["selected_sample_count"], 4)
            self.assertEqual(summary["gating_samples"]["sha256"], sha256_file(samples_path))
            self.assertEqual(summary["gating_samples"]["size_bytes"], samples_path.stat().st_size)
            self.assertEqual(json.loads(summary_path.read_text(encoding="utf-8")), summary)
            with samples_path.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle, delimiter="\t"))
            self.assertEqual([row["stable_sample_key"] for row in rows], [item[0] for item in selected])
            for row in rows:
                self.assertAlmostEqual(sum(float(row[key]) for key in ("p2", "p4", "p6")), 1.0)
                self.assertAlmostEqual(sum(float(row[key]) for key in ("w2", "w4", "w6")), 3.0)
                self.assertEqual(row["checkpoint_sha256"], sha256_file(checkpoint))


if __name__ == "__main__":
    unittest.main()
