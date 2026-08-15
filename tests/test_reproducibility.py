# encoding: utf-8
"""Synthetic tests for the unified Seed=42 training protocol."""

from __future__ import absolute_import

import json
import os
import random
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
import yaml

from config import cfg
from data.samplers.triplet_sampler import RandomIdentitySampler
from tools.run_experiment import (
    _build_training_environment,
    _launch_training_subprocess,
    _validate_run_overrides,
)
from utils.reproducibility import (
    RUNNER_SEED_ENV,
    SAMPLER_SEED_STRATEGY,
    WORKER_SEED_STRATEGY,
    apply_reproducibility,
    build_dataloader_generator,
    collect_reproducibility_evidence,
    seed_worker,
    write_reproducibility_evidence,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
FORMAL_CONFIG = REPO_ROOT / "configs" / "softmax_triplet_c2l03_seed42_autodl.yml"
BASELINE_CONFIG = (
    REPO_ROOT / "configs" / "softmax_triplet_cross_camera_positive_lambda03_autodl.yml"
)


def _apply(seed=42):
    environment = {"PYTHONHASHSEED": str(seed), RUNNER_SEED_ENV: str(seed)}
    with mock.patch.dict(os.environ, environment, clear=False):
        return apply_reproducibility(seed)


def _sampler_order(seed):
    data = []
    for pid in range(8):
        for instance in range(5):
            data.append(("p{}_{}.jpg".format(pid, instance), pid, instance % 2))
    random.seed(seed)
    np.random.seed(seed)
    return list(RandomIdentitySampler(data, batch_size=8, num_instances=2))


class ReproducibilityProtocolTest(unittest.TestCase):
    def test_seed_config_loads_explicit_42(self):
        local_cfg = cfg.clone()
        local_cfg.merge_from_file(str(FORMAL_CONFIG))
        self.assertEqual(local_cfg.SEED, 42)
        self.assertEqual(cfg.SEED, 42)

    def test_python_rng_is_reproducible(self):
        _apply()
        first = [random.random() for _ in range(5)]
        _apply()
        self.assertEqual(first, [random.random() for _ in range(5)])

    def test_numpy_rng_is_reproducible(self):
        _apply()
        first = np.random.random_sample(5)
        _apply()
        np.testing.assert_array_equal(first, np.random.random_sample(5))

    def test_torch_cpu_rng_is_reproducible(self):
        _apply()
        first = torch.rand(5)
        _apply()
        self.assertTrue(torch.equal(first, torch.rand(5)))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_torch_cuda_rng_is_reproducible(self):
        _apply()
        first = torch.rand(5, device="cuda")
        _apply()
        self.assertTrue(torch.equal(first, torch.rand(5, device="cuda")))

    def test_both_cuda_seed_apis_are_applied(self):
        with mock.patch.dict(
            os.environ,
            {"PYTHONHASHSEED": "42", RUNNER_SEED_ENV: "42"},
            clear=False,
        ), mock.patch("torch.cuda.manual_seed") as manual_seed, mock.patch(
            "torch.cuda.manual_seed_all"
        ) as manual_seed_all:
            apply_reproducibility(42)
        manual_seed.assert_called_once_with(42)
        self.assertGreaterEqual(manual_seed_all.call_count, 1)
        self.assertEqual(manual_seed_all.call_args_list[-1], mock.call(42))

    def test_dataloader_generator_is_reproducible_and_streams_are_explicit(self):
        first = torch.randperm(100, generator=build_dataloader_generator(42, "train"))
        second = torch.randperm(100, generator=build_dataloader_generator(42, "train"))
        validation = torch.randperm(
            100, generator=build_dataloader_generator(42, "validation")
        )
        self.assertTrue(torch.equal(first, second))
        self.assertFalse(torch.equal(first, validation))

    def test_worker_init_seeds_python_numpy_and_torch(self):
        initial_seed = (2 ** 32) + 123
        with mock.patch("torch.initial_seed", return_value=initial_seed), mock.patch(
            "random.seed"
        ) as python_seed, mock.patch("numpy.random.seed") as numpy_seed, mock.patch(
            "torch.manual_seed"
        ) as torch_seed:
            seed_worker(7)
        python_seed.assert_called_once_with(123)
        numpy_seed.assert_called_once_with(123)
        torch_seed.assert_called_once_with(123)

    def test_sampler_order_is_reproducible_with_same_seed(self):
        self.assertEqual(_sampler_order(42), _sampler_order(42))

    def test_sampler_order_changes_with_different_seed(self):
        self.assertNotEqual(_sampler_order(42), _sampler_order(43))

    def test_pythonhashseed_is_passed_to_training_subprocess(self):
        training_env = _build_training_environment(42, {"EXISTING": "kept"})
        completed = mock.Mock(returncode=0)
        with mock.patch(
            "tools.run_experiment.subprocess.run", return_value=completed
        ) as run_process:
            result = _launch_training_subprocess(["python", "tools/train.py"], training_env)
        self.assertIs(result, completed)
        passed_env = run_process.call_args.kwargs["env"]
        self.assertEqual(passed_env["PYTHONHASHSEED"], "42")
        self.assertEqual(passed_env[RUNNER_SEED_ENV], "42")
        self.assertEqual(passed_env["EXISTING"], "kept")

    def test_formal_overrides_fail_and_smoke_overrides_are_isolated(self):
        with self.assertRaisesRegex(RuntimeError, "Formal runs forbid"):
            _validate_run_overrides(
                "formal", ["SOLVER.MAX_EPOCHS", "1"]
            )
        _validate_run_overrides(
            "smoke",
            [
                "SOLVER.MAX_EPOCHS", "1",
                "SOLVER.CHECKPOINT_PERIOD", "1",
                "SOLVER.EVAL_PERIOD", "1",
                "OUTPUT_DIR", "/tmp/isolated-smoke",
            ],
        )
        with self.assertRaisesRegex(RuntimeError, "non-isolated"):
            _validate_run_overrides(
                "smoke", ["MODEL.PCC_MODE", "fixed_index"]
            )

    def test_cudnn_protocol_is_deterministic_without_benchmark(self):
        _apply()
        self.assertIs(torch.backends.cudnn.deterministic, True)
        self.assertIs(torch.backends.cudnn.benchmark, False)

    def test_reproducibility_evidence_is_complete_and_training_produced(self):
        with mock.patch.dict(
            os.environ,
            {"PYTHONHASHSEED": "42", RUNNER_SEED_ENV: "42"},
            clear=False,
        ):
            applied = apply_reproducibility(42)
            evidence = collect_reproducibility_evidence(42, 42, applied)
        required = {
            "source_seed", "resolved_seed", "applied_seed", "seed",
            "PYTHONHASHSEED", "python_random_seed", "numpy_seed",
            "torch_cpu_seed", "torch_cuda_seed",
            "dataloader_worker_seed_base", "dataloader_worker_seed_strategy",
            "sampler_seed", "sampler_seed_strategy", "cudnn_deterministic",
            "cudnn_benchmark", "status",
        }
        self.assertTrue(required.issubset(evidence))
        self.assertEqual(evidence["dataloader_worker_seed_strategy"], WORKER_SEED_STRATEGY)
        self.assertEqual(evidence["sampler_seed_strategy"], SAMPLER_SEED_STRATEGY)
        self.assertEqual(evidence["dataloader_train_generator_seed"], 42)
        self.assertEqual(evidence["dataloader_validation_generator_seed"], 43)
        self.assertEqual(evidence["status"], "complete")
        with tempfile.TemporaryDirectory() as directory:
            path = write_reproducibility_evidence(directory, evidence)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), evidence)

    def test_mismatched_runner_or_hash_seed_fails_closed(self):
        with mock.patch.dict(
            os.environ,
            {"PYTHONHASHSEED": "7", RUNNER_SEED_ENV: "42"},
            clear=False,
        ):
            with self.assertRaisesRegex(RuntimeError, "before Python starts"):
                apply_reproducibility(42)
        with mock.patch.dict(
            os.environ,
            {"PYTHONHASHSEED": "42", RUNNER_SEED_ENV: "7"},
            clear=False,
        ):
            with self.assertRaisesRegex(RuntimeError, "Runner/config seed conflict"):
                apply_reproducibility(42)

    def test_c2l03_algorithm_config_is_unchanged(self):
        baseline = yaml.safe_load(BASELINE_CONFIG.read_text(encoding="utf-8"))
        formal = yaml.safe_load(FORMAL_CONFIG.read_text(encoding="utf-8"))
        self.assertEqual(formal.pop("SEED"), 42)
        self.assertEqual(formal["MODEL"].pop("NAME"), "resnet50")
        self.assertNotEqual(formal.pop("OUTPUT_DIR"), baseline.pop("OUTPUT_DIR"))
        self.assertEqual(formal, baseline)
        self.assertTrue(formal["MODEL"]["PART_ATTENTION"])
        self.assertEqual(formal["MODEL"]["PART_ATTENTION_PARTS"], 6)
        self.assertTrue(formal["MODEL"]["CROSS_CAMERA_POSITIVE_ONLY"])
        self.assertEqual(formal["MODEL"]["CROSS_CAMERA_POSITIVE_LAMBDA"], 0.3)
        self.assertEqual(formal["SOLVER"]["MAX_EPOCHS"], 120)


if __name__ == "__main__":
    unittest.main()
