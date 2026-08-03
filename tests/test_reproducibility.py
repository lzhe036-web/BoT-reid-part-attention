import ast
import importlib.util
import json
import os
import random
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from torch.backends import cudnn

from config.defaults import _C
from scripts.append_experiment_result import PENDING, parse_metrics
from utils.reproducibility import (
    UINT32_LIMIT,
    ensure_python_hash_seed,
    seed_everything,
    seed_worker,
    validate_seed,
    write_reproducibility_record,
)


class DummyConfig(object):
    SEED = 314

    def __str__(self):
        return "OUTPUT_DIR: /tmp/example\nSEED: 314"


class ReproducibilityTest(unittest.TestCase):
    def test_validate_seed_accepts_uint32_and_rejects_ambiguous_values(self):
        self.assertEqual(validate_seed(0), 0)
        self.assertEqual(validate_seed(UINT32_LIMIT - 1), UINT32_LIMIT - 1)

        for value in (-1, UINT32_LIMIT, True, 1.5, "42", None):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    validate_seed(value)

    def test_same_seed_replays_python_numpy_and_torch_cpu(self):
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        torch_state = torch.get_rng_state()
        benchmark = cudnn.benchmark
        deterministic = cudnn.deterministic
        try:
            first_state = seed_everything(23)
            first = (random.random(), np.random.rand(4), torch.rand(4))

            second_state = seed_everything(23)
            second = (random.random(), np.random.rand(4), torch.rand(4))

            self.assertEqual(first[0], second[0])
            np.testing.assert_array_equal(first[1], second[1])
            self.assertTrue(torch.equal(first[2], second[2]))
            self.assertEqual(first_state["seed"], 23)
            self.assertEqual(second_state["seed"], 23)
            self.assertTrue(cudnn.deterministic)
            self.assertFalse(cudnn.benchmark)
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)
            torch.set_rng_state(torch_state)
            cudnn.benchmark = benchmark
            cudnn.deterministic = deterministic

    def test_invalid_seed_does_not_change_rng_state(self):
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        torch_state = torch.get_rng_state().clone()

        with self.assertRaises(ValueError):
            seed_everything(-1)

        self.assertEqual(random.getstate(), python_state)
        current_numpy_state = np.random.get_state()
        self.assertEqual(current_numpy_state[0], numpy_state[0])
        np.testing.assert_array_equal(current_numpy_state[1], numpy_state[1])
        self.assertEqual(current_numpy_state[2:], numpy_state[2:])
        self.assertTrue(torch.equal(torch.get_rng_state(), torch_state))

    def test_worker_seed_uses_torch_initial_seed_modulo_uint32(self):
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        try:
            with mock.patch(
                    "utils.reproducibility.torch.initial_seed",
                    return_value=UINT32_LIMIT + 17):
                seed_worker(5)
            actual_python = random.random()
            actual_numpy = np.random.rand()

            expected_python = random.Random(17).random()
            expected_numpy = np.random.RandomState(17).rand()
            self.assertEqual(actual_python, expected_python)
            self.assertEqual(actual_numpy, expected_numpy)
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)

    def test_python_hash_seed_is_applied_by_restarting_once(self):
        with mock.patch.dict(os.environ, {"PYTHONHASHSEED": "23"}):
            self.assertFalse(ensure_python_hash_seed(23, argv=["tools/train.py"]))

        with mock.patch.dict(os.environ, {"PYTHONHASHSEED": "7"}):
            with mock.patch(
                    "utils.reproducibility.os.execvpe",
                    side_effect=RuntimeError("restarted")) as exec_mock:
                with self.assertRaisesRegex(RuntimeError, "restarted"):
                    ensure_python_hash_seed(23, argv=["tools/train.py"])
        executable, arguments, environment = exec_mock.call_args[0]
        self.assertEqual(executable, sys.executable)
        self.assertEqual(arguments[1:], ["tools/train.py"])
        self.assertEqual(environment["PYTHONHASHSEED"], "23")

    def test_all_data_loaders_use_seed_worker_without_new_generator_api(self):
        build_path = Path(__file__).resolve().parents[1] / "data" / "build.py"
        tree = ast.parse(build_path.read_text(encoding="utf-8"))
        calls = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "DataLoader"
        ]
        self.assertEqual(len(calls), 3)
        for call in calls:
            keywords = {keyword.arg: keyword.value for keyword in call.keywords}
            self.assertIn("worker_init_fn", keywords)
            self.assertIsInstance(keywords["worker_init_fn"], ast.Name)
            self.assertEqual(keywords["worker_init_fn"].id, "seed_worker")
            self.assertNotIn("generator", keywords)

    def test_random_identity_sampler_replays_each_epoch_from_its_own_seed(self):
        sampler_path = (
            Path(__file__).resolve().parents[1]
            / "data"
            / "samplers"
            / "triplet_sampler.py"
        )
        spec = importlib.util.spec_from_file_location(
            "isolated_triplet_sampler", str(sampler_path)
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        samples = []
        for pid in range(4):
            for instance in range(4):
                samples.append(("{}_{}.jpg".format(pid, instance), pid, instance % 2))

        first = module.RandomIdentitySampler(samples, 8, 2, seed=23)
        second = module.RandomIdentitySampler(samples, 8, 2, seed=23)
        self.assertEqual(list(iter(first)), list(iter(second)))
        self.assertEqual(list(iter(first)), list(iter(second)))

    def test_metadata_and_resolved_config_are_saved(self):
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        torch_state = torch.get_rng_state()
        benchmark = cudnn.benchmark
        deterministic = cudnn.deterministic
        try:
            seed_state = seed_everything(314)
            with tempfile.TemporaryDirectory() as directory:
                config_path = Path(directory) / "source.yml"
                config_path.write_text("SEED: 314\n", encoding="utf-8")
                metadata_path, metadata = write_reproducibility_record(
                    output_dir=directory,
                    cfg=DummyConfig(),
                    seed_state=seed_state,
                    config_file=str(config_path),
                    cli_overrides=["SEED", "314"],
                    command=["python", "tools/train.py"],
                    repo_dir=directory,
                )

                saved = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
                resolved = Path(directory) / "config_resolved.yml"
                self.assertTrue(resolved.is_file())
                self.assertEqual(saved["seed"], 314)
                self.assertEqual(saved["seed_source"], "resolved_config.SEED")
                self.assertTrue(saved["seed_applied_before_data_loading"])
                self.assertTrue(saved["random_state"]["cudnn_deterministic"])
                self.assertFalse(saved["random_state"]["cudnn_benchmark"])
                self.assertEqual(
                    saved["environment"]["cublas_workspace_config"], ":4096:8"
                )
                self.assertEqual(saved["configuration"]["cli_overrides"], ["SEED", "314"])
                self.assertEqual(metadata["configuration"]["source_file_sha256"], saved["configuration"]["source_file_sha256"])
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)
            torch.set_rng_state(torch_state)
            cudnn.benchmark = benchmark
            cudnn.deterministic = deterministic

    def test_default_config_contains_a_valid_explicit_seed(self):
        self.assertEqual(_C.SEED, 42)
        self.assertEqual(validate_seed(_C.SEED), 42)

    def test_recorder_never_backfills_default_seed_into_historical_run(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "log.txt").write_text(
                "2026-07-14 06:17:58 Validation Results - Epoch: 120\n"
                "2026-07-14 06:17:59 mAP: 86.8%\n"
                "2026-07-14 06:18:00 CMC curve, Rank-1  :94.4%\n",
                encoding="utf-8",
            )
            metrics = parse_metrics(directory)
        self.assertEqual(metrics["seed"], PENDING)

    def test_recorder_reads_seed_from_run_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "reproducibility.json").write_text(
                json.dumps({"seed": 777}), encoding="utf-8"
            )
            metrics = parse_metrics(directory)
        self.assertEqual(metrics["seed"], "777")


if __name__ == "__main__":
    unittest.main()
