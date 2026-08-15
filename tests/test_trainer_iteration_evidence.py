import inspect
import unittest
from types import SimpleNamespace

import engine.trainer as trainer_module
from engine.trainer import _engine_epoch_length, _iteration_in_epoch


def fake_engine(global_iteration, epoch_length):
    return SimpleNamespace(state=SimpleNamespace(
        iteration=global_iteration,
        epoch_length=epoch_length,
    ))


class IgniteIterationEvidenceTest(unittest.TestCase):
    def test_iteration_in_epoch_uses_authoritative_ignite_epoch_length(self):
        loader_length = 183
        engine = fake_engine(180, 186)
        self.assertEqual(_engine_epoch_length(engine), 186)
        self.assertEqual(_iteration_in_epoch(engine), 180)
        self.assertNotEqual(_engine_epoch_length(engine), loader_length)

    def test_multi_epoch_iteration_mapping(self):
        expected = {
            1: 1,
            180: 180,
            186: 186,
            187: 1,
            372: 186,
        }
        for global_iteration, iteration_in_epoch in expected.items():
            self.assertEqual(
                _iteration_in_epoch(fake_engine(global_iteration, 186)),
                iteration_in_epoch,
            )

    def test_invalid_ignite_state_fails_closed(self):
        with self.assertRaisesRegex(RuntimeError, "epoch_length"):
            _iteration_in_epoch(fake_engine(1, 0))
        with self.assertRaisesRegex(RuntimeError, "state.iteration"):
            _iteration_in_epoch(fake_engine(0, 186))

    def test_training_logger_has_no_legacy_iteration_sources(self):
        source = inspect.getsource(trainer_module)
        self.assertNotIn("global ITER", source)
        self.assertNotIn("len(train_loader)", source)
        self.assertIn("_iteration_in_epoch(engine)", source)
        self.assertIn("_engine_epoch_length(engine)", source)


if __name__ == "__main__":
    unittest.main()
