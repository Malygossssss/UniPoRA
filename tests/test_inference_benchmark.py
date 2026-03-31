import os
import unittest
from types import SimpleNamespace
from unittest import mock

import torch
from yacs.config import CfgNode as CN

import run_unipora_inference_benchmark as benchmark


def build_config(batch_size=8):
    cfg = CN()
    cfg.DATA = CN()
    cfg.DATA.BATCH_SIZE = int(batch_size)
    cfg.AMP_ENABLE = True
    cfg.freeze()
    return cfg


class InferenceBenchmarkHelpersTest(unittest.TestCase):
    def test_clone_config_with_batch_size_keeps_original(self):
        config = build_config(batch_size=12)

        cloned = benchmark.clone_config_with_batch_size(config, batch_size=1)

        self.assertEqual(config.DATA.BATCH_SIZE, 12)
        self.assertEqual(cloned.DATA.BATCH_SIZE, 1)

    def test_get_output_dir_defaults_to_checkpoint_directory(self):
        args = SimpleNamespace(
            output_dir=None,
            checkpoint=os.path.join("some", "nested", "model.pth"),
            split="val",
        )

        output_dir = benchmark.get_output_dir(args)

        self.assertTrue(output_dir.endswith(os.path.join("some", "nested", "standalone_benchmark_val")))

    def test_extract_images_from_batch_requires_multitask_batch(self):
        image = torch.randn(1, 3, 4, 4)

        self.assertTrue(torch.equal(benchmark.extract_images_from_batch({"image": image}), image))
        with self.assertRaises(TypeError):
            benchmark.extract_images_from_batch((image, None))

    def test_build_eval_loader_for_batch_size_uses_cloned_config(self):
        config = build_config(batch_size=8)

        with mock.patch(
            "run_unipora_inference_benchmark.build_mtl_eval_loader",
            return_value=("dataset", "loader"),
        ) as mocked_build:
            loader_config, loader = benchmark.build_eval_loader_for_batch_size(config, split="val", batch_size=1)

        self.assertEqual(loader, "loader")
        self.assertEqual(loader_config.DATA.BATCH_SIZE, 1)
        self.assertEqual(config.DATA.BATCH_SIZE, 8)
        mocked_build.assert_called_once()
        called_config = mocked_build.call_args[0][0]
        self.assertEqual(called_config.DATA.BATCH_SIZE, 1)

    def test_load_model_for_benchmark_uses_shared_loading_path(self):
        fake_model = object()
        logger = mock.Mock()

        with mock.patch(
            "run_unipora_inference_benchmark.build_model_for_experiment",
            return_value=fake_model,
        ) as mocked_build, mock.patch(
            "run_unipora_inference_benchmark.load_model_state"
        ) as mocked_load:
            result = benchmark.load_model_for_benchmark(build_config(), "checkpoint.pth", torch.device("cuda:0"), logger)

        self.assertIs(result, fake_model)
        mocked_build.assert_called_once()
        mocked_load.assert_called_once_with(fake_model, "checkpoint.pth", mock.ANY, logger)

    def test_build_benchmark_payload_contains_required_fields(self):
        latency = {
            "batch_size": 1,
            "input_shape": [1, 3, 448, 448],
            "total_window_ms": 25.0,
            "average_latency_ms": 0.25,
            "peak_gpu_memory_mb": 512.0,
        }
        throughput = {
            "batch_size": 8,
            "input_shape": [8, 3, 448, 448],
            "total_window_s": 0.5,
            "throughput_images_per_sec": 1600.0,
            "peak_gpu_memory_mb": 1024.0,
        }

        payload = benchmark.build_benchmark_payload(
            checkpoint_path="checkpoint.pth",
            split="val",
            device=torch.device("cuda:0"),
            amp_enabled=True,
            warmup_iters=50,
            measure_iters=100,
            latency_result=latency,
            throughput_result=throughput,
        )

        self.assertEqual(payload["latency_batch_size"], 1)
        self.assertEqual(payload["throughput_batch_size"], 8)
        self.assertEqual(payload["inference_latency_ms"], 0.25)
        self.assertEqual(payload["throughput_images_per_sec"], 1600.0)
        self.assertEqual(payload["latency_peak_gpu_memory_mb"], 512.0)
        self.assertEqual(payload["throughput_peak_gpu_memory_mb"], 1024.0)
        self.assertIn("latency", payload)
        self.assertIn("throughput", payload)

    def test_validate_iteration_count_helpers(self):
        with self.assertRaises(ValueError):
            benchmark.validate_positive_iteration_count("measure_iters", 0)
        benchmark.validate_non_negative_iteration_count("warmup_iters", 0)


if __name__ == "__main__":
    unittest.main()
