import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import torch
from yacs.config import CfgNode as CN

from pruning.greedy import run_greedy_pruning_experiment


class FakeBackbone:
    def __init__(self, keep_indices):
        self._initial_keep_indices = self._normalize(keep_indices)
        self._keep_indices = self._normalize(keep_indices)
        self.num_layers = len(next(iter(self._keep_indices.values())))

    @staticmethod
    def _normalize(keep_indices):
        return {
            str(task): {
                int(layer_idx): list(indices) for layer_idx, indices in layer_map.items()
            }
            for task, layer_map in keep_indices.items()
        }

    def export_prompt_pruning(self):
        return self._normalize(self._keep_indices)

    def set_prompt_pruning(self, keep_indices):
        self._keep_indices = self._normalize(keep_indices)

    def collect_prompt_statistics(self, tasks):
        total_original = 0
        total_kept = 0
        for task in tasks:
            for layer_idx in range(self.num_layers):
                total_original += len(self._initial_keep_indices[task][layer_idx])
                total_kept += len(self._keep_indices[task][layer_idx])
        keep_ratio = float(total_kept) / float(total_original) if total_original > 0 else 0.0
        return {
            "total_original_tokens": total_original,
            "total_kept_tokens": total_kept,
            "total_keep_ratio": keep_ratio,
        }


class FakeModel:
    def state_dict(self):
        return {}


def build_config(output_dir, tasks, reuse_importance_within_layer):
    cfg = CN()
    cfg.MTL = True
    cfg.SEED = 0
    cfg.DETERMINISTIC = True
    cfg.TASKS = list(tasks)
    cfg.OUTPUT = output_dir
    cfg.TAG = "test"

    cfg.DATA = CN()
    cfg.DATA.BATCH_SIZE = 1
    cfg.DATA.NUM_WORKERS = 0
    cfg.DATA.PIN_MEMORY = False
    cfg.DATA.DBNAME = "PASCALContext"

    cfg.MODEL = CN()
    cfg.MODEL.RESUME = "dummy-checkpoint.pth"
    cfg.MODEL.NAME = "greedy-test-model"

    cfg.PRUNING = CN()
    cfg.PRUNING.OUTPUT_SUBDIR = "pruning"
    cfg.PRUNING.EXPERIMENT_NAME = "greedy_reuse_test"
    cfg.PRUNING.IMPORTANCE = CN()
    cfg.PRUNING.IMPORTANCE.TYPE = "base"
    cfg.PRUNING.GREEDY = CN()
    cfg.PRUNING.GREEDY.ENABLED = True
    cfg.PRUNING.GREEDY.SEARCH_VAL_RATIO = 0.10
    cfg.PRUNING.GREEDY.MIN_TOKENS_PER_LAYER = 1
    cfg.PRUNING.GREEDY.STEP_SCHEDULE = [0.5, 0.0, 0.0]
    cfg.PRUNING.GREEDY.FINAL_TOKEN_STEP = 1
    cfg.PRUNING.GREEDY.IMPORTANCE_SOURCE = "train_remainder"
    cfg.PRUNING.GREEDY.REUSE_IMPORTANCE_WITHIN_LAYER = reuse_importance_within_layer
    cfg.PRUNING.GREEDY.SAVE_IMPORTANCE_SNAPSHOTS = False
    cfg.PRUNING.GREEDY.FINAL_EVAL_SPLIT = "val"
    return cfg


def make_metrics(tasks, value=0.0):
    return {"metrics": {task: {"mIoU": float(value)} for task in tasks}}


class GreedyImportanceReuseTest(unittest.TestCase):
    def run_experiment(self, tasks, keep_indices, reuse_importance_within_layer, accept_sequence):
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = build_config(tmp_dir, tasks, reuse_importance_within_layer)
            args = SimpleNamespace(cfg="dummy.yaml")
            backbone = FakeBackbone(keep_indices)
            importance_calls = []

            def fake_compute_importance(task_config, model, data_loader, device, logger):
                task = task_config.TASKS[0]
                importance_calls.append(task)
                return {
                    "importance_type": "base",
                    "base_scores": {
                        task: {
                            layer_idx: torch.tensor(
                                [float(index + 1) for index in backbone._initial_keep_indices[task][layer_idx]],
                                dtype=torch.float32,
                            )
                            for layer_idx in range(backbone.num_layers)
                        }
                    },
                }

            def fake_collect_prompt_statistics(model, active_tasks):
                return backbone.collect_prompt_statistics(active_tasks)

            logger = mock.Mock()

            with mock.patch("pruning.greedy.set_random_seed"), mock.patch(
                "pruning.greedy.create_logger", return_value=logger
            ), mock.patch(
                "pruning.greedy.build_search_loaders",
                return_value=(
                    {"search_val_ratio": 0.10, "search_val_indices": [0], "train_remainder_indices": [1]},
                    object(),
                    object(),
                ),
            ), mock.patch(
                "pruning.greedy.build_model_for_experiment", return_value=FakeModel()
            ), mock.patch(
                "pruning.greedy.load_model_state"
            ), mock.patch(
                "pruning.greedy.ensure_prompt_backbone", return_value=backbone
            ), mock.patch(
                "pruning.greedy.compute_greedy_importance_payload",
                side_effect=fake_compute_importance,
            ), mock.patch(
                "pruning.greedy.evaluate_model",
                side_effect=lambda cfg, *_args, **_kwargs: make_metrics(cfg.TASKS, value=0.0),
            ), mock.patch(
                "pruning.greedy.evaluate_single_task",
                side_effect=lambda cfg, task, *_args, **_kwargs: (
                    {"metrics": {task: {"mIoU": 1.0}}},
                    "mIoU",
                    1.0,
                    "max",
                ),
            ), mock.patch(
                "pruning.greedy.is_metric_accepted",
                side_effect=list(accept_sequence),
            ), mock.patch(
                "pruning.greedy.collect_prompt_statistics",
                side_effect=fake_collect_prompt_statistics,
            ), mock.patch(
                "pruning.greedy.build_mtl_eval_loader",
                return_value=(None, object()),
            ):
                result = run_greedy_pruning_experiment(config, args)

        return importance_calls, result

    def test_default_mode_recomputes_after_each_accepted_round(self):
        importance_calls, result = self.run_experiment(
            tasks=["semseg"],
            keep_indices={"semseg": {0: [0, 1, 2]}},
            reuse_importance_within_layer=False,
            accept_sequence=[True, True],
        )

        self.assertEqual(importance_calls, ["semseg", "semseg"])
        self.assertFalse(result["summary"]["reuse_importance_within_layer"])

    def test_reuse_mode_computes_once_for_multiple_accepted_rounds(self):
        importance_calls, result = self.run_experiment(
            tasks=["semseg"],
            keep_indices={"semseg": {0: [0, 1, 2]}},
            reuse_importance_within_layer=True,
            accept_sequence=[True, True],
        )

        self.assertEqual(importance_calls, ["semseg"])
        self.assertTrue(result["summary"]["reuse_importance_within_layer"])

    def test_reuse_mode_resets_cache_per_task_layer_and_not_per_candidate(self):
        importance_calls, result = self.run_experiment(
            tasks=["semseg", "sal"],
            keep_indices={
                "semseg": {0: [0, 1], 1: [0, 1]},
                "sal": {0: [0, 1], 1: [0, 1]},
            },
            reuse_importance_within_layer=True,
            accept_sequence=[False, True] * 4,
        )

        self.assertEqual(importance_calls, ["semseg", "semseg", "sal", "sal"])
        self.assertEqual(result["summary"]["accepted_trials"], 4)
        self.assertTrue(result["summary"]["reuse_importance_within_layer"])


if __name__ == "__main__":
    unittest.main()
