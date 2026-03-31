import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import torch
import torch.nn as nn
from yacs.config import CfgNode as CN

from pruning.experiment import (
    build_compact_prompt_state_dict,
    export_compact_prompt_checkpoint,
    load_model_state,
)


def make_config():
    cfg = CN()
    cfg.TASKS = ["semseg", "sal"]
    cfg.MODEL = CN()
    cfg.MODEL.UPDATE_RELATIVE_POSITION = False
    cfg.MODEL.MTLORA = CN()
    cfg.MODEL.MTLORA.ENABLED = False
    cfg.MODEL.MTLORA.SPLIT_QKV = False
    return cfg


def make_tensor(shape, start):
    total = 1
    for dim in shape:
        total *= dim
    return torch.arange(start, start + total, dtype=torch.float32).view(*shape)


class FakePromptBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.prompt_config = SimpleNamespace(LOCATION="prepend", DEEP=True)
        self.share_task_prompt = False
        self.use_dynamic_prompts = False
        self.prompt_keys = ["semseg", "sal"]
        self.num_layers = 2
        self.layers = [
            SimpleNamespace(dim=2, depth=2),
            SimpleNamespace(dim=3, depth=2),
        ]
        self.prompt_embeddings = nn.ParameterDict(
            {
                "semseg": nn.Parameter(make_tensor((1, 4, 2), 0)),
                "sal": nn.Parameter(make_tensor((1, 4, 2), 100)),
            }
        )
        self.deep_prompt_embeddings = nn.ModuleList(
            [
                nn.ParameterDict(
                    {
                        "semseg": nn.Parameter(make_tensor((1, 4, 2), 200)),
                        "sal": nn.Parameter(make_tensor((1, 4, 2), 300)),
                    }
                ),
                nn.ParameterDict(
                    {
                        "semseg": nn.Parameter(make_tensor((2, 4, 3), 400)),
                        "sal": nn.Parameter(make_tensor((2, 4, 3), 500)),
                    }
                ),
            ]
        )
        self._compact_prompt_layout = False
        self._prompt_runtime_gates = None
        self.reset_prompt_pruning()

    def build_prompt_runtime_gates(self, device=None, requires_grad=True):
        return {}

    def _resolve_prompt_key(self, task):
        return task

    def _select_prompt(self, prompt_dict, task):
        return prompt_dict[task]

    def _get_prompt_token_count(self, layer_idx, task):
        if layer_idx == 0:
            return int(self.prompt_embeddings[task].shape[1])
        return int(self.deep_prompt_embeddings[layer_idx][task].shape[1])

    def reset_prompt_pruning(self):
        self._prompt_active_indices = {}
        for key in self.prompt_keys:
            self._prompt_active_indices[key] = {}
            for layer_idx in range(self.num_layers):
                self._prompt_active_indices[key][layer_idx] = torch.arange(
                    self._get_prompt_token_count(layer_idx, key), dtype=torch.long
                )

    def clear_prompt_runtime_gates(self):
        self._prompt_runtime_gates = None

    def enable_compact_prompt_layout(self, enabled=True):
        self._compact_prompt_layout = bool(enabled)
        if self._compact_prompt_layout:
            self.reset_prompt_pruning()

    def set_prompt_pruning(self, keep_indices):
        self._compact_prompt_layout = False
        normalized = {}
        for key in self.prompt_keys:
            normalized[key] = {}
            for layer_idx in range(self.num_layers):
                normalized[key][layer_idx] = torch.tensor(
                    keep_indices[key][layer_idx], dtype=torch.long
                )
        self._prompt_active_indices = normalized

    def export_prompt_pruning(self):
        return {
            key: {
                layer_idx: indices.detach().cpu().tolist()
                for layer_idx, indices in layer_map.items()
            }
            for key, layer_map in self._prompt_active_indices.items()
        }

    def _get_layer_indices(self, task, layer_idx, device):
        if self._compact_prompt_layout:
            return None
        return self._prompt_active_indices[task][layer_idx].to(device)

    def _select_layer_prompt(self, prompt_tensor, task, layer_idx):
        indices = self._get_layer_indices(task, layer_idx, prompt_tensor.device)
        if indices is not None:
            prompt_tensor = prompt_tensor.index_select(1, indices)
        return prompt_tensor

    def forward_task(self, task):
        parts = [
            self._select_layer_prompt(self.prompt_embeddings[task], task, 0).reshape(-1)
        ]
        for layer_idx in range(self.num_layers):
            parts.append(
                self._select_layer_prompt(
                    self.deep_prompt_embeddings[layer_idx][task], task, layer_idx
                ).reshape(-1)
            )
        return torch.cat(parts)


class FakePromptModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = FakePromptBackbone()
        self.decoder = nn.Linear(3, 2)

    def forward(self, task):
        return self.backbone.forward_task(task)


class CompactPromptCheckpointTest(unittest.TestCase):
    def setUp(self):
        self.keep_indices = {
            "semseg": {0: [0, 2], 1: [1, 3]},
            "sal": {0: [1, 3], 1: [0, 2]},
        }
        self.config = make_config()

    def test_build_compact_prompt_state_dict_prunes_per_task_and_stage(self):
        model = FakePromptModel()
        model.backbone.set_prompt_pruning(self.keep_indices)
        full_state = model.state_dict()

        compact_state = build_compact_prompt_state_dict(model)

        expected_semseg_prompt = full_state["backbone.prompt_embeddings.semseg"].index_select(
            1, torch.tensor([0, 2], dtype=torch.long)
        )
        expected_sal_stage1 = full_state["backbone.deep_prompt_embeddings.1.sal"].index_select(
            1, torch.tensor([0, 2], dtype=torch.long)
        )
        self.assertTrue(
            torch.equal(compact_state["backbone.prompt_embeddings.semseg"], expected_semseg_prompt)
        )
        self.assertTrue(
            torch.equal(compact_state["backbone.deep_prompt_embeddings.1.sal"], expected_sal_stage1)
        )
        self.assertEqual(compact_state["backbone.prompt_embeddings.semseg"].shape[1], 2)
        self.assertEqual(compact_state["backbone.deep_prompt_embeddings.1.sal"].shape[1], 2)

    def test_export_compact_prompt_checkpoint_writes_metadata_and_identity_mask(self):
        model = FakePromptModel()
        model.backbone.set_prompt_pruning(self.keep_indices)

        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_path = f"{tmp_dir}/compact.pth"
            export_compact_prompt_checkpoint(
                checkpoint_path,
                model,
                self.config,
                {"keep_indices": self.keep_indices},
                extra_metadata={"stage": "greedy_pruned"},
            )
            bundle = torch.load(checkpoint_path, map_location="cpu")

        self.assertTrue(bundle["compact_prompt"]["enabled"])
        self.assertEqual(bundle["compact_prompt"]["mode"], "task_specific_prepend_static")
        self.assertEqual(bundle["compact_prompt"]["source_keep_indices"], self.keep_indices)
        self.assertEqual(bundle["prompt_pruning"]["keep_indices"]["semseg"][0], [0, 1])
        self.assertEqual(bundle["prompt_pruning"]["keep_indices"]["semseg"][1], [0, 1])
        self.assertEqual(bundle["stage"], "greedy_pruned")

    def test_load_model_state_restores_compact_layout_and_matches_masked_output(self):
        source_model = FakePromptModel()
        source_model.backbone.set_prompt_pruning(self.keep_indices)

        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_path = f"{tmp_dir}/compact.pth"
            export_compact_prompt_checkpoint(
                checkpoint_path,
                source_model,
                self.config,
                {"keep_indices": self.keep_indices},
            )

            target_model = FakePromptModel()
            logger = mock.Mock()
            load_model_state(target_model, checkpoint_path, self.config, logger)

        self.assertTrue(target_model.backbone._compact_prompt_layout)
        self.assertEqual(target_model.backbone.prompt_embeddings["semseg"].shape[1], 2)
        self.assertEqual(target_model.backbone.deep_prompt_embeddings[1]["sal"].shape[1], 2)
        self.assertEqual(target_model.backbone.export_prompt_pruning()["sal"][0], [0, 1])
        self.assertEqual(target_model.backbone.export_prompt_pruning()["sal"][1], [0, 1])

        for task in self.config.TASKS:
            self.assertTrue(
                torch.equal(source_model(task), target_model(task)),
                msg=f"Compact checkpoint output mismatch for task {task}",
            )


if __name__ == "__main__":
    unittest.main()
