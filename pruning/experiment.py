import copy
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F

from data import build_loader
from evaluation.eval_edge import eval_edge_predictions
from evaluation.evaluate_utils import PerformanceMeter, get_output
from logger import create_logger
from lr_scheduler import build_scheduler
from models import build_model, build_mtl_model
from models.lora import map_old_state_dict_weights
from mtl_loss_schemes import MultiTaskLoss, get_loss
from optimizer import build_optimizer


PROMPT_PARAM_NAMES = (
    "prompt_embeddings",
    "deep_prompt_embeddings",
    "deep_prompt_pools",
    "deep_prompt_gates",
)
COMPACT_PROMPT_VERSION = 1
COMPACT_PROMPT_MODE = "task_specific_prepend_static"


def serialize_data(value):
    if isinstance(value, dict):
        return {str(k): serialize_data(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serialize_data(v) for v in value]
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return value.item()
        return value.detach().cpu().tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def save_json(path, payload):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(serialize_data(payload), handle, indent=2, ensure_ascii=False)


def set_random_seed(seed, deterministic):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    else:
        cudnn.benchmark = True


def scale_learning_rates(config):
    config = config.clone()
    config.defrost()
    linear_scaled_lr = config.TRAIN.BASE_LR * config.DATA.BATCH_SIZE / 512.0
    linear_scaled_warmup_lr = config.TRAIN.WARMUP_LR * config.DATA.BATCH_SIZE / 512.0
    linear_scaled_min_lr = config.TRAIN.MIN_LR * config.DATA.BATCH_SIZE / 512.0
    if config.TRAIN.ACCUMULATION_STEPS > 1:
        linear_scaled_lr *= config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_warmup_lr *= config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_min_lr *= config.TRAIN.ACCUMULATION_STEPS
    config.TRAIN.BASE_LR = linear_scaled_lr
    config.TRAIN.WARMUP_LR = linear_scaled_warmup_lr
    config.TRAIN.MIN_LR = linear_scaled_min_lr
    config.freeze()
    return config


def build_multitask_criterion(config):
    loss_ft = nn.ModuleDict(
        {task: get_loss(config["TASKS_CONFIG"], task, config) for task in config.TASKS}
    )
    all_loss_weights = {
        "depth": 1.0,
        "semseg": 1.0,
        "human_parts": 2.0,
        "sal": 5.0,
        "edge": 50.0,
        "normals": 10.0,
    }
    loss_weights = {task: all_loss_weights[task] for task in config.TASKS}
    return MultiTaskLoss(config.TASKS, loss_ft, loss_weights)


def build_model_for_experiment(config, device):
    backbone = build_model(config)
    model = build_mtl_model(backbone, config) if config.MTL else backbone
    model.to(device)
    return model


def ensure_prompt_backbone(model):
    backbone = getattr(model, "backbone", None)
    if backbone is None or not hasattr(backbone, "build_prompt_runtime_gates"):
        raise RuntimeError("Prompt pruning only supports MultiTaskSwin with PromptedSwinTransformer backbone.")
    if getattr(backbone, "share_task_prompt", False):
        raise RuntimeError("Prompt pruning requires task-specific prompts; shared prompt configs are not supported.")
    if getattr(backbone.prompt_config, "LOCATION", "") != "prepend":
        raise RuntimeError("Prompt pruning currently supports prepend prompts only.")
    return backbone


def ensure_compactable_prompt_backbone(model):
    backbone = ensure_prompt_backbone(model)
    if getattr(backbone, "use_dynamic_prompts", False):
        raise RuntimeError(
            "Compact prompt checkpoints only support static prompts; DYNAMIC_PROMPT=True is not supported."
        )
    return backbone


def supports_compact_prompt_checkpoint(model):
    try:
        ensure_compactable_prompt_backbone(model)
    except RuntimeError:
        return False
    return True


def get_compact_checkpoint_path(path):
    base, ext = os.path.splitext(path)
    if not ext:
        ext = ".pth"
    return f"{base}_compact{ext}"


def _extract_keep_indices(prompt_pruning):
    if prompt_pruning is None:
        return None
    if isinstance(prompt_pruning, dict) and "keep_indices" in prompt_pruning:
        return prompt_pruning["keep_indices"]
    return prompt_pruning


def _clone_state_dict_to_cpu(state_dict):
    return {key: value.detach().cpu().clone() for key, value in state_dict.items()}


def _normalize_layer_indices(indices):
    if isinstance(indices, torch.Tensor):
        return indices.detach().cpu().long()
    return torch.tensor(list(indices), dtype=torch.long)


def _get_compact_prompt_tensor_key(prompt_key, layer_idx=None):
    if layer_idx is None:
        return f"backbone.prompt_embeddings.{prompt_key}"
    return f"backbone.deep_prompt_embeddings.{layer_idx}.{prompt_key}"


def _get_compact_token_count_from_state(backbone, compact_state, prompt_key, layer_idx):
    if layer_idx == 0:
        return int(compact_state[_get_compact_prompt_tensor_key(prompt_key)].shape[1])
    if not getattr(backbone.prompt_config, "DEEP", False):
        return int(compact_state[_get_compact_prompt_tensor_key(prompt_key)].shape[1])
    return int(compact_state[_get_compact_prompt_tensor_key(prompt_key, layer_idx)].shape[1])


def build_identity_prompt_pruning_from_state_dict(model, compact_state):
    backbone = ensure_compactable_prompt_backbone(model)
    keep_indices = {}
    for prompt_key in backbone.prompt_keys:
        keep_indices[prompt_key] = {}
        for layer_idx in range(backbone.num_layers):
            token_count = _get_compact_token_count_from_state(
                backbone, compact_state, prompt_key, layer_idx
            )
            keep_indices[prompt_key][layer_idx] = list(range(token_count))
    return keep_indices


def build_compact_prompt_state_dict(model, keep_indices=None):
    backbone = ensure_compactable_prompt_backbone(model)
    active_keep_indices = (
        clone_keep_indices(keep_indices)
        if keep_indices is not None
        else clone_keep_indices(backbone.export_prompt_pruning())
    )
    compact_state = _clone_state_dict_to_cpu(model.state_dict())

    for prompt_key in backbone.prompt_keys:
        stage0_indices = _normalize_layer_indices(active_keep_indices[prompt_key][0])
        prompt_state_key = _get_compact_prompt_tensor_key(prompt_key)
        if prompt_state_key not in compact_state:
            raise RuntimeError(f"Missing prompt tensor in model state: {prompt_state_key}")
        compact_state[prompt_state_key] = compact_state[prompt_state_key].index_select(1, stage0_indices)

        if getattr(backbone.prompt_config, "DEEP", False):
            deep_prompt_embeddings = getattr(backbone, "deep_prompt_embeddings", None)
            if deep_prompt_embeddings is None:
                raise RuntimeError(
                    "Compact prompt checkpoints require static deep_prompt_embeddings for deep prompt models."
                )
            for layer_idx in range(backbone.num_layers):
                deep_state_key = _get_compact_prompt_tensor_key(prompt_key, layer_idx)
                if deep_state_key not in compact_state:
                    raise RuntimeError(f"Missing deep prompt tensor in model state: {deep_state_key}")
                layer_indices = _normalize_layer_indices(active_keep_indices[prompt_key][layer_idx])
                compact_state[deep_state_key] = compact_state[deep_state_key].index_select(1, layer_indices)

    return compact_state


def _replace_parameter(parameter_container, key, tensor):
    current_param = parameter_container[key]
    new_tensor = tensor.detach().to(device=current_param.device, dtype=current_param.dtype).clone()
    parameter_container[key] = nn.Parameter(new_tensor, requires_grad=current_param.requires_grad)


def prepare_model_for_compact_prompt_load(model, checkpoint_state):
    backbone = ensure_compactable_prompt_backbone(model)
    for prompt_key in backbone.prompt_keys:
        prompt_state_key = _get_compact_prompt_tensor_key(prompt_key)
        if prompt_state_key not in checkpoint_state:
            raise RuntimeError(f"Compact checkpoint is missing {prompt_state_key}.")
        _replace_parameter(backbone.prompt_embeddings, prompt_key, checkpoint_state[prompt_state_key])

    if getattr(backbone.prompt_config, "DEEP", False):
        deep_prompt_embeddings = getattr(backbone, "deep_prompt_embeddings", None)
        if deep_prompt_embeddings is None:
            raise RuntimeError(
                "Compact prompt checkpoints require static deep_prompt_embeddings for deep prompt models."
            )
        for layer_idx in range(backbone.num_layers):
            for prompt_key in backbone.prompt_keys:
                deep_state_key = _get_compact_prompt_tensor_key(prompt_key, layer_idx)
                if deep_state_key not in checkpoint_state:
                    raise RuntimeError(f"Compact checkpoint is missing {deep_state_key}.")
                _replace_parameter(
                    backbone.deep_prompt_embeddings[layer_idx],
                    prompt_key,
                    checkpoint_state[deep_state_key],
                )

    backbone.reset_prompt_pruning()
    backbone.clear_prompt_runtime_gates()
    return backbone


def export_compact_prompt_checkpoint(path, model, config, prompt_pruning, extra_metadata=None):
    backbone = ensure_compactable_prompt_backbone(model)
    source_keep_indices = _extract_keep_indices(prompt_pruning)
    if source_keep_indices is None:
        source_keep_indices = backbone.export_prompt_pruning()
    source_keep_indices = clone_keep_indices(source_keep_indices)
    source_prompt_statistics = collect_prompt_statistics(model, config.TASKS)
    compact_state = build_compact_prompt_state_dict(model, source_keep_indices)
    identity_keep_indices = build_identity_prompt_pruning_from_state_dict(model, compact_state)

    bundle = {
        "model": compact_state,
        "prompt_pruning": {"keep_indices": identity_keep_indices},
        "compact_prompt": {
            "enabled": True,
            "version": COMPACT_PROMPT_VERSION,
            "mode": COMPACT_PROMPT_MODE,
            "source_keep_indices": source_keep_indices,
            "source_prompt_statistics": source_prompt_statistics,
        },
        "config": config.dump(),
    }
    if extra_metadata:
        bundle.update(extra_metadata)
    torch.save(bundle, path)
    return bundle


def maybe_export_compact_prompt_checkpoint(path, model, config, prompt_pruning, logger=None, extra_metadata=None):
    try:
        export_compact_prompt_checkpoint(path, model, config, prompt_pruning, extra_metadata=extra_metadata)
    except RuntimeError as exc:
        if logger is not None:
            logger.info("Skipping compact prompt checkpoint export for %s: %s", path, exc)
        return None
    if logger is not None:
        logger.info("Saved compact prompt checkpoint to %s", path)
    return path


def prepare_state_dict_for_load(config, model, checkpoint_state, logger):
    model_state = dict(checkpoint_state)

    attn_mask_keys = [key for key in model_state.keys() if "attn_mask" in key]
    for key in attn_mask_keys:
        del model_state[key]

    if config.MODEL.UPDATE_RELATIVE_POSITION:
        drop_keys = [
            key
            for key in model_state.keys()
            if "relative_position_index" in key or "relative_coords_table" in key
        ]
        for key in drop_keys:
            del model_state[key]

    if config.MODEL.MTLORA.ENABLED:
        mapping = {}
        trainable_layers = []
        mtlora = config.MODEL.MTLORA
        if mtlora.QKV_ENABLED:
            trainable_layers.extend(["attn.qkv.weight", "attn.qkv.bias"])
        if mtlora.PROJ_ENABLED:
            trainable_layers.extend(["attn.proj.weight", "attn.proj.bias"])
        if mtlora.FC1_ENABLED:
            trainable_layers.extend(["mlp.fc1.weight", "mlp.fc1.bias"])
        if mtlora.FC2_ENABLED:
            trainable_layers.extend(["mlp.fc2.weight", "mlp.fc2.bias"])
        if mtlora.DOWNSAMPLER_ENABLED:
            trainable_layers.extend(["downsample.reduction.weight"])

        for key in list(model_state.keys()):
            last_three = ".".join(key.split(".")[-3:])
            prefix = ".".join(key.split(".")[:-3])
            if last_three in trainable_layers:
                weight_bias = last_three.split(".")[-1]
                layer_name = ".".join(last_three.split(".")[:-1])
                mapping[f"{prefix}.{layer_name}.{weight_bias}"] = (
                    f"{prefix}.{layer_name}.linear.{weight_bias}"
                )
        if mapping:
            model_state = map_old_state_dict_weights(
                model_state, mapping, "", config.MODEL.MTLORA.SPLIT_QKV
            )

    incompatible = model.load_state_dict(model_state, strict=False)
    missing = getattr(incompatible, "missing_keys", [])
    unexpected = getattr(incompatible, "unexpected_keys", [])
    if missing:
        logger.warning("Missing keys while loading checkpoint: %s", missing)
    if unexpected:
        logger.warning("Unexpected keys while loading checkpoint: %s", unexpected)


def load_model_state(model, checkpoint_path, config, logger):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    checkpoint_state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    compact_prompt = checkpoint.get("compact_prompt") if isinstance(checkpoint, dict) else None
    is_compact_prompt = isinstance(compact_prompt, dict) and bool(compact_prompt.get("enabled", False))
    if is_compact_prompt:
        compact_version = int(compact_prompt.get("version", 0))
        if compact_version != COMPACT_PROMPT_VERSION:
            raise RuntimeError(
                f"Unsupported compact prompt checkpoint version: {compact_version}. Expected {COMPACT_PROMPT_VERSION}."
            )
        compact_mode = compact_prompt.get("mode")
        if compact_mode != COMPACT_PROMPT_MODE:
            raise RuntimeError(
                f"Unsupported compact prompt checkpoint mode: {compact_mode}. Expected {COMPACT_PROMPT_MODE}."
            )
        prepare_model_for_compact_prompt_load(model, checkpoint_state)
    prepare_state_dict_for_load(config, model, checkpoint_state, logger)
    prompt_pruning = None
    if is_compact_prompt:
        backbone = ensure_compactable_prompt_backbone(model)
        backbone.reset_prompt_pruning()
        backbone.clear_prompt_runtime_gates()
        backbone.enable_compact_prompt_layout(True)
        return checkpoint
    if isinstance(checkpoint, dict):
        prompt_pruning = checkpoint.get("prompt_pruning")
        if prompt_pruning is None and isinstance(checkpoint.get("extra_state"), dict):
            prompt_pruning = checkpoint["extra_state"].get("prompt_pruning")
    if prompt_pruning:
        pruning_state = prompt_pruning["keep_indices"] if "keep_indices" in prompt_pruning else prompt_pruning
        ensure_prompt_backbone(model).set_prompt_pruning(pruning_state)
    return checkpoint


def save_model_state(path, model, config, prompt_pruning, extra_metadata=None):
    bundle = {
        "model": model.state_dict(),
        "prompt_pruning": prompt_pruning,
        "config": config.dump(),
    }
    if extra_metadata:
        bundle.update(extra_metadata)
    torch.save(bundle, path)


def get_experiment_output_dir(config, cfg_path):
    experiment_name = config.PRUNING.EXPERIMENT_NAME or Path(cfg_path).stem
    return os.path.join(config.OUTPUT, config.PRUNING.OUTPUT_SUBDIR, experiment_name)


def batch_to_device(batch, tasks, device):
    images = batch["image"].to(device, non_blocking=True)
    targets = {task: batch[task].to(device, non_blocking=True) for task in tasks}
    return images, targets


def get_prompt_stage_factor(backbone, layer_idx):
    if not getattr(backbone.prompt_config, "DEEP", False):
        return 1 if layer_idx == 0 else 0
    depth = backbone.layers[layer_idx].depth
    if layer_idx == 0 and getattr(backbone.prompt_config, "LOCATION", "") == "prepend":
        return depth
    return depth


def collect_prompt_statistics(model, tasks):
    backbone = ensure_prompt_backbone(model)
    keep_indices = backbone.export_prompt_pruning()
    per_layer = []
    total_original_tokens = 0
    total_kept_tokens = 0
    total_prompt_params = 0
    total_active_prompt_params = 0

    for task in tasks:
        for layer_idx in range(backbone.num_layers):
            original_tokens = int(backbone._get_prompt_token_count(layer_idx, task))
            kept_tokens = len(keep_indices[task][layer_idx])
            prompt_dim = int(backbone.layers[layer_idx].dim)
            factor = get_prompt_stage_factor(backbone, layer_idx)
            original_params = original_tokens * prompt_dim * factor
            kept_params = kept_tokens * prompt_dim * factor
            total_original_tokens += original_tokens
            total_kept_tokens += kept_tokens
            total_prompt_params += original_params
            total_active_prompt_params += kept_params
            per_layer.append(
                {
                    "task": task,
                    "layer": layer_idx,
                    "original_tokens": original_tokens,
                    "kept_tokens": kept_tokens,
                    "pruned_tokens": original_tokens - kept_tokens,
                    "keep_ratio": kept_tokens / max(original_tokens, 1),
                    "prompt_dim": prompt_dim,
                    "effective_prompt_params": kept_params,
                    "original_prompt_params": original_params,
                }
            )

    total_params = sum(param.numel() for param in model.parameters())
    effective_total_params = total_params - total_prompt_params + total_active_prompt_params
    return {
        "per_layer": per_layer,
        "total_original_tokens": total_original_tokens,
        "total_kept_tokens": total_kept_tokens,
        "total_keep_ratio": total_kept_tokens / max(total_original_tokens, 1),
        "total_params": total_params,
        "effective_total_params": effective_total_params,
        "original_prompt_params": total_prompt_params,
        "effective_prompt_params": total_active_prompt_params,
    }


def evaluate_model(config, data_loader, model, device, logger, output_dir, split_name):
    criterion = build_multitask_criterion(config)
    performance_meter = PerformanceMeter(config, config.DATA.DBNAME)
    loss_sum = 0.0
    batch_count = 0
    edge_eval_dir = None
    if "edge" in config.TASKS and config.DATA.DBNAME == "NYUD":
        edge_eval_dir = os.path.join(output_dir, "__edge_eval_cache__", split_name)
        os.makedirs(os.path.join(edge_eval_dir, "edge"), exist_ok=True)

    model.eval()
    try:
        with torch.no_grad():
            for batch in data_loader:
                images, targets = batch_to_device(batch, config.TASKS, device)
                outputs = model(images)
                loss, _ = criterion(outputs, targets)
                loss_sum += float(loss.item())
                batch_count += 1
                processed_output = {task: get_output(outputs[task], task) for task in config.TASKS}
                performance_meter.update(processed_output, targets)
                if edge_eval_dir is not None:
                    edge_probs = torch.sigmoid(outputs["edge"].detach()).float().cpu().numpy()
                    image_ids = batch["meta"]["image"]
                    image_ids = [image_ids] if isinstance(image_ids, str) else list(image_ids)
                    for idx, image_id in enumerate(image_ids):
                        np.save(
                            os.path.join(edge_eval_dir, "edge", f"{image_id}.npy"),
                            edge_probs[idx, 0].astype(np.float32),
                        )

        if edge_eval_dir is not None:
            formal_edge_results = eval_edge_predictions(
                database="NYUD",
                save_dir=edge_eval_dir,
                gt_root=config.DATA.DATA_PATH,
                write_outputs=False,
            )
            performance_meter.meters["edge"].set_formal_results(formal_edge_results)
        metrics = performance_meter.get_score(verbose=True)
    finally:
        if edge_eval_dir and os.path.isdir(edge_eval_dir):
            import shutil

            shutil.rmtree(edge_eval_dir, ignore_errors=True)

    summary = {"loss": loss_sum / max(batch_count, 1), "metrics": metrics}
    logger.info("%s loss %.6f", split_name, summary["loss"])
    return summary


def get_loader_subset(config, train_loader, val_loader):
    source = str(config.PRUNING.IMPORTANCE.SOURCE).lower()
    if source == "train":
        return train_loader
    if source == "val":
        return val_loader
    raise ValueError(f"Unsupported importance source: {config.PRUNING.IMPORTANCE.SOURCE}")


def init_score_dict(backbone, tasks):
    scores = {}
    for task in tasks:
        scores[task] = {}
        for layer_idx in range(backbone.num_layers):
            token_count = backbone._get_prompt_token_count(layer_idx, task)
            scores[task][layer_idx] = torch.zeros(token_count, dtype=torch.float64)
    return scores


def clone_keep_indices(keep_indices):
    return {
        task: {int(layer_idx): list(indices) for layer_idx, indices in layer_map.items()}
        for task, layer_map in keep_indices.items()
    }


def resolve_task_prune_ratios(config):
    default_ratio = float(config.PRUNING.PRUNER.RATIO)
    task_ratio_cfg = getattr(config.PRUNING.PRUNER, "TASK_RATIOS", {})
    ratio_overrides = {str(task): float(value) for task, value in task_ratio_cfg.items()}
    unknown_tasks = sorted(set(ratio_overrides.keys()) - set(config.TASKS))
    if unknown_tasks:
        raise RuntimeError(f"Unknown tasks in PRUNING.PRUNER.TASK_RATIOS: {unknown_tasks}")

    task_ratios = {}
    for task in config.TASKS:
        ratio = ratio_overrides.get(task, default_ratio)
        if ratio < 0.0 or ratio > 1.0:
            raise RuntimeError(f"Invalid prune ratio {ratio} for task {task}. Expected [0, 1].")
        task_ratios[task] = ratio
    return task_ratios


def resolve_recovery_task_selection(config):
    all_tasks = list(config.TASKS)

    def normalize(task_list, label, default_tasks):
        selected = list(default_tasks) if len(task_list) == 0 else [str(task) for task in task_list]
        unknown_tasks = sorted(set(selected) - set(all_tasks))
        if unknown_tasks:
            raise RuntimeError(f"Unknown tasks in PRUNING.RECOVERY.{label}: {unknown_tasks}")
        return selected

    prompt_tasks = normalize(
        list(config.PRUNING.RECOVERY.PROMPT_TASKS),
        "PROMPT_TASKS",
        all_tasks if config.PRUNING.RECOVERY.TRAIN_TA_PROMPT else [],
    )
    head_tasks = normalize(
        list(config.PRUNING.RECOVERY.HEAD_TASKS),
        "HEAD_TASKS",
        all_tasks if config.PRUNING.RECOVERY.TRAIN_TASK_HEADS else [],
    )
    loss_tasks = normalize(
        list(config.PRUNING.RECOVERY.LOSS_TASKS),
        "LOSS_TASKS",
        all_tasks,
    )
    distill_default = loss_tasks
    distill_tasks = normalize(
        list(config.PRUNING.RECOVERY.DISTILL_TASKS),
        "DISTILL_TASKS",
        distill_default,
    )
    return {
        "prompt_tasks": prompt_tasks,
        "head_tasks": head_tasks,
        "loss_tasks": loss_tasks,
        "distill_tasks": distill_tasks,
    }


def set_prompt_task_trainable(backbone, task, requires_grad):
    touched = False
    for attr_name in ("prompt_embeddings", "prompt_embeddings_tb", "prompt_embeddings_lr"):
        prompt_dict = getattr(backbone, attr_name, None)
        if prompt_dict is not None and task in prompt_dict:
            prompt_dict[task].requires_grad = requires_grad
            touched = True

    deep_prompt_embeddings = getattr(backbone, "deep_prompt_embeddings", None)
    if deep_prompt_embeddings is not None:
        for prompt_dict in deep_prompt_embeddings:
            if task in prompt_dict:
                prompt_dict[task].requires_grad = requires_grad
                touched = True
    return touched


def set_task_head_trainable(model, task, requires_grad):
    decoder_group = getattr(model, "decoders", None)
    task_decoders = getattr(decoder_group, "decoders", None)
    if task_decoders is None or task not in task_decoders:
        return False
    for parameter in task_decoders[task].parameters():
        parameter.requires_grad = requires_grad
    return True


def compute_selected_task_loss(criterion, outputs, targets, tasks, device):
    if len(tasks) == 0:
        return torch.tensor(0.0, device=device), {}

    task_loss_map = {task: criterion.loss_ft[task](outputs[task], targets[task]) for task in tasks}
    total_loss = torch.sum(
        torch.stack([criterion.loss_weights[task] * task_loss_map[task] for task in tasks])
    )
    return total_loss, task_loss_map


def compute_base_importance(config, model, data_loader, device, logger):
    backbone = ensure_prompt_backbone(model)
    num_batches = int(config.PRUNING.IMPORTANCE.NUM_BATCHES)
    score_sums = init_score_dict(backbone, config.TASKS)
    score_counts = 0
    task_losses = build_multitask_criterion(config).loss_ft

    model.eval()
    for batch_idx, batch in enumerate(data_loader):
        if batch_idx >= num_batches:
            break
        images, targets = batch_to_device(batch, config.TASKS, device)
        runtime_gates = backbone.build_prompt_runtime_gates(device=device, requires_grad=True)
        backbone.set_prompt_runtime_gates(runtime_gates)
        model.zero_grad(set_to_none=True)
        outputs = model(images)
        losses = {
            task: task_losses[task](outputs[task], targets[task])
            for task in config.TASKS
        }
        for task_index, task in enumerate(config.TASKS):
            gate_list = [runtime_gates[task][layer_idx] for layer_idx in range(backbone.num_layers)]
            grads = torch.autograd.grad(
                losses[task],
                gate_list,
                retain_graph=task_index < len(config.TASKS) - 1,
                allow_unused=True,
            )
            for layer_idx, grad in enumerate(grads):
                if grad is None:
                    continue
                score_sums[task][layer_idx] += grad.detach().abs().cpu().double()
        backbone.clear_prompt_runtime_gates()
        score_counts += 1

    if score_counts == 0:
        raise RuntimeError("Importance computation did not process any batches.")

    base_scores = {}
    for task in config.TASKS:
        base_scores[task] = {}
        for layer_idx in range(backbone.num_layers):
            base_scores[task][layer_idx] = (score_sums[task][layer_idx] / score_counts).float()

    logger.info("Computed base importance scores on %d batches.", score_counts)
    return base_scores


def compute_ta_lora_replacement_deltas(config, model, data_loader, device, logger):
    backbone = ensure_prompt_backbone(model)
    if not hasattr(backbone, "disable_task_lora"):
        raise RuntimeError("Stage3 TA-LoRA replacement scoring requires backbone.disable_task_lora(...).")

    num_batches = int(config.PRUNING.IMPORTANCE.NUM_BATCHES)
    task_losses = build_multitask_criterion(config).loss_ft
    delta_on_sums = init_score_dict(backbone, config.TASKS)
    delta_off_sums = init_score_dict(backbone, config.TASKS)
    counts = 0
    original_keep_indices = clone_keep_indices(backbone.export_prompt_pruning())

    model.eval()
    try:
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                if batch_idx >= num_batches:
                    break

                images, targets = batch_to_device(batch, config.TASKS, device)
                full_outputs = model(images)
                full_losses = {
                    task: float(task_losses[task](full_outputs[task], targets[task]).item())
                    for task in config.TASKS
                }
                active_keep_indices = clone_keep_indices(backbone.export_prompt_pruning())
                task_lora_off_losses = {
                    task: {layer_idx: 0.0 for layer_idx in range(backbone.num_layers)}
                    for task in config.TASKS
                }

                for task in config.TASKS:
                    for layer_idx in range(backbone.num_layers):
                        with backbone.disable_task_lora(layer_idx, task):
                            lora_off_outputs = model(images)
                        task_lora_off_losses[task][layer_idx] = float(
                            task_losses[task](lora_off_outputs[task], targets[task]).item()
                        )

                for task in config.TASKS:
                    for layer_idx in range(backbone.num_layers):
                        active_indices = list(active_keep_indices[task][layer_idx])
                        if len(active_indices) <= 1:
                            continue
                        for token_idx in active_indices:
                            runtime_gates = backbone.build_prompt_runtime_gates(
                                device=device, requires_grad=False
                            )
                            runtime_gates[task][layer_idx][token_idx] = 0.0
                            backbone.set_prompt_runtime_gates(runtime_gates)
                            token_off_outputs = model(images)
                            token_off_loss = float(
                                task_losses[task](token_off_outputs[task], targets[task]).item()
                            )
                            delta_on_sums[task][layer_idx][token_idx] += token_off_loss - full_losses[task]
                            backbone.clear_prompt_runtime_gates()

                            runtime_gates = backbone.build_prompt_runtime_gates(
                                device=device, requires_grad=False
                            )
                            runtime_gates[task][layer_idx][token_idx] = 0.0
                            backbone.set_prompt_runtime_gates(runtime_gates)
                            with backbone.disable_task_lora(layer_idx, task):
                                token_and_lora_off_outputs = model(images)
                            token_and_lora_off_loss = float(
                                task_losses[task](token_and_lora_off_outputs[task], targets[task]).item()
                            )
                            delta_off_sums[task][layer_idx][token_idx] += (
                                token_and_lora_off_loss - task_lora_off_losses[task][layer_idx]
                            )
                            backbone.clear_prompt_runtime_gates()

                counts += 1
    finally:
        backbone.set_prompt_pruning(original_keep_indices)
        backbone.clear_prompt_runtime_gates()

    if counts == 0:
        raise RuntimeError("Stage3 TA-LoRA replacement computation did not process any batches.")

    logger.info("Computed stage3 TA-LoRA replacement deltas on %d batches.", counts)
    return (
        {
            task: {
                layer_idx: (delta_on_sums[task][layer_idx] / counts).float()
                for layer_idx in range(backbone.num_layers)
            }
            for task in config.TASKS
        },
        {
            task: {
                layer_idx: (delta_off_sums[task][layer_idx] / counts).float()
                for layer_idx in range(backbone.num_layers)
            }
            for task in config.TASKS
        },
    )


def compute_ga_delta_losses(config, model, data_loader, device, logger):
    backbone = ensure_prompt_backbone(model)
    num_batches = int(config.PRUNING.IMPORTANCE.NUM_BATCHES)
    task_losses = build_multitask_criterion(config).loss_ft
    delta_sums = {
        task: {layer_idx: 0.0 for layer_idx in range(backbone.num_layers)}
        for task in config.TASKS
    }
    counts = 0

    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            if batch_idx >= num_batches:
                break
            images, targets = batch_to_device(batch, config.TASKS, device)
            full_outputs = model(images)
            full_losses = {
                task: float(task_losses[task](full_outputs[task], targets[task]).item())
                for task in config.TASKS
            }
            for layer_idx in range(backbone.num_layers):
                with backbone.disable_shared_lora(layer_idx):
                    dropped_outputs = model(images)
                for task in config.TASKS:
                    dropped_loss = float(task_losses[task](dropped_outputs[task], targets[task]).item())
                    delta_sums[task][layer_idx] += dropped_loss - full_losses[task]
            counts += 1

    if counts == 0:
        raise RuntimeError("GA delta computation did not process any batches.")

    logger.info("Computed GA delta losses on %d batches.", counts)
    return {
        task: {
            layer_idx: delta_sums[task][layer_idx] / counts
            for layer_idx in range(backbone.num_layers)
        }
        for task in config.TASKS
    }


def compute_token_delta_losses(config, model, data_loader, device, logger):
    backbone = ensure_prompt_backbone(model)
    num_batches = int(config.PRUNING.IMPORTANCE.NUM_BATCHES)
    task_losses = build_multitask_criterion(config).loss_ft
    delta_sums = init_score_dict(backbone, config.TASKS)
    counts = 0
    original_keep_indices = clone_keep_indices(backbone.export_prompt_pruning())

    model.eval()
    try:
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                if batch_idx >= num_batches:
                    break
                images, targets = batch_to_device(batch, config.TASKS, device)
                full_outputs = model(images)
                full_losses = {
                    task: float(task_losses[task](full_outputs[task], targets[task]).item())
                    for task in config.TASKS
                }
                base_keep_indices = clone_keep_indices(backbone.export_prompt_pruning())

                for task in config.TASKS:
                    for layer_idx in range(backbone.num_layers):
                        active_indices = list(base_keep_indices[task][layer_idx])
                        if len(active_indices) <= 1:
                            continue
                        for token_idx in active_indices:
                            dropped_keep_indices = copy.deepcopy(base_keep_indices)
                            dropped_keep_indices[task][layer_idx] = [
                                index for index in active_indices if index != token_idx
                            ]
                            if not dropped_keep_indices[task][layer_idx]:
                                continue
                            backbone.set_prompt_pruning(dropped_keep_indices)
                            dropped_outputs = model(images)
                            dropped_loss = float(
                                task_losses[task](dropped_outputs[task], targets[task]).item()
                            )
                            delta_sums[task][layer_idx][token_idx] += dropped_loss - full_losses[task]
                            backbone.set_prompt_pruning(base_keep_indices)
                counts += 1
    finally:
        backbone.set_prompt_pruning(original_keep_indices)

    if counts == 0:
        raise RuntimeError("Token delta computation did not process any batches.")

    logger.info("Computed token delta losses on %d batches.", counts)
    return {
        task: {
            layer_idx: (delta_sums[task][layer_idx] / counts).float()
            for layer_idx in range(backbone.num_layers)
        }
        for task in config.TASKS
    }


def build_ga_scores(base_scores, ga_delta_losses, token_delta_losses, eps):
    ga_scores = {}
    for task, layer_map in base_scores.items():
        ga_scores[task] = {}
        for layer_idx, scores in layer_map.items():
            denom = max(float(ga_delta_losses[task][layer_idx]), 0.0) + eps
            token_delta = torch.clamp_min(token_delta_losses[task][layer_idx], 0.0)
            ga_scores[task][layer_idx] = scores * token_delta / denom
    return ga_scores


def normalize_stage_delta(delta_tensor, eps):
    delta_pos = torch.clamp_min(delta_tensor.float(), 0.0)
    stage_mean = float(delta_pos.mean().item())
    if stage_mean <= 0.0:
        return delta_pos, torch.zeros_like(delta_pos)
    return delta_pos, delta_pos / (stage_mean + eps)


def build_ta_replacement_scores(base_scores, delta_on_losses, delta_off_losses, beta, eps, logger=None):
    final_scores = {}
    delta_on_pos = {}
    delta_off_pos = {}
    replacement_scores = {}
    stage_summaries = []

    for task, layer_map in base_scores.items():
        final_scores[task] = {}
        delta_on_pos[task] = {}
        delta_off_pos[task] = {}
        replacement_scores[task] = {}
        for layer_idx, scores in layer_map.items():
            on_pos, on_norm = normalize_stage_delta(delta_on_losses[task][layer_idx], eps)
            off_pos, off_norm = normalize_stage_delta(delta_off_losses[task][layer_idx], eps)
            replacement = torch.relu(off_norm - on_norm)
            final = scores.float() * torch.exp(-beta * replacement)

            delta_on_pos[task][layer_idx] = on_pos
            delta_off_pos[task][layer_idx] = off_pos
            replacement_scores[task][layer_idx] = replacement
            final_scores[task][layer_idx] = final

            summary = {
                "task": task,
                "layer": layer_idx,
                "mean_base_score": float(scores.float().mean().item()),
                "mean_delta_on_pos": float(on_pos.mean().item()),
                "mean_delta_off_pos": float(off_pos.mean().item()),
                "mean_replacement": float(replacement.mean().item()),
                "mean_final_score": float(final.mean().item()),
            }
            stage_summaries.append(summary)
            if logger is not None:
                logger.info(
                    "Stage3 TA replacement [%s][L%d]: base=%.6f on=%.6f off=%.6f R=%.6f final=%.6f",
                    task,
                    layer_idx,
                    summary["mean_base_score"],
                    summary["mean_delta_on_pos"],
                    summary["mean_delta_off_pos"],
                    summary["mean_replacement"],
                    summary["mean_final_score"],
                )

    return {
        "ga_scores": final_scores,
        "ta_delta_on_losses": delta_on_losses,
        "ta_delta_off_losses": delta_off_losses,
        "ta_delta_on_pos": delta_on_pos,
        "ta_delta_off_pos": delta_off_pos,
        "ta_replacement_scores": replacement_scores,
        "stage3_ta_summary": stage_summaries,
    }


def scores_to_records(score_payload):
    records = []
    base_scores = score_payload["base_scores"]
    ga_scores = score_payload.get("ga_scores")
    ga_delta_losses = score_payload.get("ga_delta_losses", {})
    token_delta_losses = score_payload.get("token_delta_losses", {})
    ta_delta_on_losses = score_payload.get("ta_delta_on_losses", {})
    ta_delta_off_losses = score_payload.get("ta_delta_off_losses", {})
    ta_delta_on_pos = score_payload.get("ta_delta_on_pos", {})
    ta_delta_off_pos = score_payload.get("ta_delta_off_pos", {})
    ta_replacement_scores = score_payload.get("ta_replacement_scores", {})
    for task, layer_map in base_scores.items():
        for layer_idx, base_tensor in layer_map.items():
            ga_tensor = ga_scores[task][layer_idx] if ga_scores is not None else None
            token_delta_tensor = (
                token_delta_losses[task][layer_idx]
                if token_delta_losses is not None and task in token_delta_losses and layer_idx in token_delta_losses[task]
                else None
            )
            ta_delta_on_tensor = (
                ta_delta_on_losses[task][layer_idx]
                if ta_delta_on_losses is not None and task in ta_delta_on_losses and layer_idx in ta_delta_on_losses[task]
                else None
            )
            ta_delta_off_tensor = (
                ta_delta_off_losses[task][layer_idx]
                if ta_delta_off_losses is not None and task in ta_delta_off_losses and layer_idx in ta_delta_off_losses[task]
                else None
            )
            ta_delta_on_pos_tensor = (
                ta_delta_on_pos[task][layer_idx]
                if ta_delta_on_pos is not None and task in ta_delta_on_pos and layer_idx in ta_delta_on_pos[task]
                else None
            )
            ta_delta_off_pos_tensor = (
                ta_delta_off_pos[task][layer_idx]
                if ta_delta_off_pos is not None and task in ta_delta_off_pos and layer_idx in ta_delta_off_pos[task]
                else None
            )
            ta_replacement_tensor = (
                ta_replacement_scores[task][layer_idx]
                if ta_replacement_scores is not None and task in ta_replacement_scores and layer_idx in ta_replacement_scores[task]
                else None
            )
            for token_idx, base_score in enumerate(base_tensor.tolist()):
                record = {
                    "task": task,
                    "layer": layer_idx,
                    "token_id": token_idx,
                    "base_score": float(base_score),
                }
                if ga_tensor is not None:
                    record["ga_score"] = float(ga_tensor[token_idx].item())
                    if ga_delta_losses and task in ga_delta_losses and layer_idx in ga_delta_losses[task]:
                        record["ga_delta_loss"] = float(ga_delta_losses[task][layer_idx])
                if token_delta_tensor is not None:
                    record["token_delta_loss"] = float(token_delta_tensor[token_idx].item())
                if ta_delta_on_tensor is not None:
                    record["ta_delta_on_loss"] = float(ta_delta_on_tensor[token_idx].item())
                if ta_delta_off_tensor is not None:
                    record["ta_delta_off_loss"] = float(ta_delta_off_tensor[token_idx].item())
                if ta_delta_on_pos_tensor is not None:
                    record["ta_delta_on_pos"] = float(ta_delta_on_pos_tensor[token_idx].item())
                if ta_delta_off_pos_tensor is not None:
                    record["ta_delta_off_pos"] = float(ta_delta_off_pos_tensor[token_idx].item())
                if ta_replacement_tensor is not None:
                    record["ta_replacement"] = float(ta_replacement_tensor[token_idx].item())
                records.append(record)
    return records


def save_importance_scores(output_dir, score_payload):
    pth_path = os.path.join(output_dir, "importance_scores.pth")
    json_path = os.path.join(output_dir, "importance_scores.json")
    torch.save(score_payload, pth_path)
    save_json(json_path, {"records": scores_to_records(score_payload), **score_payload})
    return pth_path, json_path


def load_importance_scores(path):
    if path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    return torch.load(path, map_location="cpu")


def tensorize_score_payload(score_payload):
    converted = dict(score_payload)
    for key in (
        "base_scores",
        "ga_scores",
        "token_delta_losses",
        "ta_delta_on_losses",
        "ta_delta_off_losses",
        "ta_delta_on_pos",
        "ta_delta_off_pos",
        "ta_replacement_scores",
    ):
        if key not in converted or converted[key] is None:
            continue
        converted[key] = {
            task: {
                int(layer_idx): (
                    tensor if isinstance(tensor, torch.Tensor) else torch.tensor(tensor, dtype=torch.float32)
                )
                for layer_idx, tensor in layer_map.items()
            }
            for task, layer_map in converted[key].items()
        }
    if "ga_delta_losses" in converted and converted["ga_delta_losses"] is not None:
        converted["ga_delta_losses"] = {
            task: {int(layer_idx): float(value) for layer_idx, value in layer_map.items()}
            for task, layer_map in converted["ga_delta_losses"].items()
        }
    return converted


def resolve_importance_payload(config, model, loader, device, logger, output_dir):
    load_path = config.PRUNING.IMPORTANCE.LOAD_PATH
    if load_path:
        logger.info("Loading importance scores from %s", load_path)
        payload = tensorize_score_payload(load_importance_scores(load_path))
        if str(config.PRUNING.IMPORTANCE.TYPE).lower() == "ga":
            if bool(config.PRUNING.IMPORTANCE.STAGE3_IMPORTANCE_USE_TA_REPLACEMENT):
                required_fields = (
                    "ta_delta_on_losses",
                    "ta_delta_off_losses",
                    "ta_replacement_scores",
                    "ga_scores",
                )
            else:
                required_fields = ("ga_delta_losses", "token_delta_losses", "ga_scores")
            missing_fields = [
                field for field in required_fields if field not in payload or payload[field] in (None, {})
            ]
            if missing_fields:
                detail = (
                    "current TA-LoRA replacement-aware implementation"
                    if bool(config.PRUNING.IMPORTANCE.STAGE3_IMPORTANCE_USE_TA_REPLACEMENT)
                    else "current implementation"
                )
                raise RuntimeError(
                    "Loaded stage3 importance file is missing "
                    + ", ".join(missing_fields)
                    + f". Recompute stage3 importance with the {detail}."
                )
        return payload

    base_scores = compute_base_importance(config, model, loader, device, logger)
    payload = {
        "importance_type": config.PRUNING.IMPORTANCE.TYPE,
        "base_scores": base_scores,
    }
    if str(config.PRUNING.IMPORTANCE.TYPE).lower() == "ga":
        if bool(config.PRUNING.IMPORTANCE.STAGE3_IMPORTANCE_USE_TA_REPLACEMENT):
            delta_on_losses, delta_off_losses = compute_ta_lora_replacement_deltas(
                config, model, loader, device, logger
            )
            payload.update(
                build_ta_replacement_scores(
                    base_scores,
                    delta_on_losses,
                    delta_off_losses,
                    float(config.PRUNING.IMPORTANCE.STAGE3_IMPORTANCE_BETA),
                    float(config.PRUNING.IMPORTANCE.STAGE3_IMPORTANCE_EPS),
                    logger=logger,
                )
            )
        else:
            ga_delta_losses = compute_ga_delta_losses(config, model, loader, device, logger)
            token_delta_losses = compute_token_delta_losses(config, model, loader, device, logger)
            ga_scores = build_ga_scores(
                base_scores,
                ga_delta_losses,
                token_delta_losses,
                float(config.PRUNING.IMPORTANCE.GA_EPS),
            )
            payload["ga_delta_losses"] = ga_delta_losses
            payload["token_delta_losses"] = token_delta_losses
            payload["ga_scores"] = ga_scores
    save_importance_scores(output_dir, payload)
    return tensorize_score_payload(payload)


def generate_pruning_mask(config, score_payload, model):
    backbone = ensure_prompt_backbone(model)
    score_type = str(config.PRUNING.IMPORTANCE.TYPE).lower()
    active_scores = score_payload["base_scores"] if score_type == "base" else score_payload["ga_scores"]
    ratio = float(config.PRUNING.PRUNER.RATIO)
    task_ratios = resolve_task_prune_ratios(config)
    min_tokens = int(config.PRUNING.PRUNER.MIN_TOKENS_PER_LAYER)
    keep_indices = {}
    summary = []

    for task in config.TASKS:
        keep_indices[task] = {}
        for layer_idx in range(backbone.num_layers):
            layer_scores = active_scores[task][layer_idx].float()
            num_tokens = int(layer_scores.numel())
            task_ratio = float(task_ratios[task])
            prune_count = int(math.floor(num_tokens * task_ratio))
            prune_count = min(prune_count, max(num_tokens - min_tokens, 0))
            sorted_indices = torch.argsort(layer_scores, dim=0)
            keep_mask = torch.ones(num_tokens, dtype=torch.bool)
            if prune_count > 0:
                keep_mask[sorted_indices[:prune_count]] = False
            kept = torch.nonzero(keep_mask, as_tuple=False).flatten().cpu().long()
            keep_indices[task][layer_idx] = kept
            summary.append(
                {
                    "task": task,
                    "layer": layer_idx,
                    "score_type": score_type,
                    "prune_ratio": task_ratio,
                    "original_tokens": num_tokens,
                    "pruned_tokens": prune_count,
                    "kept_tokens": int(kept.numel()),
                }
            )

    backbone.set_prompt_pruning(keep_indices)
    prompt_stats = collect_prompt_statistics(model, config.TASKS)
    aggregate = {
        "total_original_tokens": prompt_stats["total_original_tokens"],
        "total_kept_tokens": prompt_stats["total_kept_tokens"],
        "total_pruned_tokens": prompt_stats["total_original_tokens"] - prompt_stats["total_kept_tokens"],
        "overall_keep_ratio": prompt_stats["total_keep_ratio"],
        "overall_prune_ratio": 1.0 - prompt_stats["total_keep_ratio"],
    }
    return {
        "score_type": score_type,
        "prune_ratio": ratio,
        "task_prune_ratios": task_ratios,
        "min_tokens_per_layer": min_tokens,
        "keep_indices": {
            task: {layer_idx: indices.tolist() for layer_idx, indices in layer_map.items()}
            for task, layer_map in keep_indices.items()
        },
        "summary": summary,
        "aggregate": aggregate,
        "prompt_statistics": prompt_stats,
    }


def save_pruning_mask(output_dir, pruning_mask):
    pth_path = os.path.join(output_dir, "pruning_mask.pth")
    json_path = os.path.join(output_dir, "pruning_mask.json")
    torch.save(pruning_mask, pth_path)
    save_json(json_path, pruning_mask)
    return pth_path, json_path


def load_pruning_mask(path):
    if path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    return torch.load(path, map_location="cpu")


def apply_existing_pruning(model, pruning_mask):
    keep_indices = pruning_mask["keep_indices"] if "keep_indices" in pruning_mask else pruning_mask
    ensure_prompt_backbone(model).set_prompt_pruning(keep_indices)


def maybe_build_teacher(config, device, logger):
    teacher_path = config.PRUNING.RECOVERY.TEACHER_CHECKPOINT or config.MODEL.RESUME
    if not teacher_path:
        return None
    if config.PRUNING.DISTILL.LOGIT_WEIGHT <= 0.0 and config.PRUNING.DISTILL.FEATURE_WEIGHT <= 0.0:
        return None
    teacher = build_model_for_experiment(config, device)
    load_model_state(teacher, teacher_path, config, logger)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad = False
    return teacher


def configure_recovery_trainability(config, model, logger):
    backbone = ensure_prompt_backbone(model)
    recovery_tasks = resolve_recovery_task_selection(config)

    for parameter in model.parameters():
        parameter.requires_grad = False

    if config.PRUNING.RECOVERY.TRAIN_TASK_HEADS:
        for task in recovery_tasks["head_tasks"]:
            set_task_head_trainable(model, task, True)

    if config.PRUNING.RECOVERY.TRAIN_TA_PROMPT:
        if getattr(backbone, "use_dynamic_prompts", False) and set(recovery_tasks["prompt_tasks"]) != set(config.TASKS):
            raise RuntimeError(
                "Task-wise selective prompt recovery is not supported when dynamic prompts are enabled."
            )
        for task in recovery_tasks["prompt_tasks"]:
            set_prompt_task_trainable(backbone, task, True)

    if not config.PRUNING.RECOVERY.FREEZE_BACKBONE:
        for name, parameter in backbone.named_parameters():
            if not any(token in name for token in PROMPT_PARAM_NAMES):
                parameter.requires_grad = True

    if config.PRUNING.RECOVERY.FREEZE_GALORA:
        for name, parameter in backbone.named_parameters():
            if "lora_" in name:
                parameter.requires_grad = False
    else:
        for name, parameter in backbone.named_parameters():
            if "lora_" in name:
                parameter.requires_grad = True

    last_n = int(config.PRUNING.RECOVERY.UNFREEZE_GALORA_LAST_N)
    if last_n > 0:
        start_layer = max(backbone.num_layers - last_n, 0)
        for name, parameter in backbone.named_parameters():
            if "lora_" not in name or "layers." not in name:
                continue
            try:
                layer_idx = int(name.split("layers.")[1].split(".")[0])
            except (IndexError, ValueError):
                continue
            if layer_idx >= start_layer:
                parameter.requires_grad = True

    trainable_names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    if len(trainable_names) == 0:
        raise RuntimeError("Recovery has no trainable tensors after applying task-wise trainability settings.")
    logger.info(
        "Recovery task selection: prompt=%s head=%s loss=%s distill=%s",
        recovery_tasks["prompt_tasks"],
        recovery_tasks["head_tasks"],
        recovery_tasks["loss_tasks"],
        recovery_tasks["distill_tasks"],
    )
    logger.info("Recovery trainable tensors: %d", len(trainable_names))
    return recovery_tasks


def compute_logit_distillation_loss(student_outputs, teacher_outputs, tasks):
    losses = [F.mse_loss(student_outputs[task], teacher_outputs[task]) for task in tasks]
    return torch.stack(losses).mean() if losses else torch.tensor(0.0, device=next(iter(student_outputs.values())).device)


def extract_backbone_features(model, images, tasks):
    backbone = ensure_prompt_backbone(model)
    return {task: backbone(images, task=task, return_stages=True) for task in tasks}


def compute_feature_distillation_loss(student_features, teacher_features, tasks):
    losses = []
    for task in tasks:
        for student_stage, teacher_stage in zip(student_features[task], teacher_features[task]):
            losses.append(F.mse_loss(student_stage, teacher_stage))
    return torch.stack(losses).mean() if losses else torch.tensor(0.0, device=next(iter(student_features.values()))[0].device)


def run_recovery_training(config, model, teacher, train_loader, val_loader, device, logger, output_dir):
    recovery_config = scale_learning_rates(config)
    recovery_config = recovery_config.clone()
    recovery_config.defrost()
    recovery_config.TRAIN.EPOCHS = int(config.PRUNING.RECOVERY.EPOCHS)
    recovery_config.freeze()

    recovery_tasks = configure_recovery_trainability(config, model, logger)
    optimizer = build_optimizer(recovery_config, model)
    lr_scheduler = build_scheduler(recovery_config, optimizer, len(train_loader))
    criterion = build_multitask_criterion(config)
    use_amp = bool(device.type == "cuda" and config.AMP_ENABLE)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    model.train()
    for epoch in range(recovery_config.TRAIN.EPOCHS):
        running_loss = 0.0
        for batch_idx, batch in enumerate(train_loader):
            images, targets = batch_to_device(batch, config.TASKS, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                student_outputs = model(images)
                task_loss, _ = compute_selected_task_loss(
                    criterion, student_outputs, targets, recovery_tasks["loss_tasks"], device
                )
                total_loss = task_loss
                logit_distill_loss = torch.tensor(0.0, device=device)
                feature_distill_loss = torch.tensor(0.0, device=device)

                if (
                    teacher is not None
                    and config.PRUNING.DISTILL.LOGIT_WEIGHT > 0.0
                    and len(recovery_tasks["distill_tasks"]) > 0
                ):
                    with torch.no_grad():
                        teacher_outputs = teacher(images)
                    logit_distill_loss = compute_logit_distillation_loss(
                        student_outputs, teacher_outputs, recovery_tasks["distill_tasks"]
                    )
                    total_loss = total_loss + config.PRUNING.DISTILL.LOGIT_WEIGHT * logit_distill_loss

                if (
                    teacher is not None
                    and config.PRUNING.DISTILL.FEATURE_WEIGHT > 0.0
                    and len(recovery_tasks["distill_tasks"]) > 0
                ):
                    with torch.no_grad():
                        teacher_features = extract_backbone_features(teacher, images, recovery_tasks["distill_tasks"])
                    student_features = extract_backbone_features(model, images, recovery_tasks["distill_tasks"])
                    feature_distill_loss = compute_feature_distillation_loss(
                        student_features, teacher_features, recovery_tasks["distill_tasks"]
                    )
                    total_loss = total_loss + config.PRUNING.DISTILL.FEATURE_WEIGHT * feature_distill_loss

            scaler.scale(total_loss).backward()
            if config.TRAIN.CLIP_GRAD is not None and config.TRAIN.CLIP_GRAD > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.CLIP_GRAD)
            scaler.step(optimizer)
            scaler.update()
            lr_scheduler.step_update(epoch * len(train_loader) + batch_idx)
            running_loss += float(total_loss.item())

            if batch_idx % max(int(config.PRINT_FREQ), 1) == 0:
                logger.info(
                    "Recovery Epoch [%d/%d] Step [%d/%d] total=%.6f task=%.6f logit=%.6f feature=%.6f",
                    epoch + 1,
                    recovery_config.TRAIN.EPOCHS,
                    batch_idx,
                    len(train_loader),
                    float(total_loss.item()),
                    float(task_loss.item()),
                    float(logit_distill_loss.item()),
                    float(feature_distill_loss.item()),
                )

        logger.info(
            "Recovery epoch %d finished with avg loss %.6f",
            epoch + 1,
            running_loss / max(len(train_loader), 1),
        )

    recovery_checkpoint = os.path.join(output_dir, "recovery_checkpoint.pth")
    recovery_prompt_pruning = {
        "keep_indices": ensure_prompt_backbone(model).export_prompt_pruning()
    }
    save_model_state(
        recovery_checkpoint,
        model,
        config,
        recovery_prompt_pruning,
        extra_metadata={"stage": "recovery"},
    )
    maybe_export_compact_prompt_checkpoint(
        get_compact_checkpoint_path(recovery_checkpoint),
        model,
        config,
        recovery_prompt_pruning,
        logger=logger,
        extra_metadata={"stage": "recovery"},
    )
    metrics = evaluate_model(config, val_loader, model, device, logger, output_dir, "eval_after_recovery")
    return recovery_checkpoint, metrics


def run_pruning_experiment(config, args):
    if not config.PRUNING.ENABLED:
        raise RuntimeError("PRUNING.ENABLED must be true for the pruning experiment entrypoint.")
    if not config.MTL:
        raise RuntimeError("UniPoRA prompt pruning expects the multi-task setup to be enabled.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_random_seed(config.SEED, config.DETERMINISTIC)

    output_dir = get_experiment_output_dir(config, args.cfg)
    os.makedirs(output_dir, exist_ok=True)
    logger = create_logger(output_dir=output_dir, dist_rank=0, name="pruning")
    with open(os.path.join(output_dir, "config.json"), "w", encoding="utf-8") as handle:
        handle.write(config.dump())

    logger.info("Running pruning experiment in %s", output_dir)
    logger.info("Device: %s", device)

    _, _, train_loader, val_loader, _ = build_loader(config)
    importance_loader = get_loader_subset(config, train_loader, val_loader)

    model = build_model_for_experiment(config, device)
    base_checkpoint = config.MODEL.RESUME
    if not base_checkpoint:
        raise RuntimeError("MODEL.RESUME (or --resume) must point to the trained UniPoRA checkpoint.")
    load_model_state(model, base_checkpoint, config, logger)

    prompt_stats_before = collect_prompt_statistics(model, config.TASKS)
    importance_payload = None
    pruning_mask = None

    if config.PRUNING.PRUNER.LOAD_PRUNED_CHECKPOINT:
        load_model_state(model, config.PRUNING.PRUNER.LOAD_PRUNED_CHECKPOINT, config, logger)
        pruning_mask = {
            "keep_indices": ensure_prompt_backbone(model).export_prompt_pruning(),
            "prompt_statistics": collect_prompt_statistics(model, config.TASKS),
        }

    if config.PRUNING.COMPUTE_IMPORTANCE:
        importance_payload = resolve_importance_payload(
            config, model, importance_loader, device, logger, output_dir
        )

    if config.PRUNING.PRUNER.LOAD_MASK:
        pruning_mask = load_pruning_mask(config.PRUNING.PRUNER.LOAD_MASK)
        apply_existing_pruning(model, pruning_mask)
    elif config.PRUNING.APPLY_PRUNING:
        if importance_payload is None:
            raise RuntimeError("Importance scores are required before applying pruning.")
        pruning_mask = generate_pruning_mask(config, importance_payload, model)
        save_pruning_mask(output_dir, pruning_mask)
        save_json(os.path.join(output_dir, "pruning_summary.json"), pruning_mask)
        pruned_prompt_pruning = {
            "keep_indices": ensure_prompt_backbone(model).export_prompt_pruning()
        }
        pruned_checkpoint_path = os.path.join(output_dir, "pruned_checkpoint.pth")
        save_model_state(
            pruned_checkpoint_path,
            model,
            config,
            pruned_prompt_pruning,
            extra_metadata={"stage": "pruned"},
        )
        maybe_export_compact_prompt_checkpoint(
            get_compact_checkpoint_path(pruned_checkpoint_path),
            model,
            config,
            pruned_prompt_pruning,
            logger=logger,
            extra_metadata={"stage": "pruned"},
        )

    eval_before_recovery = None
    if pruning_mask is not None and config.PRUNING.EVALUATE_AFTER_PRUNING:
        eval_before_recovery = evaluate_model(
            config, val_loader, model, device, logger, output_dir, "eval_before_recovery"
        )
        save_json(os.path.join(output_dir, "eval_before_recovery.json"), eval_before_recovery)

    eval_after_recovery = None
    if config.PRUNING.RUN_RECOVERY:
        if pruning_mask is None and not config.PRUNING.RECOVERY.STUDENT_CHECKPOINT:
            raise RuntimeError(
                "Recovery requires either an in-memory pruned model, PRUNING.PRUNER.LOAD_PRUNED_CHECKPOINT, "
                "PRUNING.PRUNER.LOAD_MASK, or PRUNING.RECOVERY.STUDENT_CHECKPOINT."
            )
        if config.PRUNING.RECOVERY.STUDENT_CHECKPOINT:
            load_model_state(model, config.PRUNING.RECOVERY.STUDENT_CHECKPOINT, config, logger)
        teacher = maybe_build_teacher(config, device, logger)
        recovery_checkpoint, eval_after_recovery = run_recovery_training(
            config, model, teacher, train_loader, val_loader, device, logger, output_dir
        )
        save_json(os.path.join(output_dir, "eval_after_recovery.json"), eval_after_recovery)
        logger.info("Saved recovery checkpoint to %s", recovery_checkpoint)

    final_prompt_stats = collect_prompt_statistics(model, config.TASKS)
    recovery_selection = (
        resolve_recovery_task_selection(config)
        if config.PRUNING.RUN_RECOVERY
        else {"prompt_tasks": [], "head_tasks": [], "loss_tasks": [], "distill_tasks": []}
    )
    experiment_summary = {
        "experiment_name": config.PRUNING.EXPERIMENT_NAME or Path(args.cfg).stem,
        "importance_type": config.PRUNING.IMPORTANCE.TYPE,
        "compute_importance": bool(config.PRUNING.COMPUTE_IMPORTANCE),
        "apply_pruning": bool(config.PRUNING.APPLY_PRUNING or config.PRUNING.PRUNER.LOAD_MASK),
        "run_recovery": bool(config.PRUNING.RUN_RECOVERY),
        "prune_ratio": float(config.PRUNING.PRUNER.RATIO),
        "task_prune_ratios": resolve_task_prune_ratios(config),
        "recovery_prompt_tasks": recovery_selection["prompt_tasks"],
        "recovery_head_tasks": recovery_selection["head_tasks"],
        "recovery_loss_tasks": recovery_selection["loss_tasks"],
        "recovery_distill_tasks": recovery_selection["distill_tasks"],
        "prompt_statistics_before": prompt_stats_before,
        "prompt_statistics_after": final_prompt_stats,
        "eval_before_recovery": eval_before_recovery,
        "eval_after_recovery": eval_after_recovery,
    }
    save_json(os.path.join(output_dir, "experiment_summary.json"), experiment_summary)
    return {"output_dir": output_dir, "summary": experiment_summary}
