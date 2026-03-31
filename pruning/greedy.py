import json
import math
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from data import build_mtl_eval_loader
from data.mtl_ds import collate_mil, get_mtl_train_dataset, get_transformations
from logger import create_logger
from .experiment import (
    build_ga_scores,
    build_model_for_experiment,
    build_ta_replacement_scores,
    clone_keep_indices,
    collect_prompt_statistics,
    compute_base_importance,
    compute_ga_delta_losses,
    compute_ta_lora_replacement_deltas,
    compute_token_delta_losses,
    ensure_prompt_backbone,
    evaluate_model,
    get_compact_checkpoint_path,
    load_model_state,
    maybe_export_compact_prompt_checkpoint,
    save_json,
    save_model_state,
    serialize_data,
    set_random_seed,
)


SUPPORTED_GREEDY_TASKS = {"semseg", "human_parts", "sal", "normals", "edge", "depth"}


def get_greedy_experiment_output_dir(config, cfg_path):
    experiment_name = config.PRUNING.EXPERIMENT_NAME or Path(cfg_path).stem
    return os.path.join(config.OUTPUT, config.PRUNING.OUTPUT_SUBDIR, experiment_name)


def make_task_config(config, task):
    task_config = config.clone()
    task_config.defrost()
    task_config.TASKS = [task]
    task_config.freeze()
    return task_config


def build_eval_transform_train_dataset(config):
    db_name = config.DATA.DBNAME
    _, val_transforms = get_transformations(db_name, config.TASKS_CONFIG)
    return get_mtl_train_dataset(db_name, config, val_transforms)


def get_search_split_path(output_dir):
    return os.path.join(output_dir, "search_val_split.json")


def build_search_split_indices(config, dataset_size, output_dir):
    split_path = get_search_split_path(output_dir)
    ratio = float(config.PRUNING.GREEDY.SEARCH_VAL_RATIO)
    if os.path.isfile(split_path):
        with open(split_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if int(payload.get("dataset_size", -1)) != int(dataset_size):
            raise RuntimeError(
                f"Saved search-val split dataset_size={payload.get('dataset_size')} does not match current dataset_size={dataset_size}."
            )
        if int(payload.get("seed", -1)) != int(config.SEED):
            raise RuntimeError(
                f"Saved search-val split seed={payload.get('seed')} does not match current seed={config.SEED}."
            )
        if float(payload.get("search_val_ratio", -1.0)) != ratio:
            raise RuntimeError(
                f"Saved search-val split ratio={payload.get('search_val_ratio')} does not match current ratio={ratio}."
            )
        search_val_indices = [int(index) for index in payload["search_val_indices"]]
        train_remainder_indices = [int(index) for index in payload["train_remainder_indices"]]
        return payload, search_val_indices, train_remainder_indices

    if ratio <= 0.0 or ratio >= 1.0:
        raise RuntimeError("PRUNING.GREEDY.SEARCH_VAL_RATIO must be in (0, 1).")

    search_count = int(math.floor(dataset_size * ratio))
    search_count = max(search_count, 1)
    if search_count >= dataset_size:
        raise RuntimeError("Search-val split leaves no remaining train samples for importance computation.")

    rng = np.random.default_rng(int(config.SEED))
    search_val_indices = np.sort(rng.choice(dataset_size, size=search_count, replace=False)).astype(int).tolist()
    search_val_set = set(search_val_indices)
    train_remainder_indices = [index for index in range(dataset_size) if index not in search_val_set]

    payload = {
        "seed": int(config.SEED),
        "deterministic": bool(config.DETERMINISTIC),
        "dataset_size": int(dataset_size),
        "search_val_ratio": ratio,
        "search_val_count": len(search_val_indices),
        "train_remainder_count": len(train_remainder_indices),
        "search_val_indices": search_val_indices,
        "train_remainder_indices": train_remainder_indices,
    }
    save_json(split_path, payload)
    return payload, search_val_indices, train_remainder_indices


def build_subset_loader(config, dataset, indices):
    subset = Subset(dataset, indices)
    return DataLoader(
        subset,
        batch_size=config.DATA.BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        collate_fn=collate_mil,
    )


def build_search_loaders(config, output_dir):
    dataset_train_eval = build_eval_transform_train_dataset(config)
    split_payload, search_val_indices, train_remainder_indices = build_search_split_indices(
        config, len(dataset_train_eval), output_dir
    )
    search_val_loader = build_subset_loader(config, dataset_train_eval, search_val_indices)
    train_remainder_loader = build_subset_loader(config, dataset_train_eval, train_remainder_indices)
    return split_payload, search_val_loader, train_remainder_loader


def get_task_metric_spec(task, metrics):
    if task not in metrics:
        raise RuntimeError(f"Task {task} metrics are missing from evaluation output.")
    task_metrics = metrics[task]
    if task in {"semseg", "human_parts", "sal"}:
        if "mIoU" not in task_metrics:
            raise RuntimeError(f"Task {task} is missing mIoU in evaluation output.")
        return "mIoU", float(task_metrics["mIoU"]), "max"
    if task == "normals":
        if "mean" not in task_metrics:
            raise RuntimeError("Task normals is missing mean in evaluation output.")
        return "mean", float(task_metrics["mean"]), "min"
    if task == "depth":
        if "rmse" not in task_metrics:
            raise RuntimeError("Task depth is missing rmse in evaluation output.")
        return "rmse", float(task_metrics["rmse"]), "min"
    if task == "edge":
        if "odsF" in task_metrics:
            return "odsF", float(task_metrics["odsF"]), "max"
        if "loss" in task_metrics:
            return "loss", float(task_metrics["loss"]), "min"
        raise RuntimeError("Task edge is missing odsF/loss in evaluation output.")
    raise RuntimeError(f"Greedy pruning does not define an accept metric for task {task}.")


def extract_task_metric(task, metrics):
    _, value, _ = get_task_metric_spec(task, metrics)
    return value


def get_task_metric_name(task, metrics):
    name, _, _ = get_task_metric_spec(task, metrics)
    return name


def is_metric_accepted(task, candidate_metric, best_metric, metrics=None):
    if candidate_metric is None or best_metric is None:
        return False
    if math.isnan(candidate_metric) or math.isnan(best_metric):
        return False
    if task in {"semseg", "human_parts", "sal"}:
        return candidate_metric >= best_metric
    if task == "normals":
        return candidate_metric <= best_metric
    if task == "depth":
        return candidate_metric <= best_metric
    if task == "edge":
        if metrics is not None:
            metric_name = get_task_metric_name(task, metrics)
            if metric_name == "odsF":
                return candidate_metric >= best_metric
            if metric_name == "loss":
                return candidate_metric <= best_metric
        return candidate_metric >= best_metric
    raise RuntimeError(f"Greedy pruning does not define an accept rule for task {task}.")


def compute_greedy_importance_payload(config, model, data_loader, device, logger):
    base_scores = compute_base_importance(config, model, data_loader, device, logger)
    payload = {
        "importance_type": str(config.PRUNING.IMPORTANCE.TYPE).lower(),
        "base_scores": base_scores,
    }
    if str(config.PRUNING.IMPORTANCE.TYPE).lower() == "ga":
        if bool(config.PRUNING.IMPORTANCE.STAGE3_IMPORTANCE_USE_TA_REPLACEMENT):
            delta_on_losses, delta_off_losses = compute_ta_lora_replacement_deltas(
                config, model, data_loader, device, logger
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
            ga_delta_losses = compute_ga_delta_losses(config, model, data_loader, device, logger)
            token_delta_losses = compute_token_delta_losses(config, model, data_loader, device, logger)
            payload["ga_delta_losses"] = ga_delta_losses
            payload["token_delta_losses"] = token_delta_losses
            payload["ga_scores"] = build_ga_scores(
                base_scores,
                ga_delta_losses,
                token_delta_losses,
                float(config.PRUNING.IMPORTANCE.GA_EPS),
            )
    return payload


def get_active_layer_scores(score_payload, task, layer_idx):
    score_type = str(score_payload["importance_type"]).lower()
    if score_type == "base":
        return score_payload["base_scores"][task][layer_idx].float()
    return score_payload["ga_scores"][task][layer_idx].float()


def save_importance_snapshot(output_dir, task, layer_idx, accepted_iter, score_payload, active_indices):
    snapshot_dir = os.path.join(output_dir, "importance_snapshots")
    os.makedirs(snapshot_dir, exist_ok=True)
    score_type = str(score_payload["importance_type"]).lower()
    layer_scores = get_active_layer_scores(score_payload, task, layer_idx)
    active_tensor = torch.tensor(active_indices, dtype=torch.long)
    active_scores = layer_scores.index_select(0, active_tensor).cpu()
    ranking = torch.argsort(active_scores, dim=0).cpu().tolist()
    payload = {
        "task": task,
        "layer": int(layer_idx),
        "accepted_iter": int(accepted_iter),
        "score_type": score_type,
        "active_token_indices": [int(index) for index in active_indices],
        "active_scores": active_scores.tolist(),
        "active_sorted_token_indices": [int(active_indices[idx]) for idx in ranking],
    }
    snapshot_path = os.path.join(
        snapshot_dir,
        f"task_{task}_layer_{layer_idx}_accept_{accepted_iter}_{score_type}.json",
    )
    save_json(snapshot_path, payload)


def append_jsonl(path, payload):
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(serialize_data(payload), ensure_ascii=False) + "\n")


def evaluate_single_task(config, task, data_loader, model, device, logger, output_dir, split_name):
    task_config = make_task_config(config, task)
    summary = evaluate_model(task_config, data_loader, model, device, logger, output_dir, split_name)
    metric_name, metric_value, direction = get_task_metric_spec(task, summary["metrics"])
    return summary, metric_name, metric_value, direction


def save_final_mask(output_dir, pruning_mask):
    pth_path = os.path.join(output_dir, "final_pruning_mask.pth")
    json_path = os.path.join(output_dir, "final_pruning_mask.json")
    torch.save(pruning_mask, pth_path)
    save_json(json_path, pruning_mask)
    return pth_path, json_path


def summarize_layer_state(task, layer_idx, keep_indices):
    kept = keep_indices[task][layer_idx]
    return {
        "task": task,
        "layer": int(layer_idx),
        "kept_tokens": len(kept),
        "kept_indices": [int(index) for index in kept],
    }


def run_greedy_pruning_experiment(config, args):
    if not config.PRUNING.GREEDY.ENABLED:
        raise RuntimeError("PRUNING.GREEDY.ENABLED must be true for the greedy pruning entrypoint.")
    if not config.MTL:
        raise RuntimeError("Greedy prompt pruning expects the multi-task setup to be enabled.")
    unsupported_tasks = [task for task in config.TASKS if task not in SUPPORTED_GREEDY_TASKS]
    if unsupported_tasks:
        raise RuntimeError(
            f"Greedy prompt pruning currently supports {sorted(SUPPORTED_GREEDY_TASKS)} only. Got {unsupported_tasks}."
        )
    if str(config.PRUNING.GREEDY.IMPORTANCE_SOURCE).lower() != "train_remainder":
        raise RuntimeError("Greedy prompt pruning currently supports PRUNING.GREEDY.IMPORTANCE_SOURCE=train_remainder only.")
    if len(config.PRUNING.GREEDY.STEP_SCHEDULE) != 3:
        raise RuntimeError("PRUNING.GREEDY.STEP_SCHEDULE must contain exactly three percentage steps.")
    final_eval_split = str(config.PRUNING.GREEDY.FINAL_EVAL_SPLIT).lower()
    if final_eval_split not in {"val", "test"}:
        raise RuntimeError("PRUNING.GREEDY.FINAL_EVAL_SPLIT must be 'val' or 'test'.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_random_seed(config.SEED, config.DETERMINISTIC)

    output_dir = get_greedy_experiment_output_dir(config, args.cfg)
    os.makedirs(output_dir, exist_ok=True)
    logger = create_logger(output_dir=output_dir, dist_rank=0, name="greedy_pruning")
    with open(os.path.join(output_dir, "config.json"), "w", encoding="utf-8") as handle:
        handle.write(config.dump())

    logger.info("Running greedy prompt pruning in %s", output_dir)
    logger.info("Device: %s", device)
    logger.info("Greedy final evaluation split: %s", final_eval_split)
    reuse_importance_within_layer = bool(
        config.PRUNING.GREEDY.REUSE_IMPORTANCE_WITHIN_LAYER
    )
    logger.info(
        "Greedy importance reuse within layer: %s",
        reuse_importance_within_layer,
    )

    split_payload, search_val_loader, train_remainder_loader = build_search_loaders(config, output_dir)
    model = build_model_for_experiment(config, device)
    if not config.MODEL.RESUME:
        raise RuntimeError("MODEL.RESUME (or --resume) must point to the trained UniPoRA checkpoint.")
    load_model_state(model, config.MODEL.RESUME, config, logger)
    backbone = ensure_prompt_backbone(model)

    initial_metrics = evaluate_model(
        config, search_val_loader, model, device, logger, output_dir, "initial_search_val"
    )
    save_json(os.path.join(output_dir, "initial_search_val_metrics.json"), initial_metrics)
    best_metrics = {
        task: extract_task_metric(task, initial_metrics["metrics"])
        for task in config.TASKS
    }
    metric_names = {
        task: get_task_metric_name(task, initial_metrics["metrics"])
        for task in config.TASKS
    }
    initial_best_metrics = dict(best_metrics)

    trial_log_path = os.path.join(output_dir, "greedy_trials.jsonl")
    if os.path.isfile(trial_log_path):
        os.remove(trial_log_path)

    keep_indices = clone_keep_indices(backbone.export_prompt_pruning())
    min_tokens = int(config.PRUNING.GREEDY.MIN_TOKENS_PER_LAYER)
    accepted_counter = 0
    layer_summaries = []

    for task in config.TASKS:
        task_config = make_task_config(config, task)
        logger.info("Greedy pruning task %s started with best metric %.6f", task, best_metrics[task])
        for layer_idx in range(backbone.num_layers):
            logger.info("Greedy pruning task %s layer %d started", task, layer_idx)
            cached_importance_payload = None
            while True:
                backbone.set_prompt_pruning(keep_indices)
                active_indices = list(keep_indices[task][layer_idx])
                if len(active_indices) <= min_tokens:
                    logger.info(
                        "Task %s layer %d already at min tokens (%d), skipping.",
                        task,
                        layer_idx,
                        len(active_indices),
                    )
                    break

                if reuse_importance_within_layer:
                    if cached_importance_payload is None:
                        cached_importance_payload = compute_greedy_importance_payload(
                            task_config, model, train_remainder_loader, device, logger
                        )
                    importance_payload = cached_importance_payload
                else:
                    importance_payload = compute_greedy_importance_payload(
                        task_config, model, train_remainder_loader, device, logger
                    )

                layer_scores = get_active_layer_scores(importance_payload, task, layer_idx)
                active_tensor = torch.tensor(active_indices, dtype=torch.long)
                active_scores = layer_scores.index_select(0, active_tensor)
                sort_order = torch.argsort(active_scores, dim=0)
                sorted_active_indices = [int(active_indices[idx]) for idx in sort_order.cpu().tolist()]

                if bool(config.PRUNING.GREEDY.SAVE_IMPORTANCE_SNAPSHOTS):
                    save_importance_snapshot(output_dir, task, layer_idx, accepted_counter, importance_payload, active_indices)

                accepted_this_round = False
                remaining_before = len(active_indices)
                metric_before = best_metrics[task]
                percentage_steps = [
                    ("10pct", float(config.PRUNING.GREEDY.STEP_SCHEDULE[0])),
                    ("5pct", float(config.PRUNING.GREEDY.STEP_SCHEDULE[1])),
                    ("2pct", float(config.PRUNING.GREEDY.STEP_SCHEDULE[2])),
                ]

                for step_kind, ratio in percentage_steps + [("1token", None)]:
                    if step_kind == "1token":
                        prune_count = int(config.PRUNING.GREEDY.FINAL_TOKEN_STEP)
                    else:
                        prune_count = int(math.floor(remaining_before * ratio))

                    if prune_count <= 0:
                        continue
                    prune_count = min(prune_count, max(remaining_before - min_tokens, 0))
                    if prune_count <= 0:
                        continue

                    candidate_pruned_indices = sorted_active_indices[:prune_count]
                    pruned_set = set(candidate_pruned_indices)
                    candidate_keep_indices = clone_keep_indices(keep_indices)
                    candidate_keep_indices[task][layer_idx] = [
                        index for index in active_indices if index not in pruned_set
                    ]
                    if len(candidate_keep_indices[task][layer_idx]) < min_tokens:
                        continue

                    backbone.set_prompt_pruning(candidate_keep_indices)
                    candidate_eval, candidate_metric_name, candidate_metric, _ = evaluate_single_task(
                        config,
                        task,
                        search_val_loader,
                        model,
                        device,
                        logger,
                        output_dir,
                        f"search_val_{task}_layer{layer_idx}_{step_kind}",
                    )
                    accepted = is_metric_accepted(
                        task,
                        candidate_metric,
                        metric_before,
                        metrics=candidate_eval["metrics"],
                    )
                    metric_names[task] = candidate_metric_name
                    trial_record = {
                        "task": task,
                        "layer": int(layer_idx),
                        "step_kind": step_kind,
                        "remaining_before": int(remaining_before),
                        "prune_count": int(prune_count),
                        "candidate_pruned_indices": [int(index) for index in candidate_pruned_indices],
                        "metric_before": float(metric_before),
                        "metric_after": float(candidate_metric),
                        "accepted": bool(accepted),
                        "task_metric_name": candidate_metric_name,
                        "evaluation": candidate_eval,
                    }
                    append_jsonl(trial_log_path, trial_record)
                    logger.info(
                        "Task %s layer %d step %s prune=%d metric %.6f -> %.6f accepted=%s",
                        task,
                        layer_idx,
                        step_kind,
                        prune_count,
                        metric_before,
                        candidate_metric,
                        accepted,
                    )

                    if accepted:
                        keep_indices = candidate_keep_indices
                        best_metrics[task] = candidate_metric
                        backbone.set_prompt_pruning(keep_indices)
                        accepted_counter += 1
                        accepted_this_round = True
                        break

                    backbone.set_prompt_pruning(keep_indices)

                if not accepted_this_round:
                    break

            layer_summaries.append(summarize_layer_state(task, layer_idx, keep_indices))

    backbone.set_prompt_pruning(keep_indices)
    final_mask = {
        "score_type": str(config.PRUNING.IMPORTANCE.TYPE).lower(),
        "search_val_ratio": float(config.PRUNING.GREEDY.SEARCH_VAL_RATIO),
        "min_tokens_per_layer": int(config.PRUNING.GREEDY.MIN_TOKENS_PER_LAYER),
        "keep_indices": clone_keep_indices(keep_indices),
        "prompt_statistics": collect_prompt_statistics(model, config.TASKS),
    }
    final_mask["aggregate"] = {
        "total_original_tokens": final_mask["prompt_statistics"]["total_original_tokens"],
        "total_kept_tokens": final_mask["prompt_statistics"]["total_kept_tokens"],
        "total_pruned_tokens": final_mask["prompt_statistics"]["total_original_tokens"]
        - final_mask["prompt_statistics"]["total_kept_tokens"],
        "overall_keep_ratio": final_mask["prompt_statistics"]["total_keep_ratio"],
        "overall_prune_ratio": 1.0 - final_mask["prompt_statistics"]["total_keep_ratio"],
    }
    save_final_mask(output_dir, final_mask)
    save_model_state(
        os.path.join(output_dir, "final_pruned_checkpoint.pth"),
        model,
        config,
        {"keep_indices": clone_keep_indices(keep_indices)},
        extra_metadata={"stage": "greedy_pruned", "search_val_split": split_payload},
    )
    maybe_export_compact_prompt_checkpoint(
        get_compact_checkpoint_path(os.path.join(output_dir, "final_pruned_checkpoint.pth")),
        model,
        config,
        {"keep_indices": clone_keep_indices(keep_indices)},
        logger=logger,
        extra_metadata={"stage": "greedy_pruned", "search_val_split": split_payload},
    )

    final_metrics = evaluate_model(
        config, search_val_loader, model, device, logger, output_dir, "final_search_val"
    )
    save_json(os.path.join(output_dir, "final_search_val_metrics.json"), final_metrics)
    _, final_eval_loader = build_mtl_eval_loader(config, split=final_eval_split)
    final_eval_metrics = evaluate_model(
        config, final_eval_loader, model, device, logger, output_dir, f"final_{final_eval_split}"
    )
    save_json(os.path.join(output_dir, f"final_{final_eval_split}_metrics.json"), final_eval_metrics)
    if final_eval_split == "val":
        save_json(os.path.join(output_dir, "final_test_metrics.json"), final_eval_metrics)
    final_prompt_statistics = collect_prompt_statistics(model, config.TASKS)
    save_json(os.path.join(output_dir, "final_prompt_statistics.json"), final_prompt_statistics)

    summary = {
        "experiment_name": config.PRUNING.EXPERIMENT_NAME or Path(args.cfg).stem,
        "importance_type": str(config.PRUNING.IMPORTANCE.TYPE).lower(),
        "task_order": list(config.TASKS),
        "layer_order": list(range(backbone.num_layers)),
        "search_val_split": split_payload,
        "reuse_importance_within_layer": reuse_importance_within_layer,
        "initial_best_metrics": initial_best_metrics,
        "task_metric_names": metric_names,
        "final_task_metrics": {
            task: extract_task_metric(task, final_metrics["metrics"])
            for task in config.TASKS
        },
        "layer_summaries": layer_summaries,
        "final_keep_indices": clone_keep_indices(keep_indices),
        "final_prompt_statistics": final_prompt_statistics,
        "accepted_trials": int(accepted_counter),
        "initial_search_val_metrics": initial_metrics,
        "final_search_val_metrics": final_metrics,
        "final_eval_split": final_eval_split,
        "final_eval_metrics": final_eval_metrics,
        "final_test_metrics": final_eval_metrics if final_eval_split == "val" else final_eval_metrics,
    }
    save_json(os.path.join(output_dir, "greedy_summary.json"), summary)
    return {"output_dir": output_dir, "summary": summary}
