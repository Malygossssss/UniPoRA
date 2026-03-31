import argparse
import json
import os

import torch

from config import get_config
from logger import create_logger
from pruning.experiment import (
    build_model_for_experiment,
    load_model_state,
    save_json,
    set_random_seed,
)


BYTES_PER_MB = 1024.0 * 1024.0


def parse_option():
    parser = argparse.ArgumentParser("UniPoRA parameter summary", add_help=False)
    parser.add_argument("--cfg", type=str, required=True, metavar="FILE", help="path to config file")
    parser.add_argument("--checkpoint", "--resume", dest="checkpoint", required=True, help="checkpoint to inspect")
    parser.add_argument("--opts", default=None, nargs="+", help="Modify config options by adding KEY VALUE pairs.")
    parser.add_argument("--output", default="output", type=str, metavar="PATH", help="root output folder")
    parser.add_argument("--output-dir", type=str, help="explicit directory to save summary artifacts")
    parser.add_argument("--data-path", type=str, help="dataset path")
    parser.add_argument("--disable_amp", action="store_true", help="disable pytorch amp")
    parser.add_argument("--seed", type=int, help="random seed")
    parser.add_argument("--deterministic", action="store_true", help="enable deterministic mode")
    parser.add_argument("--name", type=str, help="override model name")
    parser.add_argument("--tag", help="experiment tag")
    parser.add_argument("--tasks", type=str, default="depth", help="comma separated tasks")
    parser.add_argument("--nyud", type=str, help="NYUD dataset path")
    parser.add_argument("--pascal", type=str, help="PASCAL dataset path")
    parser.add_argument("--local_rank", type=int, default=0, help="local rank")
    parser.add_argument("--local-rank", type=int, default=0, help="local rank")
    parser.add_argument(
        "--list-parameters",
        action="store_true",
        help="include every named parameter in the JSON and console output",
    )
    args = parser.parse_args()
    args.resume = args.checkpoint
    config = get_config(args)
    return args, config


def get_output_dir(args):
    if args.output_dir:
        return os.path.abspath(args.output_dir)
    checkpoint_dir = os.path.dirname(os.path.abspath(args.checkpoint))
    return os.path.join(checkpoint_dir, "standalone_parameter_summary")


def get_top_level_name(parameter_name):
    if "." not in parameter_name:
        return parameter_name
    return parameter_name.split(".", 1)[0]


def tensor_num_bytes(tensor):
    return int(tensor.numel() * tensor.element_size())


def summarize_named_parameters(model, include_parameter_details=False):
    total_params = 0
    trainable_params = 0
    total_param_bytes = 0
    by_module = {}
    parameter_details = []

    for name, param in model.named_parameters():
        numel = int(param.numel())
        num_bytes = tensor_num_bytes(param)
        top_level_name = get_top_level_name(name)

        total_params += numel
        total_param_bytes += num_bytes
        if param.requires_grad:
            trainable_params += numel

        module_summary = by_module.setdefault(
            top_level_name,
            {
                "parameter_count": 0,
                "trainable_parameter_count": 0,
                "parameter_bytes": 0,
            },
        )
        module_summary["parameter_count"] += numel
        module_summary["parameter_bytes"] += num_bytes
        if param.requires_grad:
            module_summary["trainable_parameter_count"] += numel

        if include_parameter_details:
            parameter_details.append(
                {
                    "name": name,
                    "shape": list(param.shape),
                    "dtype": str(param.dtype),
                    "requires_grad": bool(param.requires_grad),
                    "parameter_count": numel,
                    "parameter_mb": float(num_bytes / BYTES_PER_MB),
                }
            )

    module_entries = []
    for module_name, module_summary in sorted(
        by_module.items(), key=lambda item: item[1]["parameter_count"], reverse=True
    ):
        module_entries.append(
            {
                "name": module_name,
                "parameter_count": int(module_summary["parameter_count"]),
                "trainable_parameter_count": int(module_summary["trainable_parameter_count"]),
                "parameter_mb": float(module_summary["parameter_bytes"] / BYTES_PER_MB),
            }
        )

    return {
        "total_parameter_count": int(total_params),
        "trainable_parameter_count": int(trainable_params),
        "parameter_mb": float(total_param_bytes / BYTES_PER_MB),
        "module_parameter_summary": module_entries,
        "parameter_details": parameter_details if include_parameter_details else None,
    }


def summarize_named_buffers(model):
    total_buffer_tensors = 0
    total_buffer_elements = 0
    total_buffer_bytes = 0

    for _, buffer in model.named_buffers():
        total_buffer_tensors += 1
        total_buffer_elements += int(buffer.numel())
        total_buffer_bytes += tensor_num_bytes(buffer)

    return {
        "total_buffer_tensor_count": int(total_buffer_tensors),
        "total_buffer_element_count": int(total_buffer_elements),
        "buffer_mb": float(total_buffer_bytes / BYTES_PER_MB),
    }


def summarize_prompt_layout(model):
    backbone = getattr(model, "backbone", None)
    if backbone is None or not hasattr(backbone, "prompt_embeddings"):
        return None

    prompt_summary = {
        "compact_prompt_layout_enabled": bool(getattr(backbone, "_compact_prompt_layout", False)),
        "entry_prompt_tokens": {},
        "deep_prompt_tokens": {},
    }

    prompt_embeddings = getattr(backbone, "prompt_embeddings", None)
    if prompt_embeddings is not None:
        for task, prompt_tensor in prompt_embeddings.items():
            prompt_summary["entry_prompt_tokens"][task] = int(prompt_tensor.shape[1])

    deep_prompt_embeddings = getattr(backbone, "deep_prompt_embeddings", None)
    if deep_prompt_embeddings is not None:
        for layer_idx, task_prompts in enumerate(deep_prompt_embeddings):
            layer_summary = {}
            for task, prompt_tensor in task_prompts.items():
                layer_summary[task] = int(prompt_tensor.shape[1])
            prompt_summary["deep_prompt_tokens"][str(layer_idx)] = layer_summary

    if hasattr(backbone, "export_prompt_pruning"):
        try:
            pruning_state = backbone.export_prompt_pruning()
        except Exception:
            pruning_state = None
        if pruning_state is not None:
            prompt_summary["runtime_prompt_pruning"] = pruning_state

    return prompt_summary


def build_summary_payload(args, config, checkpoint_bundle, model, include_parameter_details=False):
    payload = {
        "checkpoint": os.path.abspath(args.checkpoint),
        "model_name": config.MODEL.NAME,
        "tasks": list(config.TASKS),
        "compact_prompt_checkpoint": bool(
            isinstance(checkpoint_bundle, dict)
            and isinstance(checkpoint_bundle.get("compact_prompt"), dict)
            and checkpoint_bundle["compact_prompt"].get("enabled", False)
        ),
    }
    payload.update(summarize_named_parameters(model, include_parameter_details=include_parameter_details))
    payload.update(summarize_named_buffers(model))
    prompt_layout = summarize_prompt_layout(model)
    if prompt_layout is not None:
        payload["prompt_layout"] = prompt_layout
    return payload


def print_summary(payload, include_parameter_details=False):
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if not include_parameter_details:
        return
    details = payload.get("parameter_details") or []
    if not details:
        return
    print("\nParameter details:")
    for entry in details:
        print(
            f"{entry['name']}: shape={tuple(entry['shape'])}, dtype={entry['dtype']}, "
            f"params={entry['parameter_count']}, mb={entry['parameter_mb']:.4f}, "
            f"requires_grad={entry['requires_grad']}"
        )


if __name__ == "__main__":
    args, config = parse_option()
    set_random_seed(config.SEED, config.DETERMINISTIC)

    output_dir = get_output_dir(args)
    os.makedirs(output_dir, exist_ok=True)
    logger = create_logger(output_dir=output_dir, dist_rank=0, name="parameter_summary")

    device = torch.device("cpu")
    model = build_model_for_experiment(config, device)
    checkpoint_bundle = load_model_state(model, args.checkpoint, config, logger)

    payload = build_summary_payload(
        args,
        config,
        checkpoint_bundle,
        model,
        include_parameter_details=bool(args.list_parameters),
    )
    save_json(os.path.join(output_dir, "parameter_summary.json"), payload)
    print_summary(payload, include_parameter_details=bool(args.list_parameters))
