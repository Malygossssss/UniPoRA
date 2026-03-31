import argparse
import json
import os

import torch
from ptflops import get_model_complexity_info

from config import get_config
from logger import create_logger
from pruning.experiment import (
    build_model_for_experiment,
    load_model_state,
    save_json,
    set_random_seed,
)


MACS_PER_GMAC = 1e9
PARAMS_PER_MILLION = 1e6
SUPPORTED_BACKENDS = ("aten",)


def parse_option():
    parser = argparse.ArgumentParser("UniPoRA FLOPs summary", add_help=False)
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
        "--print-per-layer-stat",
        action="store_true",
        help="print per-layer ptflops statistics",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="enable ptflops verbose output",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="aten",
        choices=SUPPORTED_BACKENDS,
        help="ptflops backend to use; aten is required for transformer-style models",
    )
    args = parser.parse_args()
    args.resume = args.checkpoint
    config = get_config(args)
    return args, config


def get_output_dir(args):
    if args.output_dir:
        return os.path.abspath(args.output_dir)
    checkpoint_dir = os.path.dirname(os.path.abspath(args.checkpoint))
    return os.path.join(checkpoint_dir, "standalone_flops")


def normalize_image_size(config):
    image_size = config.DATA.IMG_SIZE
    if isinstance(image_size, (list, tuple)):
        if len(image_size) != 2:
            raise ValueError(f"Expected DATA.IMG_SIZE to have length 2, got {image_size}.")
        return int(image_size[0]), int(image_size[1])
    return int(image_size), int(image_size)


def get_decoder_head_summary(config):
    decoder_map = getattr(config.MODEL, "DECODER_HEAD", {})
    return {
        task: decoder_map.get(task, "hrnet") if hasattr(decoder_map, "get") else "hrnet"
        for task in config.TASKS
    }


def build_counting_warnings(config):
    warnings = [
        "This script reports theoretical MACs/GFLOPs from ptflops aten backend, not latency.",
        "ptflops aten backend mainly counts aten mm/matmul/addmm/bmm/convolution ops.",
        "Ops such as softmax, layernorm, interpolate, cat, index_select, roll, zeros, mean and reshape-like memory ops may be omitted or only partially covered.",
    ]
    if config.MTL:
        warnings.append(
            "The counting scope is the full multitask inference forward, including backbone, downsampler and decoder heads; it is not limited to trainable parameters."
        )
    if getattr(config.MODEL.PROMPT, "ENABLED", False):
        warnings.append(
            "Prompt-enabled MultiTaskSwin runs the backbone once per task in the actual forward path; reported MACs are for that full repeated-backbone inference path."
        )
    if getattr(config.MODEL.MTLORA, "ENABLED", False):
        warnings.append(
            "MTLoRA modules are profiled in their current unmerged runtime form. This matches actual implementation, but non-matmul tensor ops inside custom wrappers may still be undercounted."
        )
    return warnings


def compute_flops_summary(model, input_shape, backend, print_per_layer_stat=False, verbose=False):
    model.eval()
    macs, params = get_model_complexity_info(
        model,
        input_shape,
        as_strings=False,
        backend=backend,
        print_per_layer_stat=bool(print_per_layer_stat),
        verbose=bool(verbose),
    )
    macs = float(macs)
    params = float(params)
    return {
        "input_shape": list(input_shape),
        "macs": macs,
        "gmacs": macs / MACS_PER_GMAC,
        "gflops": (2.0 * macs) / MACS_PER_GMAC,
        "parameter_count": int(params),
        "parameter_millions": params / PARAMS_PER_MILLION,
    }


def build_summary_payload(args, config, checkpoint_bundle, flops_summary):
    return {
        "checkpoint": os.path.abspath(args.checkpoint),
        "profile_backend": args.backend,
        "model_name": config.MODEL.NAME,
        "model_type": config.MODEL.TYPE,
        "tasks": list(config.TASKS),
        "task_count": int(len(config.TASKS)),
        "img_size": list(flops_summary["input_shape"][1:]),
        "decoder_heads": get_decoder_head_summary(config),
        "mtl_enabled": bool(config.MTL),
        "prompt_enabled": bool(getattr(config.MODEL.PROMPT, "ENABLED", False)),
        "mtlora_enabled": bool(getattr(config.MODEL.MTLORA, "ENABLED", False)),
        "compact_prompt_checkpoint": bool(
            isinstance(checkpoint_bundle, dict)
            and isinstance(checkpoint_bundle.get("compact_prompt"), dict)
            and checkpoint_bundle["compact_prompt"].get("enabled", False)
        ),
        "input_shape": flops_summary["input_shape"],
        "macs": flops_summary["macs"],
        "gmacs": flops_summary["gmacs"],
        "gflops": flops_summary["gflops"],
        "parameter_count": flops_summary["parameter_count"],
        "parameter_millions": flops_summary["parameter_millions"],
        "warnings": build_counting_warnings(config),
        "notes": {
            "gflops_definition": "Computed as 2 * MACs / 1e9.",
            "device": "cpu",
            "counting_scope": "full model inference forward",
        },
    }


if __name__ == "__main__":
    args, config = parse_option()
    set_random_seed(config.SEED, config.DETERMINISTIC)

    output_dir = get_output_dir(args)
    os.makedirs(output_dir, exist_ok=True)
    logger = create_logger(output_dir=output_dir, dist_rank=0, name="flops_summary")

    device = torch.device("cpu")
    model = build_model_for_experiment(config, device)
    checkpoint_bundle = load_model_state(model, args.checkpoint, config, logger)

    input_h, input_w = normalize_image_size(config)
    flops_summary = compute_flops_summary(
        model,
        (3, input_h, input_w),
        backend=args.backend,
        print_per_layer_stat=bool(args.print_per_layer_stat),
        verbose=bool(args.verbose),
    )
    payload = build_summary_payload(args, config, checkpoint_bundle, flops_summary)

    save_json(os.path.join(output_dir, "flops_summary.json"), payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
