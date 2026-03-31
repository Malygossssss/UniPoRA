import argparse
import json
import os

import torch

from config import get_config
from logger import create_logger
from pruning.experiment import (
    export_compact_prompt_checkpoint,
    get_compact_checkpoint_path,
    load_model_state,
    build_model_for_experiment,
    set_random_seed,
)


def parse_option():
    parser = argparse.ArgumentParser("UniPoRA compact prompt checkpoint export", add_help=False)
    parser.add_argument("--cfg", type=str, required=True, metavar="FILE", help="path to config file")
    parser.add_argument("--checkpoint", "--resume", dest="checkpoint", required=True, help="checkpoint to compact")
    parser.add_argument("--output-path", type=str, help="explicit output checkpoint path")
    parser.add_argument("--opts", default=None, nargs="+", help="Modify config options by adding KEY VALUE pairs.")
    parser.add_argument("--data-path", type=str, help="dataset path")
    parser.add_argument("--disable_amp", action="store_true", help="disable pytorch amp")
    parser.add_argument("--seed", type=int, help="random seed")
    parser.add_argument("--deterministic", action="store_true", help="enable deterministic mode")
    parser.add_argument("--output", default="output", type=str, metavar="PATH", help="root output folder")
    parser.add_argument("--name", type=str, help="override model name")
    parser.add_argument("--tag", help="experiment tag")
    parser.add_argument("--tasks", type=str, default="depth", help="comma separated tasks")
    parser.add_argument("--nyud", type=str, help="NYUD dataset path")
    parser.add_argument("--pascal", type=str, help="PASCAL dataset path")
    parser.add_argument("--local_rank", type=int, default=0, help="local rank")
    parser.add_argument("--local-rank", type=int, default=0, help="local rank")
    args = parser.parse_args()
    args.resume = args.checkpoint
    config = get_config(args)
    return args, config


def get_output_path(args):
    if args.output_path:
        return os.path.abspath(args.output_path)
    return os.path.abspath(get_compact_checkpoint_path(args.checkpoint))


def extract_passthrough_metadata(checkpoint, source_checkpoint):
    payload = {"source_checkpoint": os.path.abspath(source_checkpoint)}
    if not isinstance(checkpoint, dict):
        return payload
    for key, value in checkpoint.items():
        if key in {"model", "prompt_pruning", "compact_prompt", "config"}:
            continue
        payload[key] = value
    return payload


if __name__ == "__main__":
    args, config = parse_option()
    device = torch.device("cpu")
    set_random_seed(config.SEED, config.DETERMINISTIC)

    output_path = get_output_path(args)
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    logger = create_logger(output_dir=output_dir, dist_rank=0, name="compact_checkpoint")
    logger.info("Exporting compact prompt checkpoint")
    logger.info("Source checkpoint: %s", args.checkpoint)
    logger.info("Output checkpoint: %s", output_path)

    model = build_model_for_experiment(config, device)
    checkpoint = load_model_state(model, args.checkpoint, config, logger)
    bundle = export_compact_prompt_checkpoint(
        output_path,
        model,
        config,
        checkpoint.get("prompt_pruning") if isinstance(checkpoint, dict) else None,
        extra_metadata=extract_passthrough_metadata(checkpoint, args.checkpoint),
    )

    payload = {
        "source_checkpoint": os.path.abspath(args.checkpoint),
        "compact_checkpoint": output_path,
        "compact_prompt": bundle["compact_prompt"],
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
