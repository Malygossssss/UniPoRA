import argparse
import json
import os
import time

import torch

from config import get_config
from data import build_mtl_eval_loader
from logger import create_logger
from pruning.experiment import (
    build_model_for_experiment,
    load_model_state,
    save_json,
    set_random_seed,
)


BYTES_PER_MB = 1024.0 * 1024.0


def parse_option():
    parser = argparse.ArgumentParser("UniPoRA inference benchmark", add_help=False)
    parser.add_argument("--cfg", type=str, required=True, metavar="FILE", help="path to config file")
    parser.add_argument("--checkpoint", "--resume", dest="checkpoint", required=True, help="checkpoint to benchmark")
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"], help="evaluation split")
    parser.add_argument("--opts", default=None, nargs="+", help="Modify config options by adding KEY VALUE pairs.")
    parser.add_argument("--batch-size", type=int, help="throughput batch size")
    parser.add_argument("--warmup-iters", type=int, default=50, help="number of warmup iterations")
    parser.add_argument("--measure-iters", type=int, default=100, help="number of measured iterations")
    parser.add_argument("--data-path", type=str, help="dataset path")
    parser.add_argument("--disable_amp", action="store_true", help="disable pytorch amp")
    parser.add_argument("--seed", type=int, help="random seed")
    parser.add_argument("--deterministic", action="store_true", help="enable deterministic mode")
    parser.add_argument("--output", default="output", type=str, metavar="PATH", help="root output folder")
    parser.add_argument("--output-dir", type=str, help="explicit directory to save benchmark artifacts")
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


def get_output_dir(args):
    if args.output_dir:
        return args.output_dir
    checkpoint_dir = os.path.dirname(os.path.abspath(args.checkpoint))
    return os.path.join(checkpoint_dir, f"standalone_benchmark_{args.split}")


def clone_config_with_batch_size(config, batch_size):
    cloned = config.clone()
    cloned.defrost()
    cloned.DATA.BATCH_SIZE = int(batch_size)
    cloned.freeze()
    return cloned


def build_eval_loader_for_batch_size(config, split, batch_size):
    loader_config = clone_config_with_batch_size(config, batch_size)
    _, data_loader = build_mtl_eval_loader(loader_config, split=split)
    return loader_config, data_loader


def load_model_for_benchmark(config, checkpoint_path, device, logger):
    model = build_model_for_experiment(config, device)
    load_model_state(model, checkpoint_path, config, logger)
    return model


def extract_images_from_batch(batch):
    if not isinstance(batch, dict) or "image" not in batch:
        raise TypeError("Expected a multitask batch dict containing the 'image' tensor.")
    return batch["image"]


def get_first_batch_images(data_loader, device):
    iterator = iter(data_loader)
    try:
        batch = next(iterator)
    except StopIteration as exc:
        raise RuntimeError("Evaluation loader is empty; cannot benchmark inference.") from exc
    images = extract_images_from_batch(batch)
    return images.to(device, non_blocking=True)


def validate_positive_iteration_count(name, value):
    if int(value) <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value}.")


def validate_non_negative_iteration_count(name, value):
    if int(value) < 0:
        raise ValueError(f"{name} must be a non-negative integer, got {value}.")


def get_device_index(device):
    if device.type != "cuda":
        raise RuntimeError("Inference benchmarking requires a CUDA device.")
    return torch.device(device).index if torch.device(device).index is not None else torch.cuda.current_device()


def get_autocast_context(amp_enabled):
    return torch.cuda.amp.autocast(enabled=amp_enabled)


def run_forward_pass(model, images, amp_enabled):
    with torch.inference_mode():
        with get_autocast_context(amp_enabled):
            model(images)


def run_warmup(model, images, warmup_iters, amp_enabled):
    run_forward_pass(model, images, amp_enabled)
    for _ in range(int(warmup_iters)):
        run_forward_pass(model, images, amp_enabled)


def benchmark_latency(model, images, warmup_iters, measure_iters, amp_enabled, device):
    validate_non_negative_iteration_count("warmup_iters", warmup_iters)
    validate_positive_iteration_count("measure_iters", measure_iters)
    device_index = get_device_index(device)

    model.eval()
    run_warmup(model, images, warmup_iters, amp_enabled)
    torch.cuda.synchronize(device_index)
    torch.cuda.reset_peak_memory_stats(device_index)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(int(measure_iters)):
        run_forward_pass(model, images, amp_enabled)
    end_event.record()
    torch.cuda.synchronize(device_index)

    total_window_ms = float(start_event.elapsed_time(end_event))
    return {
        "batch_size": int(images.shape[0]),
        "input_shape": list(images.shape),
        "total_window_ms": total_window_ms,
        "average_latency_ms": total_window_ms / float(measure_iters),
        "peak_gpu_memory_mb": float(torch.cuda.max_memory_allocated(device_index) / BYTES_PER_MB),
    }


def benchmark_throughput(model, images, warmup_iters, measure_iters, amp_enabled, device):
    validate_non_negative_iteration_count("warmup_iters", warmup_iters)
    validate_positive_iteration_count("measure_iters", measure_iters)
    device_index = get_device_index(device)

    model.eval()
    run_warmup(model, images, warmup_iters, amp_enabled)
    torch.cuda.synchronize(device_index)
    torch.cuda.reset_peak_memory_stats(device_index)
    torch.cuda.synchronize(device_index)

    start_time = time.perf_counter()
    for _ in range(int(measure_iters)):
        run_forward_pass(model, images, amp_enabled)
    torch.cuda.synchronize(device_index)
    elapsed_seconds = time.perf_counter() - start_time

    return {
        "batch_size": int(images.shape[0]),
        "input_shape": list(images.shape),
        "total_window_s": float(elapsed_seconds),
        "throughput_images_per_sec": float(images.shape[0] * int(measure_iters) / elapsed_seconds),
        "peak_gpu_memory_mb": float(torch.cuda.max_memory_allocated(device_index) / BYTES_PER_MB),
    }


def build_benchmark_payload(checkpoint_path, split, device, amp_enabled, warmup_iters, measure_iters, latency_result, throughput_result):
    return {
        "checkpoint": os.path.abspath(checkpoint_path),
        "split": split,
        "device": str(device),
        "amp_enabled": bool(amp_enabled),
        "warmup_iters": int(warmup_iters),
        "measure_iters": int(measure_iters),
        "latency_batch_size": int(latency_result["batch_size"]),
        "throughput_batch_size": int(throughput_result["batch_size"]),
        "inference_latency_ms": float(latency_result["average_latency_ms"]),
        "throughput_images_per_sec": float(throughput_result["throughput_images_per_sec"]),
        "latency_peak_gpu_memory_mb": float(latency_result["peak_gpu_memory_mb"]),
        "throughput_peak_gpu_memory_mb": float(throughput_result["peak_gpu_memory_mb"]),
        "latency_input_shape": latency_result["input_shape"],
        "throughput_input_shape": throughput_result["input_shape"],
        "latency": latency_result,
        "throughput": throughput_result,
    }


def main():
    args, config = parse_option()
    if not torch.cuda.is_available():
        raise RuntimeError("Inference benchmarking requires CUDA; CPU fallback is not supported.")

    device = torch.device("cuda", args.local_rank)
    torch.cuda.set_device(device)
    set_random_seed(config.SEED, config.DETERMINISTIC)

    throughput_batch_size = int(args.batch_size or config.DATA.BATCH_SIZE)
    validate_non_negative_iteration_count("warmup_iters", args.warmup_iters)
    validate_positive_iteration_count("measure_iters", args.measure_iters)

    output_dir = get_output_dir(args)
    os.makedirs(output_dir, exist_ok=True)
    logger = create_logger(output_dir=output_dir, dist_rank=0, name="inference_benchmark")
    with open(os.path.join(output_dir, "config.json"), "w", encoding="utf-8") as handle:
        handle.write(config.dump())

    logger.info("Running inference benchmark on %s", args.split)
    logger.info("Checkpoint: %s", args.checkpoint)
    logger.info("Output dir: %s", output_dir)
    logger.info("Latency batch size: 1")
    logger.info("Throughput batch size: %d", throughput_batch_size)
    logger.info("Warmup iterations: %d", args.warmup_iters)
    logger.info("Measured iterations: %d", args.measure_iters)

    latency_config, latency_loader = build_eval_loader_for_batch_size(config, args.split, batch_size=1)
    throughput_config, throughput_loader = build_eval_loader_for_batch_size(
        config, args.split, batch_size=throughput_batch_size
    )

    model = load_model_for_benchmark(config, args.checkpoint, device, logger)
    latency_images = get_first_batch_images(latency_loader, device)
    throughput_images = get_first_batch_images(throughput_loader, device)

    latency_result = benchmark_latency(
        model,
        latency_images,
        warmup_iters=args.warmup_iters,
        measure_iters=args.measure_iters,
        amp_enabled=latency_config.AMP_ENABLE,
        device=device,
    )
    throughput_result = benchmark_throughput(
        model,
        throughput_images,
        warmup_iters=args.warmup_iters,
        measure_iters=args.measure_iters,
        amp_enabled=throughput_config.AMP_ENABLE,
        device=device,
    )

    payload = build_benchmark_payload(
        checkpoint_path=args.checkpoint,
        split=args.split,
        device=device,
        amp_enabled=config.AMP_ENABLE,
        warmup_iters=args.warmup_iters,
        measure_iters=args.measure_iters,
        latency_result=latency_result,
        throughput_result=throughput_result,
    )
    save_json(os.path.join(output_dir, f"benchmark_{args.split}.json"), payload)

    logger.info("Inference latency (ms/image, bs=1): %.4f", payload["inference_latency_ms"])
    logger.info("Throughput (images/s, bs=%d): %.4f", throughput_batch_size, payload["throughput_images_per_sec"])
    logger.info("Latency peak GPU memory (MB): %.2f", payload["latency_peak_gpu_memory_mb"])
    logger.info("Throughput peak GPU memory (MB): %.2f", payload["throughput_peak_gpu_memory_mb"])
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
