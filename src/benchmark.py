#!/usr/bin/env python3
"""Training benchmark module for comparing fine-tuning configurations.

Supports benchmarking LoRA vs QLoRA vs DeepSpeed ZeRO-3, single-GPU vs
multi-GPU, and different batch sizes / gradient accumulation settings.

Outputs machine-readable JSON results to artifacts/benchmarks/.
"""

import argparse
import json
import os
import subprocess
import sys
import time
import logging
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Container for a single benchmark run result."""

    config_name: str
    config_path: str
    method: str  # lora, qlora, deepspeed_z3
    num_gpus: int = 1
    batch_size: int = 0
    gradient_accumulation_steps: int = 1
    effective_batch_size: int = 0
    total_training_time_sec: float = 0.0
    avg_step_time_sec: float = 0.0
    tokens_per_sec: float = 0.0
    samples_per_sec: float = 0.0
    gpu_memory_peak_mb: float = 0.0
    gpu_utilization_avg_pct: Optional[float] = None
    training_loss_final: Optional[float] = None
    training_loss_history: list = field(default_factory=list)
    num_steps: int = 0
    num_epochs: int = 0
    timestamp: str = ""
    notes: str = ""


def detect_num_gpus() -> int:
    # Purpose: detect how many CUDA GPUs are available so benchmark runs can
    # choose a sensible default without manual input.
    # Beginner view: this auto-detects your hardware before scheduling tests.
    """Detect the number of available CUDA GPUs."""
    try:
        import torch

        return torch.cuda.device_count()
    except ImportError:
        return 0


def parse_training_metrics(metrics_path: str) -> dict:
    # Purpose: aggregate per-step training logs into benchmark-friendly summary
    # numbers like average step time, average throughput, and final loss.
    # Beginner view: converts raw training logs into easy-to-compare metrics.
    """Parse training metrics JSONL file and compute aggregates.

    Returns a dict with aggregate stats from the training run.
    """
    if not os.path.exists(metrics_path):
        logger.warning(f"Metrics file not found: {metrics_path}")
        return {}

    records = []
    with open(metrics_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if not records:
        return {}

    step_times = [
        r.get("step_time_sec", 0) for r in records if r.get("step_time_sec", 0) > 0
    ]
    tokens_per_sec = [
        r.get("throughput_tokens_per_sec", 0)
        for r in records
        if r.get("throughput_tokens_per_sec", 0) > 0
    ]
    samples_per_sec = [
        r.get("throughput_samples_per_sec", 0)
        for r in records
        if r.get("throughput_samples_per_sec", 0) > 0
    ]
    losses = [r.get("loss", 0) for r in records if r.get("loss") is not None]

    return {
        "num_steps": len(records),
        "avg_step_time_sec": round(sum(step_times) / len(step_times), 4)
        if step_times
        else 0.0,
        "avg_tokens_per_sec": round(sum(tokens_per_sec) / len(tokens_per_sec), 2)
        if tokens_per_sec
        else 0.0,
        "avg_samples_per_sec": round(sum(samples_per_sec) / len(samples_per_sec), 2)
        if samples_per_sec
        else 0.0,
        "training_loss_final": round(losses[-1], 6) if losses else None,
        "training_loss_history": [round(loss_value, 6) for loss_value in losses[-10:]],
    }


def parse_gpu_metrics(metrics_path: str) -> dict:
    # Purpose: summarize GPU behavior (peak memory and average utilization)
    # from runtime telemetry logs.
    # Beginner view: helps compare efficiency and hardware pressure per method.
    """Parse GPU metrics JSONL file for peak memory and avg utilization."""
    if not os.path.exists(metrics_path):
        return {}

    records = []
    with open(metrics_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if not records:
        return {}

    peak_mem = max(r.get("gpu_max_memory_allocated_mb", 0) for r in records)
    utils = [
        r.get("gpu_utilization_pct", 0)
        for r in records
        if r.get("gpu_utilization_pct") is not None
    ]

    return {
        "gpu_memory_peak_mb": round(peak_mem, 2),
        "gpu_utilization_avg_pct": round(sum(utils) / len(utils), 2) if utils else None,
    }


def run_benchmark(
    config_path: str,
    config_name: str,
    method: str,
    num_gpus: int = 1,
    output_dir: str = "artifacts/benchmarks",
) -> BenchmarkResult:
    # Purpose: execute one training config end-to-end, measure wall-clock time,
    # parse generated logs, and package the result in a standard schema.
    # Beginner view: this is one complete "experiment run".
    """Run a single training benchmark.

    Launches the training script, measures wall-clock time, then parses
    the resulting metrics files for detailed statistics.
    """
    import yaml

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    batch_size = config.get("per_device_train_batch_size", 1)
    grad_accum = config.get("gradient_accumulation_steps", 1)
    effective_bs = batch_size * grad_accum * num_gpus

    result = BenchmarkResult(
        config_name=config_name,
        config_path=config_path,
        method=method,
        num_gpus=num_gpus,
        batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        effective_batch_size=effective_bs,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )

    # Build the training command
    if num_gpus > 1 and config.get("deepspeed"):
        cmd = [
            "deepspeed",
            f"--num_gpus={num_gpus}",
            "src/train.py",
            "--config",
            config_path,
        ]
    else:
        cmd = [
            sys.executable,
            "src/train.py",
            "--config",
            config_path,
        ]

    logger.info(f"Starting benchmark: {config_name} ({method}) on {num_gpus} GPU(s)")
    logger.info(f"Command: {' '.join(cmd)}")

    start_time = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
        if proc.returncode != 0:
            logger.error(f"Training failed:\n{proc.stderr[-1000:]}")
            result.notes = f"Training failed with return code {proc.returncode}"
            return result
    except subprocess.TimeoutExpired:
        result.notes = "Training timed out after 2 hours"
        return result

    result.total_training_time_sec = round(time.time() - start_time, 2)

    # Parse metrics from training output
    train_output_dir = config.get("output_dir", "outputs/default")
    training_metrics_path = os.path.join(train_output_dir, "training_metrics.jsonl")
    gpu_metrics_path = os.path.join(train_output_dir, "gpu_metrics.jsonl")

    train_stats = parse_training_metrics(training_metrics_path)
    result.num_steps = train_stats.get("num_steps", 0)
    result.avg_step_time_sec = train_stats.get("avg_step_time_sec", 0.0)
    result.tokens_per_sec = train_stats.get("avg_tokens_per_sec", 0.0)
    result.samples_per_sec = train_stats.get("avg_samples_per_sec", 0.0)
    result.training_loss_final = train_stats.get("training_loss_final")
    result.training_loss_history = train_stats.get("training_loss_history", [])

    gpu_stats = parse_gpu_metrics(gpu_metrics_path)
    result.gpu_memory_peak_mb = gpu_stats.get("gpu_memory_peak_mb", 0.0)
    result.gpu_utilization_avg_pct = gpu_stats.get("gpu_utilization_avg_pct")

    # Save individual benchmark result
    os.makedirs(output_dir, exist_ok=True)
    result_path = os.path.join(output_dir, f"benchmark_{config_name}.json")
    with open(result_path, "w") as f:
        json.dump(asdict(result), f, indent=2)
    logger.info(f"Benchmark result saved to {result_path}")

    return result


def run_comparison_suite(
    configs: list[dict],
    output_dir: str = "artifacts/benchmarks",
) -> list[BenchmarkResult]:
    # Purpose: orchestrate a list of benchmark runs and save a single report
    # containing all results for side-by-side analysis.
    # Beginner view: this is the batch runner for your experiment suite.
    """Run a suite of benchmark configurations and save a comparison report."""
    results = []
    for cfg in configs:
        result = run_benchmark(
            config_path=cfg["config_path"],
            config_name=cfg["config_name"],
            method=cfg["method"],
            num_gpus=cfg.get("num_gpus", 1),
            output_dir=output_dir,
        )
        results.append(result)

    # Save comparison report
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "comparison_report.json")
    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "num_benchmarks": len(results),
        "results": [asdict(r) for r in results],
    }
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Comparison report saved to {report_path}")

    # Print summary table
    print("\n" + "=" * 100)
    print(
        f"{'Config':<25} {'Method':<12} {'GPUs':<5} {'Eff.BS':<7} {'Time(s)':<10} "
        f"{'Tok/s':<10} {'GPU Mem(MB)':<12} {'Loss':<10}"
    )
    print("=" * 100)
    for r in results:
        loss_str = f"{r.training_loss_final:.4f}" if r.training_loss_final else "N/A"
        print(
            f"{r.config_name:<25} {r.method:<12} {r.num_gpus:<5} {r.effective_batch_size:<7} "
            f"{r.total_training_time_sec:<10.1f} {r.tokens_per_sec:<10.1f} "
            f"{r.gpu_memory_peak_mb:<12.1f} {loss_str:<10}"
        )
    print("=" * 100 + "\n")

    return results


def main():
    # Purpose: parse CLI arguments, infer method types from config names,
    # determine GPU counts, and launch the full comparison suite.
    # Beginner view: this is the command-line starting point of benchmark.py.
    parser = argparse.ArgumentParser(description="Benchmark training configurations")
    parser.add_argument(
        "--configs",
        nargs="+",
        default=[
            "configs/lora_sft.yaml",
            "configs/qlora_sft.yaml",
            "configs/ds_z3_dist_sft.yaml",
        ],
        help="Config files to benchmark",
    )
    parser.add_argument(
        "--output-dir", default="artifacts/benchmarks", help="Output directory"
    )
    parser.add_argument("--num-gpus", type=int, default=None, help="Override GPU count")
    args = parser.parse_args()

    num_gpus = args.num_gpus or detect_num_gpus() or 1

    # Build benchmark suite
    config_suite = []
    for config_path in args.configs:
        name = Path(config_path).stem
        if "qlora" in name:
            method = "qlora"
        elif "ds_z3" in name:
            method = "deepspeed_z3"
        else:
            method = "lora"

        gpus = num_gpus if method == "deepspeed_z3" else 1
        config_suite.append(
            {
                "config_path": config_path,
                "config_name": name,
                "method": method,
                "num_gpus": gpus,
            }
        )

    run_comparison_suite(config_suite, args.output_dir)


if __name__ == "__main__":
    main()
