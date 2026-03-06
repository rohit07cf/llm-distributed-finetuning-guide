#!/usr/bin/env python3
"""Training entrypoint for distributed LLM fine-tuning.

Loads a YAML configuration and launches training via LLaMA-Factory,
with structured logging and metrics emission.
"""

import argparse
import json
import os
import sys
import time
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def setup_metrics_file(output_dir: str) -> str:
    """Create metrics JSONL file path inside the output directory."""
    os.makedirs(output_dir, exist_ok=True)
    return os.path.join(output_dir, "training_metrics.jsonl")


def log_metric(metrics_path: str, record: dict) -> None:
    """Append a metrics record to the JSONL log."""
    with open(metrics_path, "a") as f:
        f.write(json.dumps(record) + "\n")


class TrainingMetricsCallback:
    """Callback to capture and log training metrics per step."""

    def __init__(self, metrics_path: str, log_interval: int = 10):
        self.metrics_path = metrics_path
        self.log_interval = log_interval
        self.step_start_time = None
        self.global_step = 0
        self.total_samples = 0
        self.total_tokens = 0

    def on_step_begin(self, step: int, **kwargs):
        self.step_start_time = time.time()
        self.global_step = step

    def on_step_end(self, step: int, loss: float = 0.0, learning_rate: float = 0.0,
                     batch_size: int = 0, seq_length: int = 1024, epoch: float = 0.0,
                     gradient_accumulation_steps: int = 1, **kwargs):
        step_time = time.time() - self.step_start_time if self.step_start_time else 0.0
        samples_in_step = batch_size * gradient_accumulation_steps
        tokens_in_step = samples_in_step * seq_length
        self.total_samples += samples_in_step
        self.total_tokens += tokens_in_step

        throughput_samples = samples_in_step / step_time if step_time > 0 else 0.0
        throughput_tokens = tokens_in_step / step_time if step_time > 0 else 0.0

        record = {
            "step": step,
            "epoch": round(epoch, 4),
            "loss": round(loss, 6),
            "learning_rate": learning_rate,
            "tokens_processed": self.total_tokens,
            "samples_processed": self.total_samples,
            "step_time_sec": round(step_time, 4),
            "effective_batch_size": samples_in_step,
            "throughput_samples_per_sec": round(throughput_samples, 2),
            "throughput_tokens_per_sec": round(throughput_tokens, 2),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }

        if step % self.log_interval == 0:
            logger.info(
                f"step={step} | epoch={record['epoch']} | loss={record['loss']:.4f} | "
                f"lr={learning_rate:.2e} | tokens/s={record['throughput_tokens_per_sec']:.0f} | "
                f"step_time={record['step_time_sec']:.2f}s"
            )

        log_metric(self.metrics_path, record)


def parse_args():
    parser = argparse.ArgumentParser(description="LLM fine-tuning training entrypoint")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML training config")
    parser.add_argument("--dry-run", action="store_true", help="Validate config without training")
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load YAML configuration."""
    import yaml
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    logger.info(f"Loaded config from {config_path}")
    return config


def validate_config(config: dict) -> bool:
    """Validate required configuration fields."""
    required_fields = ["model_name_or_path", "stage", "output_dir"]
    missing = [f for f in required_fields if f not in config]
    if missing:
        logger.error(f"Missing required config fields: {missing}")
        return False
    logger.info("Config validation passed")
    return True


def run_llamafactory_training(config: dict, config_path: str) -> None:
    """Launch training using LLaMA-Factory CLI."""
    try:
        from llamafactory.train.tuner import run_exp
    except ImportError:
        logger.error(
            "LLaMA-Factory is not installed. Install with: "
            "pip install llamafactory"
        )
        sys.exit(1)

    output_dir = config.get("output_dir", "outputs/default")
    metrics_path = setup_metrics_file(output_dir)
    logger.info(f"Metrics will be logged to: {metrics_path}")
    logger.info(f"Output directory: {output_dir}")

    # Log training configuration summary
    logger.info("=" * 60)
    logger.info("Training Configuration Summary")
    logger.info("=" * 60)
    logger.info(f"  Model:            {config.get('model_name_or_path')}")
    logger.info(f"  Stage:            {config.get('stage')}")
    logger.info(f"  Fine-tuning type: {config.get('finetuning_type')}")
    logger.info(f"  LoRA rank:        {config.get('lora_rank', 'N/A')}")
    logger.info(f"  Batch size:       {config.get('per_device_train_batch_size')}")
    logger.info(f"  Grad accum:       {config.get('gradient_accumulation_steps')}")
    logger.info(f"  Learning rate:    {config.get('learning_rate')}")
    logger.info(f"  Epochs:           {config.get('num_train_epochs')}")
    logger.info(f"  DeepSpeed:        {config.get('deepspeed', 'None')}")
    if config.get("quantization_bit"):
        logger.info(f"  Quantization:     {config['quantization_bit']}-bit")
    logger.info("=" * 60)

    # Run LLaMA-Factory training
    run_exp(args=config)

    logger.info("Training completed successfully.")


def main():
    args = parse_args()
    config = load_config(args.config)

    if not validate_config(config):
        sys.exit(1)

    if args.dry_run:
        logger.info("Dry run completed — config is valid.")
        return

    run_llamafactory_training(config, args.config)


if __name__ == "__main__":
    main()
