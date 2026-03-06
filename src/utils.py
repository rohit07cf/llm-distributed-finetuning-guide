"""Shared utilities for the distributed LLM fine-tuning project."""

import json
import os
import time
from pathlib import Path
from typing import Any

import yaml


def load_yaml_config(path: str) -> dict[str, Any]:
    """Load and return a YAML configuration file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_json(path: str) -> Any:
    """Load and return a JSON file."""
    with open(path, "r") as f:
        return json.load(f)


def save_json(data: Any, path: str, indent: int = 2) -> None:
    """Save data as a JSON file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=indent)


def save_jsonl(records: list[dict], path: str) -> None:
    """Append records to a JSONL file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def get_timestamp() -> str:
    """Return an ISO 8601 timestamp string."""
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def ensure_dir(path: str) -> Path:
    """Create directory if it does not exist and return the Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def count_parameters(model) -> dict[str, int]:
    """Count total and trainable parameters in a PyTorch model."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total_params": total,
        "trainable_params": trainable,
        "trainable_pct": round(100.0 * trainable / total, 4) if total > 0 else 0.0,
    }


def format_params(n: int) -> str:
    """Format a parameter count into a human-readable string (e.g., 7.0B)."""
    if n >= 1e9:
        return f"{n / 1e9:.1f}B"
    if n >= 1e6:
        return f"{n / 1e6:.1f}M"
    if n >= 1e3:
        return f"{n / 1e3:.1f}K"
    return str(n)


def estimate_model_memory_gb(num_params: int, bytes_per_param: int = 2) -> float:
    """Estimate model memory in GB.

    Args:
        num_params: Number of model parameters.
        bytes_per_param: Bytes per parameter (2 for fp16/bf16, 4 for fp32).
    """
    return round(num_params * bytes_per_param / (1024**3), 2)
