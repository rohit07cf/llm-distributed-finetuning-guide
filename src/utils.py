"""Shared utilities for the distributed LLM fine-tuning project."""

import json
import os
import time
from pathlib import Path
from typing import Any

import yaml


def load_yaml_config(path: str) -> dict[str, Any]:
    # Purpose: read a YAML file and return it as a Python dictionary.
    # Beginner view: this is the standard way this project loads training
    # configs and other structured settings files.
    """Load and return a YAML configuration file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_json(path: str) -> Any:
    # Purpose: read JSON from disk into Python objects (dict/list/etc).
    # Beginner view: helper for loading saved artifacts or config-like JSON files.
    """Load and return a JSON file."""
    with open(path, "r") as f:
        return json.load(f)


def save_json(data: Any, path: str, indent: int = 2) -> None:
    # Purpose: write Python data to pretty-printed JSON, creating directories
    # first so callers do not need to manage filesystem setup.
    # Beginner view: safe one-call JSON save utility.
    """Save data as a JSON file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=indent)


def save_jsonl(records: list[dict], path: str) -> None:
    # Purpose: append many metric/event records to a JSONL file (one JSON per line).
    # Beginner view: ideal format for streaming logs over time.
    """Append records to a JSONL file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def get_timestamp() -> str:
    # Purpose: provide a consistent timestamp format across modules.
    # Beginner view: keeps filenames/logs easy to sort by time.
    """Return an ISO 8601 timestamp string."""
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def ensure_dir(path: str) -> Path:
    # Purpose: create a directory tree if missing and return a Path object.
    # Beginner view: avoids repeating mkdir logic in calling code.
    """Create directory if it does not exist and return the Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def count_parameters(model) -> dict[str, int]:
    # Purpose: calculate total vs trainable parameters for a model object.
    # Beginner view: this quantifies how parameter-efficient LoRA/QLoRA is.
    """Count total and trainable parameters in a PyTorch model."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total_params": total,
        "trainable_params": trainable,
        "trainable_pct": round(100.0 * trainable / total, 4) if total > 0 else 0.0,
    }


def format_params(n: int) -> str:
    # Purpose: convert raw parameter counts into human-readable units.
    # Beginner view: turns large numbers like 8000000000 into 8.0B.
    """Format a parameter count into a human-readable string (e.g., 7.0B)."""
    if n >= 1e9:
        return f"{n / 1e9:.1f}B"
    if n >= 1e6:
        return f"{n / 1e6:.1f}M"
    if n >= 1e3:
        return f"{n / 1e3:.1f}K"
    return str(n)


def estimate_model_memory_gb(num_params: int, bytes_per_param: int = 2) -> float:
    # Purpose: estimate memory footprint from parameter count and precision.
    # Beginner view: quick sizing formula before running expensive experiments.
    """Estimate model memory in GB.

    Args:
        num_params: Number of model parameters.
        bytes_per_param: Bytes per parameter (2 for fp16/bf16, 4 for fp32).
    """
    return round(num_params * bytes_per_param / (1024**3), 2)
