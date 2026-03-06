#!/usr/bin/env python3
"""GPU and system metrics logger for training observability.

Captures GPU memory, utilization, CPU RAM, and throughput metrics
during training runs. Outputs to JSONL for downstream analysis.
"""

import json
import os
import platform
import subprocess
import time
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class GPUMetrics:
    """Snapshot of GPU and system metrics at a point in time."""
    timestamp: str
    step: Optional[int] = None
    gpu_id: int = 0
    gpu_memory_allocated_mb: float = 0.0
    gpu_memory_reserved_mb: float = 0.0
    gpu_max_memory_allocated_mb: float = 0.0
    gpu_utilization_pct: Optional[float] = None
    gpu_temperature_c: Optional[float] = None
    cpu_ram_used_mb: float = 0.0
    cpu_ram_total_mb: float = 0.0
    throughput_tokens_per_sec: float = 0.0
    throughput_samples_per_sec: float = 0.0


def get_torch_gpu_metrics(device_id: int = 0) -> dict:
    """Collect GPU memory metrics using PyTorch CUDA APIs.

    Returns a dict with memory_allocated_mb, memory_reserved_mb,
    and max_memory_allocated_mb.
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return {}

        device = torch.device(f"cuda:{device_id}")
        return {
            "gpu_memory_allocated_mb": round(
                torch.cuda.memory_allocated(device) / (1024**2), 2
            ),
            "gpu_memory_reserved_mb": round(
                torch.cuda.memory_reserved(device) / (1024**2), 2
            ),
            "gpu_max_memory_allocated_mb": round(
                torch.cuda.max_memory_allocated(device) / (1024**2), 2
            ),
        }
    except Exception as e:
        logger.debug(f"Could not collect torch GPU metrics: {e}")
        return {}


def get_nvidia_smi_metrics(device_id: int = 0) -> dict:
    """Parse nvidia-smi for utilization and temperature.

    Returns a dict with gpu_utilization_pct and gpu_temperature_c.
    """
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                f"--id={device_id}",
                "--query-gpu=utilization.gpu,temperature.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return {}

        parts = result.stdout.strip().split(",")
        if len(parts) >= 2:
            return {
                "gpu_utilization_pct": float(parts[0].strip()),
                "gpu_temperature_c": float(parts[1].strip()),
            }
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError) as e:
        logger.debug(f"nvidia-smi not available: {e}")
    return {}


def get_pynvml_metrics(device_id: int = 0) -> dict:
    """Collect GPU metrics using pynvml (if available).

    Fallback when nvidia-smi is not in PATH or subprocess is restricted.
    """
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)

        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)

        pynvml.nvmlShutdown()
        return {
            "gpu_utilization_pct": float(util.gpu),
            "gpu_temperature_c": float(temp),
        }
    except Exception as e:
        logger.debug(f"pynvml not available: {e}")
        return {}


def get_cpu_ram_metrics() -> dict:
    """Get CPU RAM usage using /proc/meminfo or psutil."""
    try:
        import psutil
        mem = psutil.virtual_memory()
        return {
            "cpu_ram_used_mb": round(mem.used / (1024**2), 2),
            "cpu_ram_total_mb": round(mem.total / (1024**2), 2),
        }
    except ImportError:
        pass

    # Fallback: parse /proc/meminfo on Linux
    if platform.system() == "Linux":
        try:
            with open("/proc/meminfo", "r") as f:
                lines = f.readlines()
            info = {}
            for line in lines:
                parts = line.split()
                if len(parts) >= 2:
                    info[parts[0].rstrip(":")] = int(parts[1])
            total_kb = info.get("MemTotal", 0)
            available_kb = info.get("MemAvailable", 0)
            return {
                "cpu_ram_used_mb": round((total_kb - available_kb) / 1024, 2),
                "cpu_ram_total_mb": round(total_kb / 1024, 2),
            }
        except Exception:
            pass

    return {"cpu_ram_used_mb": 0.0, "cpu_ram_total_mb": 0.0}


def collect_metrics(
    device_id: int = 0,
    step: Optional[int] = None,
    throughput_tokens: float = 0.0,
    throughput_samples: float = 0.0,
) -> GPUMetrics:
    """Collect a full snapshot of GPU and system metrics.

    Tries torch CUDA APIs first, then nvidia-smi, then pynvml for
    utilization data.
    """
    metrics = GPUMetrics(
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
        step=step,
        gpu_id=device_id,
        throughput_tokens_per_sec=throughput_tokens,
        throughput_samples_per_sec=throughput_samples,
    )

    # Torch CUDA memory
    torch_metrics = get_torch_gpu_metrics(device_id)
    metrics.gpu_memory_allocated_mb = torch_metrics.get("gpu_memory_allocated_mb", 0.0)
    metrics.gpu_memory_reserved_mb = torch_metrics.get("gpu_memory_reserved_mb", 0.0)
    metrics.gpu_max_memory_allocated_mb = torch_metrics.get("gpu_max_memory_allocated_mb", 0.0)

    # GPU utilization: try nvidia-smi first, then pynvml
    util_metrics = get_nvidia_smi_metrics(device_id)
    if not util_metrics:
        util_metrics = get_pynvml_metrics(device_id)
    metrics.gpu_utilization_pct = util_metrics.get("gpu_utilization_pct")
    metrics.gpu_temperature_c = util_metrics.get("gpu_temperature_c")

    # CPU RAM
    ram = get_cpu_ram_metrics()
    metrics.cpu_ram_used_mb = ram.get("cpu_ram_used_mb", 0.0)
    metrics.cpu_ram_total_mb = ram.get("cpu_ram_total_mb", 0.0)

    return metrics


class MetricsLogger:
    """Persistent metrics logger that writes snapshots to a JSONL file."""

    def __init__(self, output_path: str, device_id: int = 0, log_interval_steps: int = 10):
        self.output_path = output_path
        self.device_id = device_id
        self.log_interval_steps = log_interval_steps

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        logger.info(f"MetricsLogger initialized -> {output_path}")

    def log(
        self,
        step: Optional[int] = None,
        throughput_tokens: float = 0.0,
        throughput_samples: float = 0.0,
    ) -> GPUMetrics:
        """Collect and write metrics for the current step."""
        snapshot = collect_metrics(
            device_id=self.device_id,
            step=step,
            throughput_tokens=throughput_tokens,
            throughput_samples=throughput_samples,
        )
        with open(self.output_path, "a") as f:
            f.write(json.dumps(asdict(snapshot)) + "\n")
        return snapshot

    def should_log(self, step: int) -> bool:
        """Check whether metrics should be logged at this step."""
        return step % self.log_interval_steps == 0
