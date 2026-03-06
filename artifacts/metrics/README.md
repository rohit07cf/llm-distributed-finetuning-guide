# Training Metrics

This directory stores GPU and training metrics captured during runs.

## Files

- `*.jsonl` — Metrics in JSON Lines format (one record per line)
- `*.csv` — Metrics in CSV format (for spreadsheet analysis)

## Generating Metrics

Metrics are automatically captured during training when using `src/train.py`
and `src/metrics_logger.py`. GPU monitoring can also be done independently:

```bash
bash scripts/monitor_gpu.sh 5    # 5-second interval
```

## Metrics Collected

| Metric | Source |
|---|---|
| GPU memory allocated | `torch.cuda.memory_allocated()` |
| GPU memory reserved | `torch.cuda.memory_reserved()` |
| GPU max memory | `torch.cuda.max_memory_allocated()` |
| GPU utilization % | `nvidia-smi` or `pynvml` |
| GPU temperature | `nvidia-smi` or `pynvml` |
| CPU RAM usage | `psutil` or `/proc/meminfo` |
| Throughput (tokens/sec) | Training loop measurement |
| Throughput (samples/sec) | Training loop measurement |
