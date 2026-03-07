# LLM Distributed Fine-Tuning

Production-grade distributed LLM fine-tuning with LoRA, QLoRA, and DeepSpeed ZeRO-3. Built for reproducible experiments, multi-GPU scaling, training benchmarking, and GPU observability.

## Project Overview

This project demonstrates end-to-end infrastructure for fine-tuning large language models with a focus on memory efficiency, distributed training, and performance engineering.

**Key capabilities:**

- **LoRA fine-tuning** — parameter-efficient fine-tuning reducing trainable parameters by ~99%
- **QLoRA** — 4-bit quantized LoRA for single-GPU training with minimal memory
- **DeepSpeed ZeRO-3** — distributed training with full parameter/optimizer/gradient sharding
- **Benchmarking** — systematic comparison of training methods, throughput, and GPU utilization
- **Observability** — real-time GPU memory, utilization, and throughput metrics
- **Deployment** — FastAPI inference server with Docker support
- **Reproducibility** — YAML-driven experiment configuration

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        Training Pipeline                         │
│                                                                  │
│  ┌─────────────┐   ┌──────────────┐   ┌──────────────────────┐  │
│  │ YAML Config │──▶│  train.py    │──▶│  LLaMA-Factory       │  │
│  │ (lora/qlora │   │  (entrypoint)│   │  (training backend)  │  │
│  │  /deepspeed)│   └──────────────┘   └──────────────────────┘  │
│  └─────────────┘          │                      │               │
│                           ▼                      ▼               │
│                  ┌─────────────────┐   ┌─────────────────────┐  │
│                  │ metrics_logger  │   │  Checkpoints/        │  │
│                  │ (GPU metrics)   │   │  Adapters            │  │
│                  └────────┬────────┘   └──────────┬──────────┘  │
│                           │                       │              │
│                           ▼                       ▼              │
│                  ┌─────────────────┐   ┌─────────────────────┐  │
│                  │ artifacts/      │   │  inference.py        │  │
│                  │  metrics/       │   │  api_server.py       │  │
│                  └─────────────────┘   └──────────┬──────────┘  │
│                                                   │              │
│                  ┌─────────────────┐              │              │
│                  │ benchmark.py    │              ▼              │
│                  │ (comparison)    │   ┌─────────────────────┐  │
│                  └────────┬────────┘   │  FastAPI /generate   │  │
│                           │            │  Docker deployment   │  │
│                           ▼            └─────────────────────┘  │
│                  ┌─────────────────┐                             │
│                  │ artifacts/      │                             │
│                  │  benchmarks/    │                             │
│                  └─────────────────┘                             │
└──────────────────────────────────────────────────────────────────┘
```

## Repository Structure

```
llm-dist-finetuning/
├── README.md
├── requirements.txt
├── .gitignore
│
├── configs/
│   ├── lora_sft.yaml           # Standard LoRA config
│   ├── qlora_sft.yaml          # 4-bit QLoRA config
│   ├── ds_z3_dist_sft.yaml     # DeepSpeed ZeRO-3 distributed config
│   └── ds_z3_config.json       # DeepSpeed engine configuration
│
├── data/
│   ├── custom_data.json        # Synthetic medical QA dataset (50 examples)
│   └── dataset_info.json       # LLaMA-Factory dataset registration
│
├── scripts/
│   ├── generate_dataset.py     # Dataset generation script
│   ├── run_training.sh         # Training launcher (LoRA/QLoRA/DeepSpeed)
│   ├── run_eval.sh             # Inference evaluation
│   ├── benchmark_training.sh   # Benchmark runner
│   └── monitor_gpu.sh          # GPU monitoring loop
│
├── src/
│   ├── train.py                # Training entrypoint
│   ├── inference.py            # Inference with LoRA adapter
│   ├── api_server.py           # FastAPI inference server
│   ├── benchmark.py            # Training benchmark comparisons
│   ├── metrics_logger.py       # GPU/system metrics collection
│   └── utils.py                # Shared utilities
│
├── deploy/
│   ├── Dockerfile              # Container for inference API
│   └── docker-compose.yml      # Docker Compose service definition
│
├── artifacts/
│   ├── benchmarks/             # Benchmark results (JSON)
│   └── metrics/                # Training metrics (JSONL)
│
├── notebooks/
│   └── training_analysis.ipynb # Visualization and analysis
│
└── .github/workflows/
    └── ci.yml                  # CI pipeline (lint, validate, test)
```

## Setup Instructions

### Prerequisites

- Python 3.10+
- CUDA 12.1+ with compatible NVIDIA drivers
- 1+ NVIDIA GPUs (24GB+ VRAM recommended)

### Installation

```bash
git clone <repo-url> && cd llm-dist-finetuning

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

### Generate Dataset

```bash
python scripts/generate_dataset.py --num-examples 50
```

This creates a synthetic 50-example medical QA dataset in Alpaca format at `data/custom_data.json`.

## Training LoRA

Standard LoRA fine-tuning with 16-bit precision on a single GPU.

```bash
bash scripts/run_training.sh lora
```

Or directly:

```bash
python src/train.py --config configs/lora_sft.yaml
```

**Configuration highlights** (`configs/lora_sft.yaml`):
- LoRA rank: 16, alpha: 32
- Targets: all attention + MLP projections
- Batch size: 4 x 4 gradient accumulation = 16 effective
- FP16 training

## Training QLoRA

4-bit quantized LoRA for memory-constrained environments.

```bash
bash scripts/run_training.sh qlora
```

**Configuration highlights** (`configs/qlora_sft.yaml`):
- 4-bit NF4 quantization via bitsandbytes
- Batch size: 2 x 8 gradient accumulation = 16 effective
- Significantly reduced VRAM footprint

## Distributed Training with DeepSpeed

Multi-GPU training with ZeRO Stage 3 parameter sharding.

```bash
# Auto-detect GPUs
bash scripts/run_training.sh deepspeed

# Specify GPU count
bash scripts/run_training.sh deepspeed 4
```

Or directly:

```bash
FORCE_TORCHRUN=1 deepspeed --num_gpus 4 \
    src/train.py --config configs/ds_z3_dist_sft.yaml
```

**DeepSpeed ZeRO-3 features** (`configs/ds_z3_config.json`):
- Stage 3: full parameter, gradient, and optimizer state sharding
- CPU offloading for optimizer and parameters
- BF16 mixed precision
- Overlapped communication

## Running Inference

After training, run inference with the fine-tuned adapter:

```bash
bash scripts/run_eval.sh outputs/lora_sft
```

Or with custom prompts:

```bash
python src/inference.py \
    --base-model meta-llama/Meta-Llama-3-8B-Instruct \
    --adapter-path outputs/lora_sft \
    --prompt "Explain the symptoms of a heart attack" \
    --quantization 4
```

## Deploying API

### Local

```bash
BASE_MODEL=meta-llama/Meta-Llama-3-8B-Instruct \
ADAPTER_PATH=outputs/lora_sft \
QUANTIZATION=4 \
uvicorn src.api_server:app --host 0.0.0.0 --port 8000
```

### Docker

```bash
cd deploy
docker-compose up --build
```

### Cloud Portfolio Deployment (HTTPS)

Use this path to expose a recruiter-facing API with TLS on your own domain.

Prerequisites:
- Cloud VM with NVIDIA GPU
- Docker + Docker Compose plugin installed
- NVIDIA Container Toolkit installed
- DNS A record pointing your domain/subdomain to the VM public IP

Production deployment assets:
- `deploy/docker-compose.prod.yml` - inference API + Caddy reverse proxy
- `deploy/Caddyfile` - HTTPS termination and reverse proxy config
- `scripts/deploy_cloud.sh` - one-command cloud deployment helper

1. Train or copy an adapter into `outputs/` (for example `outputs/lora_sft`).
2. Deploy with your domain and email:

```bash
bash scripts/deploy_cloud.sh \
    --domain llm.yourname.dev \
    --email you@example.com \
    --adapter-path /app/outputs/lora_sft \
    --base-model meta-llama/Meta-Llama-3-8B-Instruct \
    --quantization 4
```

3. Verify service:

```bash
curl https://llm.yourname.dev/healthz
curl -X POST https://llm.yourname.dev/generate \
    -H "Content-Type: application/json" \
    -d '{"prompt":"Explain heart attack symptoms","max_new_tokens":128}'
```

Notes:
- Open inbound ports `80` and `443` on your VM firewall/security group.
- The script writes deployment variables to `deploy/.env.prod`.
- If your adapter path differs, pass `--adapter-path` explicitly.

### API Usage

```bash
# Health check
curl http://localhost:8000/healthz

# Generate
curl -X POST http://localhost:8000/generate \
    -H "Content-Type: application/json" \
    -d '{"prompt": "Explain heart attack symptoms", "max_new_tokens": 256}'
```

## Benchmarking Training Performance

The benchmark suite compares training configurations systematically.

### Running Benchmarks

```bash
bash scripts/benchmark_training.sh
```

Or with options:

```bash
python src/benchmark.py \
    --configs configs/lora_sft.yaml configs/qlora_sft.yaml configs/ds_z3_dist_sft.yaml \
    --output-dir artifacts/benchmarks
```

### Metrics Captured

| Metric | Description |
|---|---|
| Total training time | Wall-clock time for the full run |
| Average step time | Mean time per training step |
| Tokens/sec | Training throughput |
| Samples/sec | Training throughput (sample-level) |
| GPU memory peak | Maximum GPU memory allocated |
| GPU utilization % | Average compute utilization |
| Training loss trend | Loss curve over training |

### Benchmark Comparison Template

> **Note:** The table below is an output format template. Run benchmarks on your hardware to populate with real numbers.

| Method | GPUs | Batch Size | Tokens/sec | GPU Mem (GB) | Notes |
|---|---|---|---|---|---|
| LoRA | 1 | 16 | — | — | Baseline single-GPU |
| QLoRA (4-bit) | 1 | 16 | — | — | ~60% memory reduction |
| DeepSpeed ZeRO-3 | 4 | 32 | — | — | Distributed sharding |

Results are saved as JSON in `artifacts/benchmarks/` for programmatic analysis.

## GPU Observability & Metrics

### Metrics Logger

The `src/metrics_logger.py` module captures GPU and system metrics during training:

- **GPU memory**: allocated, reserved, peak (via `torch.cuda`)
- **GPU utilization**: compute % and temperature (via `nvidia-smi` or `pynvml`)
- **CPU RAM**: used and total (via `psutil` or `/proc/meminfo`)
- **Throughput**: tokens/sec and samples/sec

Metrics are written to JSONL files for time-series analysis.

### GPU Monitoring

For real-time monitoring during training:

```bash
# 5-second interval
bash scripts/monitor_gpu.sh 5

# 1-second interval, 60 samples
bash scripts/monitor_gpu.sh 1 60
```

### Analysis Notebook

Open `notebooks/training_analysis.ipynb` for interactive visualization of:
- Training loss curves across methods
- Throughput comparison charts
- GPU memory usage over time
- GPU utilization profiles
- Benchmark comparison tables

## Performance Engineering & Benchmarking

### Why Throughput Matters

Measuring tokens/sec is critical because:

1. **Cost efficiency** — Higher throughput means less GPU-hours per experiment, directly reducing cloud compute costs
2. **Iteration speed** — Faster training enables more experiments, better hyperparameter searches, and quicker model iteration
3. **Scaling decisions** — Throughput per GPU reveals whether adding more GPUs will actually help (communication overhead may dominate)
4. **"It runs" is not enough** — A training job that completes but only uses 30% of available GPU compute is wasting 70% of your hardware budget

### Comparing LoRA vs QLoRA vs ZeRO-3

| Dimension | LoRA | QLoRA | DeepSpeed ZeRO-3 |
|---|---|---|---|
| **Memory** | Moderate | Low (4-bit base model) | Sharded across GPUs |
| **Speed** | Fast (single GPU) | Slower (dequantization overhead) | Depends on interconnect |
| **Scalability** | Single GPU | Single GPU | Multi-GPU |
| **Use case** | Standard fine-tuning | Memory-constrained | Large-scale training |

### GPU Utilization as a Diagnostic Tool

Low GPU utilization (<70%) often indicates:

- **Data loader bottleneck** — CPU preprocessing cannot feed the GPU fast enough. Fix: increase `preprocessing_num_workers`, use faster storage, or preprocess offline.
- **CPU offloading overhead** — DeepSpeed CPU offloading adds latency for parameter transfers. Fix: reduce offloading if GPU memory allows, or use NVMe offload.
- **Communication bottleneck** — In distributed training, gradient synchronization may dominate. Fix: use gradient accumulation to reduce sync frequency, or upgrade interconnect (NVLink, InfiniBand).
- **Small batch size** — GPU is idle between small kernel launches. Fix: increase batch size or gradient accumulation.
- **I/O bottleneck** — Checkpoint saving or logging stalls training. Fix: async checkpointing, reduce save frequency.

Monitoring GPU utilization alongside throughput reveals whether you are compute-bound (good) or bottlenecked elsewhere (actionable).

## Memory Math Explained

### Why Full Fine-Tuning Is Expensive

For a 7B parameter model in FP16:

| Component | Calculation | Memory |
|---|---|---|
| Model weights | 7B x 2 bytes (fp16) | ~14 GB |
| Gradients | 7B x 2 bytes | ~14 GB |
| Optimizer states (Adam) | 7B x 8 bytes (fp32 momentum + variance + master weights) | ~56 GB |
| Activations | Varies by batch size and sequence length | ~4-16 GB |
| **Total** | | **~88-100 GB** |

This exceeds the memory of most single GPUs (A100 80GB, H100 80GB).

### How LoRA Reduces Memory

LoRA freezes the base model and trains low-rank adapter matrices:

- **Base model**: 7B parameters, frozen, stored in fp16 (~14 GB)
- **LoRA adapters**: ~0.1-1% of base parameters, trainable
- **Gradients**: only for LoRA parameters (~14-140 MB)
- **Optimizer states**: only for LoRA parameters (~56-560 MB)

With rank 16 applied to attention and MLP layers, LoRA typically trains **~42M parameters** out of 7B — a **99.4% reduction** in trainable parameters.

### How QLoRA Further Reduces Memory

QLoRA stores the frozen base model in 4-bit NF4 format:

- **Base model**: 7B x 0.5 bytes (4-bit) = **3.5 GB** (vs 14 GB in fp16)
- **LoRA adapters**: same as LoRA (~14-140 MB trainable)
- **Total**: ~4-6 GB — fits on a single consumer GPU

The trade-off is a dequantization step during forward/backward passes that adds computational overhead.

### How DeepSpeed ZeRO-3 Enables Scaling

ZeRO Stage 3 shards all three model states across N GPUs:

| State | Without ZeRO | ZeRO-3 (4 GPUs) |
|---|---|---|
| Parameters | 14 GB each GPU | 3.5 GB each GPU |
| Gradients | 14 GB each GPU | 3.5 GB each GPU |
| Optimizer states | 56 GB each GPU | 14 GB each GPU |
| **Total per GPU** | **~84 GB** | **~21 GB** |

With CPU offloading, ZeRO-3 can further reduce per-GPU memory by moving optimizer states and parameters to host RAM, enabling training of models that exceed total GPU memory.

## Tradeoffs and Engineering Decisions

### LoRA Target Selection

We target all linear layers (`q_proj`, `v_proj`, `k_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`) rather than just query/value projections. This increases trainable parameters slightly but produces better downstream quality, especially for instruction-following tasks.

### Gradient Accumulation vs Batch Size

Larger effective batch sizes improve training stability but increase memory. We use gradient accumulation (4-8 steps) to achieve effective batch sizes of 16-32 without increasing per-step memory. The trade-off is slightly longer per-step time due to additional forward/backward passes.

### CPU Offloading

DeepSpeed CPU offloading is enabled by default for ZeRO-3 to maximize the model size that can be trained. This introduces CPU-GPU data transfer latency. For GPU clusters with ample VRAM, disabling offloading and relying purely on ZeRO sharding yields better throughput.

### FP16 vs BF16

- **FP16**: used for LoRA and QLoRA (wider hardware support, A100/V100/consumer GPUs)
- **BF16**: used for DeepSpeed (better numerical stability for distributed training, requires Ampere+)

### Config-Driven Design

All training parameters are in YAML configs rather than command-line arguments. This ensures:
- Reproducibility: every experiment is fully specified by its config
- Diffability: changes between experiments are visible in version control
- Auditability: the exact configuration that produced a model is always recorded

### Dataset Choice

The synthetic medical QA dataset (50 examples) is intentionally small for demonstration and CI testing. For production fine-tuning, replace with a larger domain-specific dataset and update `data/dataset_info.json`.

## Model Configuration

The default model is `meta-llama/Meta-Llama-3-8B-Instruct`. To use an alternative model, update the `model_name_or_path` field in any YAML config:

```yaml
# Example: switch to Qwen
model_name_or_path: Qwen/Qwen2.5-7B
template: qwen
```

## License

This project is provided for educational and portfolio purposes.
