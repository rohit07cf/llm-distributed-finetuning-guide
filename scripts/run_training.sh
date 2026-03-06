#!/usr/bin/env bash
# Training launcher script for LoRA, QLoRA, and DeepSpeed distributed training.
#
# Usage:
#   bash scripts/run_training.sh lora       # Standard LoRA training
#   bash scripts/run_training.sh qlora      # QLoRA (4-bit) training
#   bash scripts/run_training.sh deepspeed  # DeepSpeed ZeRO-3 distributed
#   bash scripts/run_training.sh deepspeed 4  # DeepSpeed on 4 GPUs

set -euo pipefail

MODE="${1:-lora}"
NUM_GPUS="${2:-$(nvidia-smi -L 2>/dev/null | wc -l || echo 1)}"

echo "========================================"
echo " LLM Distributed Fine-Tuning Launcher"
echo "========================================"
echo " Mode:     ${MODE}"
echo " GPUs:     ${NUM_GPUS}"
echo " Time:     $(date -Iseconds)"
echo "========================================"

case "${MODE}" in
    lora)
        echo "[INFO] Starting LoRA SFT training..."
        python src/train.py --config configs/lora_sft.yaml
        ;;
    qlora)
        echo "[INFO] Starting QLoRA SFT training (4-bit)..."
        python src/train.py --config configs/qlora_sft.yaml
        ;;
    deepspeed|ds)
        echo "[INFO] Starting DeepSpeed ZeRO-3 distributed training on ${NUM_GPUS} GPUs..."
        FORCE_TORCHRUN=1 deepspeed \
            --num_gpus "${NUM_GPUS}" \
            src/train.py \
            --config configs/ds_z3_dist_sft.yaml
        ;;
    *)
        echo "[ERROR] Unknown mode: ${MODE}"
        echo "Usage: $0 {lora|qlora|deepspeed} [num_gpus]"
        exit 1
        ;;
esac

echo ""
echo "[INFO] Training completed at $(date -Iseconds)"
