#!/usr/bin/env bash
# Evaluation / inference test script.
#
# Runs inference on sample prompts using the fine-tuned adapter
# and prints the generated outputs.
#
# Usage:
#   bash scripts/run_eval.sh                          # default adapter
#   bash scripts/run_eval.sh outputs/qlora_sft        # custom adapter path

set -euo pipefail

ADAPTER_PATH="${1:-outputs/lora_sft}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
QUANTIZATION="${QUANTIZATION:-0}"

echo "========================================"
echo " Inference Evaluation"
echo "========================================"
echo " Base model:    ${BASE_MODEL}"
echo " Adapter:       ${ADAPTER_PATH}"
echo " Quantization:  ${QUANTIZATION}-bit"
echo "========================================"

if [ ! -d "${ADAPTER_PATH}" ]; then
    echo "[ERROR] Adapter directory not found: ${ADAPTER_PATH}"
    echo "Train a model first with: bash scripts/run_training.sh lora"
    exit 1
fi

python src/inference.py \
    --base-model "${BASE_MODEL}" \
    --adapter-path "${ADAPTER_PATH}" \
    --quantization "${QUANTIZATION}" \
    --max-new-tokens 512 \
    --temperature 0.7

echo ""
echo "[INFO] Evaluation completed at $(date -Iseconds)"
