#!/usr/bin/env bash
# Training benchmark script.
#
# Runs benchmark comparisons across LoRA, QLoRA, and DeepSpeed ZeRO-3
# configurations and saves results to artifacts/benchmarks/.
#
# Usage:
#   bash scripts/benchmark_training.sh                 # run all benchmarks
#   bash scripts/benchmark_training.sh --num-gpus 4    # override GPU count

set -euo pipefail

OUTPUT_DIR="artifacts/benchmarks"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "========================================"
echo " Training Benchmark Suite"
echo "========================================"
echo " Output:  ${OUTPUT_DIR}"
echo " Time:    $(date -Iseconds)"
echo "========================================"

mkdir -p "${OUTPUT_DIR}"

# Run the benchmark suite
python src/benchmark.py \
    --configs \
        configs/lora_sft.yaml \
        configs/qlora_sft.yaml \
        configs/ds_z3_dist_sft.yaml \
    --output-dir "${OUTPUT_DIR}" \
    "$@"

echo ""
echo "[INFO] Benchmark results saved to ${OUTPUT_DIR}/"
echo "[INFO] Completed at $(date -Iseconds)"

# List output files
echo ""
echo "Output files:"
ls -la "${OUTPUT_DIR}"/*.json 2>/dev/null || echo "  (no JSON files generated)"
