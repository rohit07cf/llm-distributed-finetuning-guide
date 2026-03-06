#!/usr/bin/env bash
# GPU monitoring script using nvidia-smi.
#
# Runs a simple loop to capture GPU utilization, memory, and temperature
# at regular intervals. Useful for observing GPU behavior during training.
#
# Usage:
#   bash scripts/monitor_gpu.sh            # default 5-second interval
#   bash scripts/monitor_gpu.sh 2          # 2-second interval
#   bash scripts/monitor_gpu.sh 1 60       # 1s interval, 60 iterations

set -euo pipefail

INTERVAL="${1:-5}"
MAX_ITERATIONS="${2:-0}"  # 0 = run indefinitely

if ! command -v nvidia-smi &>/dev/null; then
    echo "[ERROR] nvidia-smi not found. Ensure NVIDIA drivers are installed."
    exit 1
fi

echo "========================================"
echo " GPU Monitor"
echo "========================================"
echo " Interval:    ${INTERVAL}s"
echo " Iterations:  $([ "${MAX_ITERATIONS}" -eq 0 ] && echo 'unlimited' || echo "${MAX_ITERATIONS}")"
echo " Started at:  $(date -Iseconds)"
echo "========================================"
echo ""

# Print header
printf "%-20s  %-6s  %-10s  %-12s  %-12s  %-8s\n" \
    "Timestamp" "GPU" "Util(%)" "Mem Used(MB)" "Mem Total(MB)" "Temp(C)"
printf "%s\n" "$(printf '%.0s-' {1..80})"

iteration=0
while true; do
    # Query nvidia-smi for all GPUs
    nvidia-smi \
        --query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu \
        --format=csv,noheader,nounits \
    | while IFS=',' read -r gpu_id util mem_used mem_total temp; do
        printf "%-20s  %-6s  %-10s  %-12s  %-12s  %-8s\n" \
            "$(date +%H:%M:%S)" \
            "$(echo "${gpu_id}" | xargs)" \
            "$(echo "${util}" | xargs)" \
            "$(echo "${mem_used}" | xargs)" \
            "$(echo "${mem_total}" | xargs)" \
            "$(echo "${temp}" | xargs)"
    done

    iteration=$((iteration + 1))
    if [ "${MAX_ITERATIONS}" -gt 0 ] && [ "${iteration}" -ge "${MAX_ITERATIONS}" ]; then
        echo ""
        echo "[INFO] Monitoring completed after ${iteration} iterations."
        break
    fi

    sleep "${INTERVAL}"
done
