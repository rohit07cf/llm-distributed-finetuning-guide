#!/usr/bin/env bash
# Cloud deployment helper for a recruiter-facing inference demo.
#
# Usage:
#   bash scripts/deploy_cloud.sh \
#     --domain llm.yourname.dev \
#     --email you@example.com \
#     --adapter-path /app/outputs/lora_sft \
#     --base-model meta-llama/Meta-Llama-3-8B-Instruct \
#     --quantization 4

set -euo pipefail

DOMAIN=""
ACME_EMAIL=""
BASE_MODEL="meta-llama/Meta-Llama-3-8B-Instruct"
ADAPTER_PATH="/app/outputs/lora_sft"
QUANTIZATION="4"
COMPOSE_FILE="deploy/docker-compose.prod.yml"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --domain)
            DOMAIN="$2"
            shift 2
            ;;
        --email)
            ACME_EMAIL="$2"
            shift 2
            ;;
        --base-model)
            BASE_MODEL="$2"
            shift 2
            ;;
        --adapter-path)
            ADAPTER_PATH="$2"
            shift 2
            ;;
        --quantization)
            QUANTIZATION="$2"
            shift 2
            ;;
        --compose-file)
            COMPOSE_FILE="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 --domain <fqdn> --email <acme_email> [--base-model <model>] [--adapter-path <container_path>] [--quantization 0|4|8]"
            exit 0
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

if [[ -z "$DOMAIN" ]]; then
    echo "[ERROR] --domain is required (example: llm.yourname.dev)"
    exit 1
fi

if ! command -v docker >/dev/null 2>&1; then
    echo "[ERROR] docker not found. Install Docker first."
    exit 1
fi

if ! docker compose version >/dev/null 2>&1; then
    echo "[ERROR] docker compose not found. Install Docker Compose plugin."
    exit 1
fi

if [[ ! -d "outputs" ]]; then
    echo "[ERROR] outputs/ directory not found. Train and export an adapter first."
    exit 1
fi

if [[ ! -d "outputs/lora_sft" && "$ADAPTER_PATH" == "/app/outputs/lora_sft" ]]; then
    echo "[WARN] outputs/lora_sft does not exist. If you trained another adapter, pass --adapter-path accordingly."
fi

cat > deploy/.env.prod <<EOF
DOMAIN=${DOMAIN}
ACME_EMAIL=${ACME_EMAIL}
BASE_MODEL=${BASE_MODEL}
ADAPTER_PATH=${ADAPTER_PATH}
QUANTIZATION=${QUANTIZATION}
EOF

echo "[INFO] Wrote deploy/.env.prod"

docker compose \
    --env-file deploy/.env.prod \
    -f "${COMPOSE_FILE}" \
    up --build -d

echo "[INFO] Deployment started."
echo "[INFO] Health URL: https://${DOMAIN}/healthz"
echo "[INFO] API docs:   https://${DOMAIN}/docs"
