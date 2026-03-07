#!/usr/bin/env bash
# Cloud deployment script for LLM fine-tuning inference demo.
# Installs prerequisites, builds Docker image, and launches services.
#
# Usage (on a fresh Ubuntu GPU VM):
#   export DOMAIN=llm.yourname.dev          # or "localhost" for IP-only
#   export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx # required for gated models
#   bash scripts/deploy_cloud.sh
#
# Re-run is safe — it will rebuild and restart containers.

set -euo pipefail
cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"

# ─── Configurable defaults ──────────────────────────────────────────
DOMAIN="${DOMAIN:-localhost}"
ACME_EMAIL="${ACME_EMAIL:-}"
BASE_MODEL="${BASE_MODEL:-meta-llama/Meta-Llama-3-8B-Instruct}"
ADAPTER_PATH="${ADAPTER_PATH:-/app/outputs/lora_sft}"
QUANTIZATION="${QUANTIZATION:-4}"
HF_TOKEN="${HF_TOKEN:-}"
COMPOSE_FILE="${COMPOSE_FILE:-deploy/docker-compose.prod.yml}"

echo "════════════════════════════════════════════════════════════"
echo "  LLM Inference – Cloud Deployment"
echo "════════════════════════════════════════════════════════════"
echo "  DOMAIN:       ${DOMAIN}"
echo "  BASE_MODEL:   ${BASE_MODEL}"
echo "  QUANTIZATION: ${QUANTIZATION}"
echo "  COMPOSE_FILE: ${COMPOSE_FILE}"
echo "════════════════════════════════════════════════════════════"

# ─── 1. Pre-flight checks ──────────────────────────────────────────
echo ""
echo "[1/5] Pre-flight checks..."

if ! command -v docker >/dev/null 2>&1; then
    echo "[ERROR] Docker not found. Install Docker first."
    echo "  curl -fsSL https://get.docker.com | sh"
    echo "  sudo usermod -aG docker \$USER && newgrp docker"
    exit 1
fi

if ! docker compose version >/dev/null 2>&1; then
    echo "[ERROR] Docker Compose plugin not found."
    echo "  sudo apt-get install -y docker-compose-plugin"
    exit 1
fi

# Check NVIDIA runtime (warn but don't block — compose will fail if truly missing)
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "[WARN] nvidia-smi not found. GPU inference requires NVIDIA drivers."
    echo "  Install: sudo apt-get install -y nvidia-driver-535"
fi

if ! docker info 2>/dev/null | grep -q "nvidia"; then
    echo "[WARN] NVIDIA Container Toolkit may not be configured for Docker."
    echo "  Install: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
    echo "  Quick:   sudo apt-get install -y nvidia-container-toolkit && sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker"
fi

# Check for outputs directory (adapter weights)
if [[ ! -d "outputs" ]]; then
    echo "[WARN] outputs/ directory not found."
    echo "  The API will start in degraded mode (no adapter loaded)."
    echo "  Train an adapter first: python src/train.py --config configs/lora_sft.yaml"
    mkdir -p outputs
fi

if [[ -z "$HF_TOKEN" ]]; then
    echo "[WARN] HF_TOKEN not set. Gated models (Llama, Mistral, etc.) will fail to download."
    echo "  Set it: export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx"
fi

echo "[OK] Pre-flight checks complete."

# ─── 2. Write .env.prod ────────────────────────────────────────────
echo ""
echo "[2/5] Writing deploy/.env.prod..."

cat > deploy/.env.prod <<EOF
DOMAIN=${DOMAIN}
ACME_EMAIL=${ACME_EMAIL}
HF_TOKEN=${HF_TOKEN}
BASE_MODEL=${BASE_MODEL}
ADAPTER_PATH=${ADAPTER_PATH}
QUANTIZATION=${QUANTIZATION}
EOF

echo "[OK] deploy/.env.prod written (git-ignored)."

# ─── 3. Build and start ────────────────────────────────────────────
echo ""
echo "[3/5] Building and starting containers..."

docker compose \
    --env-file deploy/.env.prod \
    -f "${COMPOSE_FILE}" \
    up --build -d

echo "[OK] Containers started."

# ─── 4. Wait for healthy ───────────────────────────────────────────
echo ""
echo "[4/5] Waiting for inference-api to become healthy (up to 3 min)..."
SECONDS_WAITED=0
MAX_WAIT=180
while [[ $SECONDS_WAITED -lt $MAX_WAIT ]]; do
    STATUS=$(docker inspect --format='{{.State.Health.Status}}' llm-inference-api 2>/dev/null || echo "missing")
    if [[ "$STATUS" == "healthy" ]]; then
        echo "[OK] inference-api is healthy after ${SECONDS_WAITED}s."
        break
    fi
    sleep 5
    SECONDS_WAITED=$((SECONDS_WAITED + 5))
    echo "  ... waiting (${SECONDS_WAITED}s, status: ${STATUS})"
done

if [[ "$STATUS" != "healthy" ]]; then
    echo "[WARN] inference-api not healthy after ${MAX_WAIT}s."
    echo "  Check logs: docker compose -f ${COMPOSE_FILE} logs inference-api"
fi

# ─── 5. Summary ────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════════"
echo "  Deployment complete!"
echo "════════════════════════════════════════════════════════════"

if [[ "$DOMAIN" == "localhost" || "$DOMAIN" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    SCHEME="http"
else
    SCHEME="https"
fi

echo "  Health:   ${SCHEME}://${DOMAIN}/healthz"
echo "  API docs: ${SCHEME}://${DOMAIN}/docs"
echo "  Generate: curl -X POST ${SCHEME}://${DOMAIN}/generate \\"
echo "              -H 'Content-Type: application/json' \\"
echo "              -d '{\"prompt\": \"Hello\", \"max_new_tokens\": 64}'"
echo ""
echo "  Logs:     docker compose -f ${COMPOSE_FILE} logs -f"
echo "  Stop:     docker compose -f ${COMPOSE_FILE} down"
echo "  Restart:  docker compose -f ${COMPOSE_FILE} restart"
echo "════════════════════════════════════════════════════════════"
