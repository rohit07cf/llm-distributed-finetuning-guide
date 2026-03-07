#!/usr/bin/env bash
# Smoke-test a running deployment: health check + sample inference.
# Usage: DOMAIN=llm.yourname.dev ./scripts/check_deploy.sh
#   or:  ./scripts/check_deploy.sh              (defaults to localhost)

set -euo pipefail

DOMAIN="${DOMAIN:-localhost}"

# Choose scheme: HTTPS for real domains, HTTP for localhost / raw IP.
if [[ "$DOMAIN" == "localhost" || "$DOMAIN" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    BASE_URL="http://${DOMAIN}"
else
    BASE_URL="https://${DOMAIN}"
fi

echo "════════════════════════════════════════════════════════════"
echo "  Deployment Smoke Test  →  ${BASE_URL}"
echo "════════════════════════════════════════════════════════════"

# ── 1. Health check ──────────────────────────────────────────────
echo ""
echo "1) GET ${BASE_URL}/healthz"
HTTP_CODE=$(curl -s -o /tmp/healthz_response.json -w "%{http_code}" \
    --max-time 10 "${BASE_URL}/healthz" 2>/dev/null || echo "000")

if [[ "$HTTP_CODE" == "200" ]]; then
    echo "   ✅ Health check passed (HTTP ${HTTP_CODE})"
    cat /tmp/healthz_response.json | python3 -m json.tool 2>/dev/null || cat /tmp/healthz_response.json
else
    echo "   ❌ Health check FAILED (HTTP ${HTTP_CODE})"
    cat /tmp/healthz_response.json 2>/dev/null
    echo ""
    echo "   Hint: check 'docker compose logs inference-api' for errors."
    exit 1
fi

# ── 2. API docs reachable ────────────────────────────────────────
echo ""
echo "2) GET ${BASE_URL}/docs"
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" \
    --max-time 10 "${BASE_URL}/docs" 2>/dev/null || echo "000")

if [[ "$HTTP_CODE" == "200" ]]; then
    echo "   ✅ Swagger docs reachable (HTTP ${HTTP_CODE})"
else
    echo "   ⚠️  Docs returned HTTP ${HTTP_CODE} (may still work — FastAPI serves docs at /docs)"
fi

# ── 3. Inference test ────────────────────────────────────────────
echo ""
echo "3) POST ${BASE_URL}/generate  (small prompt, max_new_tokens=64)"
RESPONSE=$(curl -s --max-time 120 \
    -X POST "${BASE_URL}/generate" \
    -H "Content-Type: application/json" \
    -d '{"prompt": "Explain what a LoRA adapter is in one sentence.", "max_new_tokens": 64, "temperature": 0.7}' \
    2>/dev/null)

if [[ -n "$RESPONSE" ]]; then
    echo "   ✅ Inference returned a response:"
    echo "$RESPONSE" | python3 -m json.tool 2>/dev/null || echo "$RESPONSE"
else
    echo "   ❌ Inference returned empty response or timed out."
    echo "   Hint: model may still be loading. Check 'docker compose logs -f inference-api'."
    exit 1
fi

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  All checks passed!  🎉"
echo "  Share these links with recruiters:"
echo "    Health:  ${BASE_URL}/healthz"
echo "    Docs:    ${BASE_URL}/docs"
echo "    Generate: curl -X POST ${BASE_URL}/generate -H 'Content-Type: application/json' -d '{"prompt": "Hello", "max_new_tokens": 64}'"
echo "════════════════════════════════════════════════════════════"
