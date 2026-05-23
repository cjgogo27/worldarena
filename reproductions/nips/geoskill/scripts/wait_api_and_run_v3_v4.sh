#!/bin/bash
# Wait for nowcoding.ai API to recover, then launch v3 and v4 experiments back-to-back.

set -euo pipefail

API_URL="https://nowcoding.ai/v1/chat/completions"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT="$(cd "${SCRIPT_DIR}/.." && pwd)"
API_KEY="${VLM_API_KEY:-}"
CONDA_ENV="openclaw"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

if [[ -z "$API_KEY" ]]; then
    log "ERROR: VLM_API_KEY is not set. Export VLM_API_KEY before running this script."
    exit 1
fi

log "Starting API monitor. Will launch v3+v4 experiments when API recovers."

while true; do
    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$API_URL" \
        -H "Authorization: Bearer $API_KEY" \
        -H "Content-Type: application/json" \
        -d '{"model":"claude-sonnet-4-5","messages":[{"role":"user","content":"hi"}],"max_tokens":3}' \
        --max-time 15 2>/dev/null || echo "000")

    if [[ "$HTTP_CODE" == "200" ]]; then
        log "API is UP (HTTP $HTTP_CODE)! Launching experiments..."
        break
    fi

    log "API still down (HTTP $HTTP_CODE). Retrying in 60s..."
    sleep 60
done

log "=== Running v3 smoke test (3 games) ==="
conda run -n "$CONDA_ENV" python "$PROJECT/scripts/run_experiment.py" \
    --methods skill_conditioned_v3 \
    --max-games 3 \
    2>&1 | tee "$PROJECT/experiments/full_100/smoke_v3.log"

log "=== Running v4 smoke test (3 games) ==="
conda run -n "$CONDA_ENV" python "$PROJECT/scripts/run_experiment.py" \
    --methods skill_conditioned_v4 \
    --max-games 3 \
    2>&1 | tee "$PROJECT/experiments/full_100/smoke_v4.log"

log "=== Smoke tests done. Launching full 100-game v3 run ==="
conda run -n "$CONDA_ENV" python "$PROJECT/scripts/run_experiment.py" \
    --methods skill_conditioned_v3 \
    2>&1 | tee "$PROJECT/experiments/full_100/skill_conditioned_v3_run.log"

log "=== v3 done. Launching full 100-game v4 run ==="
conda run -n "$CONDA_ENV" python "$PROJECT/scripts/run_experiment.py" \
    --methods skill_conditioned_v4 \
    2>&1 | tee "$PROJECT/experiments/full_100/skill_conditioned_v4_run.log"

log "=== Recomputing metrics ==="
conda run -n "$CONDA_ENV" python "$PROJECT/scripts/recompute_metrics_v2.py" \
    2>&1 | tee "$PROJECT/experiments/full_100/metrics_recompute.log"

log "=== ALL DONE ==="
cat "$PROJECT/experiments/full_100/metrics_recompute.log"
