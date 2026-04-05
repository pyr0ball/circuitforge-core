#!/bin/bash
set -e

MODE="${1:-coordinator}"
PORT="${CF_ORCH_PORT:-7700}"

case "$MODE" in
  coordinator)
    echo "[cf-orch] Starting coordinator on port $PORT"
    exec python -m circuitforge_core.resources.cli coordinator \
      --host 0.0.0.0 --port "$PORT"
    ;;
  agent)
    COORDINATOR="${CF_COORDINATOR_URL:?CF_COORDINATOR_URL must be set for agent mode}"
    GPU_IDS="${CF_AGENT_GPU_IDS:-0}"
    echo "[cf-orch] Starting agent — coordinator=$COORDINATOR gpu_ids=$GPU_IDS"
    exec python -m circuitforge_core.resources.cli agent \
      --coordinator "$COORDINATOR" \
      --gpu-ids "$GPU_IDS"
    ;;
  *)
    echo "Usage: cf-orch [coordinator|agent]"
    exit 1
    ;;
esac
