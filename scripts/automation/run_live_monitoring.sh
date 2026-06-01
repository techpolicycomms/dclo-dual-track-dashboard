#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG_PATH="${1:-$ROOT_DIR/config/survey_automation.yml}"
SYNC_INTERVAL="${2:-20}"
PORT="${3:-8787}"

echo "[live-monitor] root: $ROOT_DIR"
echo "[live-monitor] config: $CONFIG_PATH"
echo "[live-monitor] sync interval: ${SYNC_INTERVAL}s"
echo "[live-monitor] webhook port: $PORT"

cd "$ROOT_DIR"

python3 src/automation/twilio_webhook_app.py --config "$CONFIG_PATH" --port "$PORT" &
WEBHOOK_PID=$!
sleep 2

npx localtunnel --port "$PORT" &
TUNNEL_PID=$!
sleep 5

python3 src/automation/live_data_sync.py --config "$CONFIG_PATH" --interval-seconds "$SYNC_INTERVAL" &
SYNC_PID=$!

echo "[live-monitor] webhook pid: $WEBHOOK_PID"
echo "[live-monitor] tunnel pid: $TUNNEL_PID"
echo "[live-monitor] sync pid: $SYNC_PID"
echo "[live-monitor] open dashboard at Streamlit URL below"

streamlit run dashboard/survey_live_dashboard.py --server.port 8502

kill "$WEBHOOK_PID" "$TUNNEL_PID" "$SYNC_PID" >/dev/null 2>&1 || true
