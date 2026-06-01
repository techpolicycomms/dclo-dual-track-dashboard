#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG_PATH="${1:-$ROOT_DIR/config/survey_automation.yml}"
MODE="${2:-prototype}"
DRY_RUN="${3:-false}"

notify() {
  local title="$1"
  local message="$2"
  osascript -e "display notification \"${message}\" with title \"${title}\"" >/dev/null 2>&1 || true
}

echo "[run_campaign] root: $ROOT_DIR"
echo "[run_campaign] config: $CONFIG_PATH"
echo "[run_campaign] mode: $MODE"
echo "[run_campaign] dry-run: $DRY_RUN"

cd "$ROOT_DIR"

notify "Survey Automation" "Validating environment"
python3 src/automation/orchestrator.py --config "$CONFIG_PATH" --action validate-env

notify "Survey Automation" "Running $MODE pipeline"
if [[ "$DRY_RUN" == "true" ]]; then
  python3 src/automation/orchestrator.py --config "$CONFIG_PATH" --mode "$MODE" --dry-run
else
  python3 src/automation/orchestrator.py --config "$CONFIG_PATH" --mode "$MODE"
fi

notify "Survey Automation" "Pipeline completed"
echo "[run_campaign] done"
