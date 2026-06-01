import argparse
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

from common import ensure_dir, load_yaml


def run_step(command: List[str], cwd: Path, dry_run: bool) -> int:
    joined = " ".join(command)
    print(f"[orchestrator] step: {joined}")
    if dry_run:
        return 0
    result = subprocess.run(command, cwd=str(cwd), check=False)
    return int(result.returncode)


def required_env_vars(config: Dict[str, object]) -> List[str]:
    env_cfg = config.get("env", {})
    if not isinstance(env_cfg, dict):
        return []
    required = env_cfg.get("required", [])
    if not isinstance(required, list):
        return []
    return [str(item) for item in required]


def validate_environment(config: Dict[str, object]) -> int:
    missing: List[str] = []
    for name in required_env_vars(config):
        if not str(__import__("os").getenv(name, "")).strip():
            missing.append(name)
    if missing:
        print("[orchestrator] missing environment variables:")
        for item in missing:
            print(f" - {item}")
        return 1
    print("[orchestrator] environment validation passed")
    return 0


def run_pipeline(config_path: str, dry_run: bool, campaign_mode: str) -> int:
    root = Path(__file__).resolve().parents[2]
    config = load_yaml(config_path)
    log_dir = root / "data" / "logs" / "survey_automation"
    ensure_dir(log_dir)

    steps: List[List[str]] = []
    steps.append([sys.executable, "src/automation/twilio_prototype_call.py", "--config", config_path, "--mode", campaign_mode])
    steps.append([sys.executable, "src/automation/compute_eligibility.py", "--config", config_path])
    steps.append([sys.executable, "src/automation/export_dpi_dclo_primary.py", "--config", config_path])
    steps.append([sys.executable, "src/automation/generate_ux_report.py", "--config", config_path, "--mode", campaign_mode])
    steps.append([sys.executable, "src/automation/prepare_whatsapp_queue.py", "--config", config_path, "--mode", campaign_mode])

    for step in steps:
        code = run_step(step, cwd=root, dry_run=dry_run)
        if code != 0:
            print(f"[orchestrator] failed step with exit code {code}: {' '.join(step)}")
            return code
    print("[orchestrator] run complete")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Survey automation orchestrator")
    parser.add_argument("--config", default="config/survey_automation.yml", help="Path to survey automation config")
    parser.add_argument(
        "--action",
        default="run",
        choices=["validate-env", "run"],
        help="Action to execute",
    )
    parser.add_argument(
        "--mode",
        default="prototype",
        choices=["prototype", "cohort1", "cohort2", "full"],
        help="Campaign mode for automation pipelines",
    )
    parser.add_argument("--dry-run", action="store_true", help="Preview orchestration without external calls")
    args = parser.parse_args()

    config = load_yaml(args.config)
    if args.action == "validate-env":
        raise SystemExit(validate_environment(config))
    raise SystemExit(run_pipeline(config_path=args.config, dry_run=args.dry_run, campaign_mode=args.mode))


if __name__ == "__main__":
    main()
