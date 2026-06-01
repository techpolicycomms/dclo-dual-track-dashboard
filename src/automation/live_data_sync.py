import argparse
import subprocess
import sys
import time
from pathlib import Path


def run_once(root: Path, config_path: str) -> int:
    steps = [
        [sys.executable, "src/automation/compute_eligibility.py", "--config", config_path],
        [sys.executable, "src/automation/export_dpi_dclo_primary.py", "--config", config_path],
        [sys.executable, "src/automation/generate_ux_report.py", "--config", config_path, "--mode", "prototype"],
    ]
    for step in steps:
        result = subprocess.run(step, cwd=str(root), check=False)
        if result.returncode != 0:
            return int(result.returncode)
    return 0


def run_loop(config_path: str, interval_seconds: int) -> None:
    root = Path(__file__).resolve().parents[2]
    print(f"[live-sync] started with interval={interval_seconds}s")
    while True:
        code = run_once(root=root, config_path=config_path)
        if code != 0:
            print(f"[live-sync] sync iteration failed with exit code {code}")
        time.sleep(interval_seconds)


def main() -> None:
    parser = argparse.ArgumentParser(description="Continuously sync incoming survey data into dashboard-ready outputs")
    parser.add_argument("--config", default="config/survey_automation.yml")
    parser.add_argument("--interval-seconds", type=int, default=20)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    if args.once:
        raise SystemExit(run_once(root=root, config_path=args.config))
    run_loop(config_path=args.config, interval_seconds=max(5, args.interval_seconds))


if __name__ == "__main__":
    main()
