#!/usr/bin/env python3
"""
dclo_weekly_loops_agent.py
--------------------------
Orchestrates one or all DCLO weekly secondary-data loops.

Usage:
  python scripts/dclo_weekly_loops_agent.py --loop srv
  python scripts/dclo_weekly_loops_agent.py --loop all

For each loop it:
  1. Runs the appropriate ingestion client
  2. Runs the shared normaliser (build_weekly_loop.py)
  3. Verifies output SHA-256 + row count
  4. Writes a JSON audit manifest to data/gold/<loop>_audit_manifest.json
"""

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
CONFIG_PATH = str(ROOT_DIR / "config" / "weekly_secondary_loops.yml")

ALL_LOOPS = ["acc", "skl", "srv", "agr", "eco", "out"]

# Map loop name → ingestion module (relative to src/ingestion)
INGEST_MODULE = {
    "acc": ("fetch_cloudflare_radar", "run"),
    "skl": ("fetch_wikimedia_pageviews", "run"),
    "srv": ("fetch_dpg_registry", "run"),
    "agr": ("fetch_gdelt_discourse", "run"),
    "eco": ("fetch_upi_datagov", "run"),
    "out": ("fetch_openaq", "run"),
}

GOLD_OUTPUT = {
    "acc": "dclo_acc_connectivity_weekly.csv",
    "skl": "dclo_skl_language_participation_weekly.csv",
    "srv": "dclo_srv_dpg_registry_weekly.csv",
    "agr": "dclo_agr_dpi_discourse_weekly.csv",
    "eco": "dclo_eco_upi_monthly.csv",
    "out": "dclo_out_air_quality_weekly.csv",
}


def get_sha256(path: Path) -> str:
    if not path.exists():
        return "file_not_found"
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(65536):
            h.update(chunk)
    return h.hexdigest()


def get_git_commit() -> str:
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(ROOT_DIR), capture_output=True, text=True, check=True
        )
        return res.stdout.strip()
    except Exception:
        return "unknown"


def run_ingest(loop_name: str) -> None:
    sys.path.insert(0, str(SRC_DIR / "ingestion"))
    module_name, func_name = INGEST_MODULE[loop_name]
    import importlib
    mod = importlib.import_module(module_name)
    fn = getattr(mod, func_name)
    fn(CONFIG_PATH)


def run_transform(loop_name: str) -> pd.DataFrame:
    sys.path.insert(0, str(SRC_DIR / "transforms"))
    import importlib
    mod = importlib.import_module("build_weekly_loop")
    return mod.run(loop_name, CONFIG_PATH)


def run_loop(loop_name: str) -> dict:
    started_at = datetime.now(timezone.utc)
    run_id = started_at.strftime(f"%Y%m%dT%H%M%SZ_{loop_name}")
    print(f"\n=== Loop: {loop_name.upper()} | run_id={run_id} ===")

    gold_dir = ROOT_DIR / "data" / "gold"
    gold_dir.mkdir(parents=True, exist_ok=True)
    gold_path = gold_dir / GOLD_OUTPUT[loop_name]

    stages = []
    status = "failed"
    rows_out = 0

    try:
        print(f"[1/2] Ingestion → {loop_name}")
        run_ingest(loop_name)
        stages.append({
            "stage": "ingestion",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "status": "ok",
        })
    except Exception as exc:
        stages.append({"stage": "ingestion", "status": "error", "error": str(exc)})
        print(f"  ERROR in ingestion: {exc}")

    try:
        print(f"[2/2] Transform → {loop_name}")
        df_out = run_transform(loop_name)
        rows_out = len(df_out)
        stages.append({
            "stage": "transform",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "rows_out": rows_out,
            "status": "ok",
        })
        status = "completed"
    except Exception as exc:
        stages.append({"stage": "transform", "status": "error", "error": str(exc)})
        print(f"  ERROR in transform: {exc}")

    completed_at = datetime.now(timezone.utc)
    sha = get_sha256(gold_path)

    manifest = {
        "pipeline": f"dclo_weekly_{loop_name}",
        "run_id": run_id,
        "started_at_utc": started_at.isoformat(),
        "completed_at_utc": completed_at.isoformat(),
        "environment": {
            "python_version": sys.version,
            "platform": platform.platform(),
            "git_commit": get_git_commit(),
        },
        "inputs": {
            "config": {
                "path": CONFIG_PATH,
                "exists": True,
                "sha256": get_sha256(Path(CONFIG_PATH)),
            }
        },
        "stages": stages,
        "outputs": {
            "gold_csv": {
                "path": str(gold_path.relative_to(ROOT_DIR)),
                "exists": gold_path.exists(),
                "sha256": sha,
                "size_bytes": gold_path.stat().st_size if gold_path.exists() else 0,
                "rows": rows_out,
            }
        },
        "status": status,
    }

    manifest_path = gold_dir / f"dclo_{loop_name}_audit_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"Manifest → {manifest_path}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DCLO weekly secondary data loops")
    parser.add_argument(
        "--loop",
        required=True,
        choices=ALL_LOOPS + ["all"],
        help="Loop name or 'all'",
    )
    args = parser.parse_args()

    loops = ALL_LOOPS if args.loop == "all" else [args.loop]
    results = {}
    for loop_name in loops:
        manifest = run_loop(loop_name)
        results[loop_name] = manifest["status"]

    print("\n=== Weekly Loops Summary ===")
    for name, st in results.items():
        print(f"  {name}: {st}")

    failed = [n for n, s in results.items() if s != "completed"]
    if failed:
        print(f"\nFAILED loops: {failed}")
        sys.exit(1)


if __name__ == "__main__":
    main()
