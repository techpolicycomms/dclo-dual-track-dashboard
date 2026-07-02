#!/usr/bin/env python3
"""
Detect new or revised ICT / economic indicators from international org APIs.

Compares freshly fetched raw files against a persisted sync state. Emits a
rebuild signal for GitHub Actions when World Bank or RestCountries data moves.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import yaml

ROOT_DIR = Path(__file__).resolve().parents[1]

# ICT + macro prefixes used in DCLO country API track (World Bank WDI family).
CORE_INDICATOR_PREFIXES = (
    "IT.",
    "NY.",
    "SL.",
    "NE.",
    "CC.",
    "PV.",
    "RL.",
    "SE.",
    "EG.",
    "SP.",
    "SI.",
    "EN.",
    "SH.",
    "FB.",
)


def read_config(config_path: Path) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_state(state_path: Path) -> Dict[str, Any]:
    if not state_path.exists():
        return {"version": 1, "last_checked_utc": None, "sources": {}}
    return json.loads(state_path.read_text(encoding="utf-8"))


def save_state(state_path: Path, state: Dict[str, Any]) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def is_core_indicator(code: str) -> bool:
    normalized = str(code).upper()
    return any(normalized.startswith(prefix) for prefix in CORE_INDICATOR_PREFIXES)


def fingerprint_world_bank(wb_path: Path) -> Dict[str, Any]:
    if not wb_path.exists():
        return {"exists": False}

    df = pd.read_csv(wb_path)
    required = {"indicator_code", "iso3c", "year", "value"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"World Bank file missing columns: {sorted(missing)}")

    df = df.dropna(subset=["iso3c", "year"])
    df["indicator_code"] = df["indicator_code"].astype(str)
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    core = df[df["indicator_code"].map(is_core_indicator)].copy()
    observed = core.dropna(subset=["value"])

    max_year_by_indicator: Dict[str, int] = {}
    for code, group in observed.groupby("indicator_code"):
        max_year_by_indicator[str(code)] = int(group["year"].max())

    fingerprint_rows = (
        observed[["indicator_code", "iso3c", "year", "value"]]
        .sort_values(["indicator_code", "iso3c", "year"])
        .astype({"indicator_code": str, "iso3c": str, "year": int, "value": float})
    )
    fingerprint_text = fingerprint_rows.to_csv(index=False)
    return {
        "exists": True,
        "row_count": int(len(df)),
        "core_row_count": int(len(core)),
        "observed_core_row_count": int(len(observed)),
        "indicator_count": int(df["indicator_code"].nunique()),
        "core_indicator_count": int(core["indicator_code"].nunique()),
        "max_year_overall": int(observed["year"].max()) if len(observed) else None,
        "max_year_by_indicator": max_year_by_indicator,
        "fingerprint_sha256": sha256_text(fingerprint_text),
    }


def fingerprint_restcountries(rc_path: Path) -> Dict[str, Any]:
    if not rc_path.exists():
        return {"exists": False}

    df = pd.read_csv(rc_path)
    required = {"iso3c", "country_name", "population"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"RestCountries file missing columns: {sorted(missing)}")

    summary = (
        df[["iso3c", "country_name", "population"]]
        .dropna(subset=["iso3c"])
        .sort_values("iso3c")
        .astype({"iso3c": str, "country_name": str, "population": float})
    )
    fingerprint_text = summary.to_csv(index=False)
    return {
        "exists": True,
        "country_count": int(len(summary)),
        "fingerprint_sha256": sha256_text(fingerprint_text),
    }


def fingerprint_dpi_long(dpi_path: Path) -> Dict[str, Any]:
    if not dpi_path.exists():
        return {"exists": False}

    df = pd.read_csv(dpi_path, usecols=lambda c: c in {"indicator_code", "economy", "year", "norm_score"})
    if df.empty:
        return {"exists": True, "row_count": 0, "fingerprint_sha256": sha256_text("")}

    df = df.dropna(subset=["indicator_code", "economy", "year"])
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    text = (
        df.sort_values(["indicator_code", "economy", "year"])
        .astype({"indicator_code": str, "economy": str, "year": int})
        .to_csv(index=False)
    )
    return {
        "exists": True,
        "row_count": int(len(df)),
        "max_year_overall": int(df["year"].max()) if df["year"].notna().any() else None,
        "fingerprint_sha256": sha256_text(text),
    }


def build_current_snapshot(
    raw_dir: Path,
    config: Dict[str, Any],
    dpi_config_path: Optional[Path],
) -> Dict[str, Any]:
    output_cfg = config.get("output", {})
    wb_cfg = config.get("world_bank", {})
    rc_cfg = config.get("restcountries", {})

    wb_path = raw_dir / str(wb_cfg.get("output_file", "wb_indicator_long_api.csv"))
    rc_path = raw_dir / str(rc_cfg.get("output_file", "country_master_restcountries.csv"))

    snapshot: Dict[str, Any] = {
        "world_bank": fingerprint_world_bank(wb_path),
        "restcountries": fingerprint_restcountries(rc_path),
    }

    if dpi_config_path and dpi_config_path.exists():
        dpi_cfg = read_config(dpi_config_path)
        inputs = dpi_cfg.get("inputs", {})
        if isinstance(inputs, dict):
            dpi_long = Path(str(inputs.get("dpi_long_path", "")))
            snapshot["dpi_long"] = fingerprint_dpi_long(dpi_long)

    return snapshot


def _max_year_increased(previous: Dict[str, Any], current: Dict[str, Any]) -> bool:
    prev_map = previous.get("max_year_by_indicator", {})
    curr_map = current.get("max_year_by_indicator", {})
    if not isinstance(prev_map, dict) or not isinstance(curr_map, dict):
        return False
    for code, year in curr_map.items():
        prev_year = prev_map.get(code)
        if prev_year is None:
            continue
        if int(year) > int(prev_year):
            return True
    return False


def detect_changes(previous_sources: Dict[str, Any], current_snapshot: Dict[str, Any]) -> Tuple[bool, List[str]]:
    reasons: List[str] = []

    if not previous_sources:
        if any(src.get("exists") for src in current_snapshot.values()):
            reasons.append("initial_sync_or_empty_state")
        return bool(reasons), reasons

    for source_name, current in current_snapshot.items():
        previous = previous_sources.get(source_name, {})
        if not current.get("exists"):
            continue
        if not previous.get("exists"):
            reasons.append(f"{source_name}:new_source_file")
            continue

        prev_hash = previous.get("fingerprint_sha256")
        curr_hash = current.get("fingerprint_sha256")
        if prev_hash != curr_hash:
            reasons.append(f"{source_name}:fingerprint_changed")

        if source_name == "world_bank" and _max_year_increased(previous, current):
            reasons.append("world_bank:new_indicator_year")

        for metric in ("row_count", "core_row_count", "observed_core_row_count", "country_count"):
            if metric in current and metric in previous and current[metric] != previous[metric]:
                reasons.append(f"{source_name}:{metric}_changed")

        prev_max = previous.get("max_year_overall")
        curr_max = current.get("max_year_overall")
        if prev_max is not None and curr_max is not None and int(curr_max) > int(prev_max):
            reasons.append(f"{source_name}:max_year_increased")

    return bool(reasons), reasons


def write_github_output(output_path: Path, should_rebuild: bool, reasons: List[str]) -> None:
    lines = [
        f"should_rebuild={'true' if should_rebuild else 'false'}",
        f"change_reasons={json.dumps(reasons)}",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect international indicator updates for DCLO refresh.")
    parser.add_argument("--config", default="config/country_api_sources.yml")
    parser.add_argument("--dpi-config", default="config/dpi_country_sources.yml")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--state-file", default="data/gold/dclo_indicator_sync_state.json")
    parser.add_argument("--update-state", action="store_true", help="Persist current snapshot after successful rebuild")
    parser.add_argument("--github-output", default=None, help="Path to GITHUB_OUTPUT for Actions")
    args = parser.parse_args()

    config_path = ROOT_DIR / args.config
    dpi_config_path = ROOT_DIR / args.dpi_config
    raw_dir = ROOT_DIR / args.raw_dir
    state_path = ROOT_DIR / args.state_file

    config = read_config(config_path)
    state = load_state(state_path)
    current_snapshot = build_current_snapshot(raw_dir=raw_dir, config=config, dpi_config_path=dpi_config_path)
    should_rebuild, reasons = detect_changes(state.get("sources", {}), current_snapshot)

    print(f"[change-detector] should_rebuild={should_rebuild}")
    if reasons:
        print("[change-detector] reasons:")
        for reason in reasons:
            print(f"  - {reason}")

    if args.github_output:
        write_github_output(Path(args.github_output), should_rebuild, reasons)

    if args.update_state:
        state["version"] = 1
        state["last_checked_utc"] = datetime.now(timezone.utc).isoformat()
        state["sources"] = current_snapshot
        if should_rebuild:
            state["last_rebuild_utc"] = state["last_checked_utc"]
            state["last_rebuild_reasons"] = reasons
        save_state(state_path, state)
        print(f"[change-detector] updated state: {state_path}")

    raise SystemExit(0)


if __name__ == "__main__":
    main()
