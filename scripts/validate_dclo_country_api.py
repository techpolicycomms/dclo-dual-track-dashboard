#!/usr/bin/env python3
"""Lightweight QA for the World Bank / RestCountries API country-year DCLO track."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]

REQUIRED_COLUMNS = [
    "economy",
    "iso3c",
    "year",
    "DCLO_score",
    "n_domains_used",
    "coverage_ratio",
    "model_trust_tier",
]

DOMAIN_SCORE_COLUMNS = [
    "ACC_score",
    "SKL_score",
    "SRV_score",
    "AGR_score",
    "ECO_score",
    "OUT_score",
]


def validate_api_country_output(path: Path, year_min: int, year_max: int) -> Dict[str, Any]:
    issues: List[str] = []

    if not path.exists():
        return {"passed": False, "issues": [f"missing_output:{path}"]}

    df = pd.read_csv(path)
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        issues.append(f"missing_columns:{','.join(missing_cols)}")

    if df.duplicated(subset=["iso3c", "year"]).any():
        issues.append(f"duplicate_keys:{int(df.duplicated(subset=['iso3c', 'year']).sum())}")

    null_score = int(df["DCLO_score"].isna().sum()) if "DCLO_score" in df.columns else len(df)
    if null_score > 0:
        issues.append(f"null_dclo_score:{null_score}")

    years = pd.to_numeric(df["year"], errors="coerce")
    if years.notna().any():
        if int(years.min()) < year_min:
            issues.append(f"year_below_min:{int(years.min())}")
        if int(years.max()) > year_max:
            issues.append(f"year_above_max:{int(years.max())}")

    for col in DOMAIN_SCORE_COLUMNS:
        if col not in df.columns:
            issues.append(f"missing_domain_score:{col}")

    trust_values = set(df["model_trust_tier"].dropna().astype(str).unique()) if "model_trust_tier" in df.columns else set()
    invalid_trust = trust_values - {"High", "Medium", "Low"}
    if invalid_trust:
        issues.append(f"invalid_trust_tiers:{','.join(sorted(invalid_trust))}")

    if len(df) < 100:
        issues.append(f"too_few_rows:{len(df)}")

    return {
        "passed": len(issues) == 0,
        "issues": issues,
        "rows": int(len(df)),
        "n_economies": int(df["iso3c"].nunique()) if "iso3c" in df.columns else 0,
        "year_min": int(years.min()) if years.notna().any() else None,
        "year_max": int(years.max()) if years.notna().any() else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate dclo_country_year_api.csv")
    parser.add_argument("--data-dir", default="data/gold")
    parser.add_argument("--output-file", default="dclo_country_year_api.csv")
    parser.add_argument("--year-min", type=int, default=2014)
    parser.add_argument("--year-max", type=int, default=2025)
    parser.add_argument("--report-file", default=None, help="Optional JSON report path")
    args = parser.parse_args()

    path = ROOT_DIR / args.data_dir / args.output_file
    result = validate_api_country_output(path=path, year_min=args.year_min, year_max=args.year_max)

    print(json.dumps(result, indent=2))
    if args.report_file:
        report_path = ROOT_DIR / args.report_file
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
