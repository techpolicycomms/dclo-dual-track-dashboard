import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd


def check_required_columns(df: pd.DataFrame, required: List[str]) -> List[str]:
    return [col for col in required if col not in df.columns]


def build_state_checks(state_df: pd.DataFrame) -> Dict[str, object]:
    issues: List[str] = []
    required = ["state_name", "year", "DCLO_score"]
    missing_cols = check_required_columns(state_df, required)
    if missing_cols:
        issues.append(f"state_missing_columns:{','.join(missing_cols)}")
        return {"passed": False, "issues": issues}

    if state_df.duplicated(subset=["state_name", "year"]).any():
        dup_count = int(state_df.duplicated(subset=["state_name", "year"]).sum())
        issues.append(f"state_duplicate_keys:{dup_count}")

    null_score = int(state_df["DCLO_score"].isna().sum())
    if null_score > 0:
        issues.append(f"state_null_dclo_score:{null_score}")

    if "DCLO_score_context_adjusted" in state_df.columns:
        null_ctx = int(state_df["DCLO_score_context_adjusted"].isna().sum())
        if null_ctx > 0:
            issues.append(f"state_null_context_adjusted_score:{null_ctx}")

    return {
        "passed": len(issues) == 0,
        "issues": issues,
        "rows": int(len(state_df)),
        "year_min": int(pd.to_numeric(state_df["year"], errors="coerce").min()),
        "year_max": int(pd.to_numeric(state_df["year"], errors="coerce").max()),
    }


def build_country_checks(country_df: pd.DataFrame, intake_df: pd.DataFrame) -> Dict[str, object]:
    issues: List[str] = []
    required = ["economy", "year", "DCLO_score", "DCLO_score_confidence_weighted", "model_trust_tier"]
    missing_cols = check_required_columns(country_df, required)
    if missing_cols:
        issues.append(f"country_missing_columns:{','.join(missing_cols)}")
        return {"passed": False, "issues": issues}

    if country_df.duplicated(subset=["economy", "year"]).any():
        dup_count = int(country_df.duplicated(subset=["economy", "year"]).sum())
        issues.append(f"country_duplicate_keys:{dup_count}")

    for col in ["DCLO_score", "DCLO_score_confidence_weighted"]:
        null_count = int(country_df[col].isna().sum())
        if null_count > 0:
            issues.append(f"country_null_{col}:{null_count}")

    allowed_tiers = {"High", "Medium", "Low"}
    found_tiers = set(country_df["model_trust_tier"].dropna().astype(str).unique().tolist())
    if not found_tiers.issubset(allowed_tiers):
        issues.append("country_invalid_trust_tier_values")

    if "n_domains_used" in country_df.columns:
        high_share = float((country_df["model_trust_tier"] == "High").mean())
        if high_share < 0.05:
            issues.append("country_low_high_trust_share")

    if not intake_df.empty and "role" in intake_df.columns and "dclo_domain" in intake_df.columns:
        core = intake_df[intake_df["role"] == "core_formative"]
        if core.empty:
            issues.append("country_no_core_formative_indicators")
        else:
            domain_counts = core.groupby("dclo_domain")["indicator_code"].nunique()
            sparse_domains = domain_counts[domain_counts < 2]
            for domain, count in sparse_domains.items():
                issues.append(f"country_sparse_core_domain:{domain}:{int(count)}")

    return {
        "passed": len(issues) == 0,
        "issues": issues,
        "rows": int(len(country_df)),
        "year_min": int(pd.to_numeric(country_df["year"], errors="coerce").min()),
        "year_max": int(pd.to_numeric(country_df["year"], errors="coerce").max()),
    }


def run(data_dir: str) -> None:
    root = Path(data_dir)
    state_path = root / "dclo_state_year.csv"
    country_path = root / "dclo_country_year.csv"
    intake_path = root / "dpi_indicator_intake.csv"

    state_df = pd.read_csv(state_path) if state_path.exists() else pd.DataFrame()
    country_df = pd.read_csv(country_path) if country_path.exists() else pd.DataFrame()
    intake_df = pd.read_csv(intake_path) if intake_path.exists() else pd.DataFrame()

    state_result = (
        build_state_checks(state_df)
        if not state_df.empty
        else {"passed": False, "issues": ["missing_state_output"], "rows": 0}
    )
    country_result = (
        build_country_checks(country_df, intake_df)
        if not country_df.empty
        else {"passed": False, "issues": ["missing_country_output"], "rows": 0}
    )

    summary = {
        "state_track": state_result,
        "country_track": country_result,
        "overall_passed": bool(state_result.get("passed", False) and country_result.get("passed", False)),
    }

    out_json = root / "dclo_standard_checks_summary.json"
    out_md = root / "dclo_standard_checks_summary.md"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# DCLO Standard Checks Summary",
        "",
        f"- overall_passed: `{summary['overall_passed']}`",
        "",
        "## State Track",
        f"- passed: `{state_result.get('passed')}`",
        f"- rows: `{state_result.get('rows', 0)}`",
        f"- years: `{state_result.get('year_min', 'n/a')}` to `{state_result.get('year_max', 'n/a')}`",
        "- issues:",
    ]
    for issue in state_result.get("issues", []):
        lines.append(f"  - {issue}")
    lines.extend(
        [
            "",
            "## Country Track",
            f"- passed: `{country_result.get('passed')}`",
            f"- rows: `{country_result.get('rows', 0)}`",
            f"- years: `{country_result.get('year_min', 'n/a')}` to `{country_result.get('year_max', 'n/a')}`",
            "- issues:",
        ]
    )
    for issue in country_result.get("issues", []):
        lines.append(f"  - {issue}")
    out_md.write_text("\n".join(lines), encoding="utf-8")

    print(f"Wrote standard checks JSON: {out_json}")
    print(f"Wrote standard checks Markdown: {out_md}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run standard QA checks for DCLO outputs")
    parser.add_argument("--data-dir", default="./data/gold", help="Directory containing DCLO gold outputs")
    args = parser.parse_args()
    run(args.data_dir)


if __name__ == "__main__":
    main()
