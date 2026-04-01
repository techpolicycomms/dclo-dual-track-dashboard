"""Comprehensive QA checks for DCLO pipeline outputs.

Validates all three tracks (state-year, country-year, causal) with checks
aligned to academic publication standards:
- Data integrity: no duplicates, no unexpected nulls, valid key ranges
- Value plausibility: score ranges, percentage bounds, trust tier validity
- Statistical adequacy: panel size, domain coverage, causal diagnostics
- Reproducibility: audit manifest verification, input checksum validation
- Outlier screening: IQR-based flagging per Tukey (1977)

References:
- OECD (2008), Handbook on Constructing Composite Indicators
- Wooldridge (2010), Econometric Analysis of Cross Section and Panel Data
- Cameron & Miller (2015), A Practitioner's Guide to Cluster-Robust Inference
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd


def check_required_columns(df: pd.DataFrame, required: List[str]) -> List[str]:
    return [col for col in required if col not in df.columns]


def _check_value_range(series: pd.Series, col_name: str, low: float, high: float) -> List[str]:
    """Check that numeric values fall within [low, high]."""
    s = pd.to_numeric(series, errors="coerce").dropna()
    issues = []
    below = int((s < low).sum())
    above = int((s > high).sum())
    if below > 0:
        issues.append(f"{col_name}_below_{low}:{below}")
    if above > 0:
        issues.append(f"{col_name}_above_{high}:{above}")
    return issues


def _check_outliers_iqr(series: pd.Series, col_name: str, multiplier: float = 3.0) -> List[str]:
    """Flag far outliers using Tukey's IQR method."""
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) < 10:
        return []
    q1 = float(s.quantile(0.25))
    q3 = float(s.quantile(0.75))
    iqr = q3 - q1
    if iqr == 0:
        return []
    lower = q1 - multiplier * iqr
    upper = q3 + multiplier * iqr
    n_outliers = int(((s < lower) | (s > upper)).sum())
    if n_outliers > 0:
        return [f"{col_name}_outliers_iqr:{n_outliers}"]
    return []


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

    # Range validation for domain scores
    domain_cols = ["ACC_score", "SKL_score", "SRV_score", "AGR_score", "ECO_score", "OUT_score"]
    for col in domain_cols:
        if col in state_df.columns:
            issues.extend(_check_outliers_iqr(state_df[col], f"state_{col}"))

    # Check z-scored indicators are within plausible range
    z_cols = [col for col in state_df.columns if col.startswith("Z_")]
    for col in z_cols:
        s = pd.to_numeric(state_df[col], errors="coerce").dropna()
        extreme = int((s.abs() > 5).sum())
        if extreme > 0:
            issues.append(f"state_{col}_extreme_zscore:{extreme}")

    # Percentage indicator range checks
    pct_cols = ["ACC_pop_hh_elec", "SKL_fem_literacy", "SRV_pop_hh_sf",
                "AGR_fem_15_24_hyg_period", "ECO_prop_hh_microfin", "OUT_prop_saving"]
    for col in pct_cols:
        if col in state_df.columns:
            issues.extend(_check_value_range(state_df[col], f"state_{col}", 0.0, 100.0))

    # Non-negative checks
    non_neg_cols = ["OUT_hh_income_monthly", "AGR_shg_member_scale", "ECO_shg_credit_scale"]
    for col in non_neg_cols:
        if col in state_df.columns:
            issues.extend(_check_value_range(state_df[col], f"state_{col}", 0.0, float("inf")))

    # Domain coverage check: at least 3 domains per observation
    available_domains = [col for col in domain_cols if col in state_df.columns]
    if available_domains:
        domains_per_row = state_df[available_domains].notna().sum(axis=1)
        sparse_rows = int((domains_per_row < 3).sum())
        if sparse_rows > 0:
            issues.append(f"state_sparse_domain_coverage:{sparse_rows}")

    return {
        "passed": len(issues) == 0,
        "issues": issues,
        "rows": int(len(state_df)),
        "year_min": int(pd.to_numeric(state_df["year"], errors="coerce").min()),
        "year_max": int(pd.to_numeric(state_df["year"], errors="coerce").max()),
        "n_states": int(state_df["state_name"].nunique()),
        "n_years": int(pd.to_numeric(state_df["year"], errors="coerce").nunique()),
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

    # Range validation for scores
    for col in ["DCLO_score", "DCLO_score_confidence_weighted"]:
        issues.extend(_check_outliers_iqr(country_df[col], f"country_{col}"))

    # Domain score outlier checks
    domain_cols = ["ACC_score", "SKL_score", "SRV_score", "AGR_score", "ECO_score", "OUT_score"]
    for col in domain_cols:
        if col in country_df.columns:
            issues.extend(_check_outliers_iqr(country_df[col], f"country_{col}"))

    # Panel balance diagnostic
    n_economies = country_df["economy"].nunique()
    n_years = pd.to_numeric(country_df["year"], errors="coerce").nunique()
    potential = n_economies * n_years
    actual = len(country_df.drop_duplicates(subset=["economy", "year"]))
    balance = actual / max(potential, 1)
    if balance < 0.3:
        issues.append(f"country_panel_unbalanced:{balance:.2f}")

    return {
        "passed": len(issues) == 0,
        "issues": issues,
        "rows": int(len(country_df)),
        "year_min": int(pd.to_numeric(country_df["year"], errors="coerce").min()),
        "year_max": int(pd.to_numeric(country_df["year"], errors="coerce").max()),
        "n_economies": n_economies,
        "n_years": n_years,
        "panel_balance": round(balance, 4),
    }


def build_causal_checks(coeff_df: pd.DataFrame, fit_df: pd.DataFrame, stability_df: pd.DataFrame) -> Dict[str, object]:
    issues: List[str] = []
    if coeff_df.empty:
        return {"passed": False, "issues": ["missing_causal_coefficients"], "rows": 0}
    if fit_df.empty:
        return {"passed": False, "issues": ["missing_causal_model_fit"], "rows": 0}

    req_coef = ["spec_kind", "spec_id", "outcome", "predictor", "coef", "std_error", "p_value_norm_approx"]
    req_fit = ["spec_kind", "spec_id", "n_obs", "n_entities", "n_years", "r2_within"]
    miss_coef = check_required_columns(coeff_df, req_coef)
    miss_fit = check_required_columns(fit_df, req_fit)
    if miss_coef:
        issues.append(f"causal_missing_coefficient_columns:{','.join(miss_coef)}")
    if miss_fit:
        issues.append(f"causal_missing_fit_columns:{','.join(miss_fit)}")
        return {"passed": False, "issues": issues, "rows": int(len(coeff_df))}

    # Minimum panel adequacy checks per Wooldridge (2010)
    baseline_fit = fit_df[fit_df["spec_kind"].astype(str).eq("baseline")]
    if baseline_fit.empty:
        issues.append("causal_missing_baseline_spec")
    else:
        b_row = baseline_fit.iloc[0]
        n_obs = int(pd.to_numeric(b_row.get("n_obs"), errors="coerce"))
        n_ent = int(pd.to_numeric(b_row.get("n_entities"), errors="coerce"))
        n_yrs = int(pd.to_numeric(b_row.get("n_years"), errors="coerce"))
        if n_obs < 200:
            issues.append("causal_low_n_obs_baseline")
        if n_ent < 25:
            issues.append("causal_low_n_entities_baseline")
        if n_yrs < 8:
            issues.append("causal_low_n_years_baseline")

        # Check degrees of freedom are adequate
        if "df_residual" in b_row.index:
            df_resid = int(pd.to_numeric(b_row.get("df_residual"), errors="coerce"))
            if df_resid < 30:
                issues.append(f"causal_low_df_residual:{df_resid}")

        # Check F-statistic significance
        if "f_statistic" in b_row.index:
            f_stat = pd.to_numeric(b_row.get("f_statistic"), errors="coerce")
            if pd.notna(f_stat) and f_stat < 4.0:
                issues.append(f"causal_weak_f_statistic:{float(f_stat):.2f}")

        # Check Durbin-Watson for residual autocorrelation
        if "durbin_watson" in b_row.index:
            dw = pd.to_numeric(b_row.get("durbin_watson"), errors="coerce")
            if pd.notna(dw) and (dw < 1.0 or dw > 3.0):
                issues.append(f"causal_residual_autocorrelation_dw:{float(dw):.2f}")

        # R-squared sanity: too high may indicate overfitting
        r2 = pd.to_numeric(b_row.get("r2_within"), errors="coerce")
        if pd.notna(r2) and r2 > 0.95:
            issues.append(f"causal_suspiciously_high_r2:{float(r2):.4f}")

    # Placebo should generally be weaker than baseline in this framework
    base_sig = coeff_df[
        coeff_df["spec_kind"].astype(str).eq("baseline")
        & coeff_df["p_value_norm_approx"].apply(pd.to_numeric, errors="coerce").lt(0.05)
    ]
    placebo_sig = coeff_df[
        coeff_df["spec_kind"].astype(str).eq("placebo")
        & coeff_df["p_value_norm_approx"].apply(pd.to_numeric, errors="coerce").lt(0.05)
    ]
    if not base_sig.empty and len(placebo_sig) >= len(base_sig):
        issues.append("causal_placebo_not_weaker_than_baseline")

    # Coefficient sign stability across robustness specs
    if "coef" in coeff_df.columns and "predictor" in coeff_df.columns:
        for predictor in coeff_df["predictor"].unique():
            pred_coefs = coeff_df[coeff_df["predictor"] == predictor]
            non_placebo = pred_coefs[pred_coefs["spec_kind"].astype(str) != "placebo"]
            if len(non_placebo) >= 2:
                signs = pd.to_numeric(non_placebo["coef"], errors="coerce").dropna().apply(np.sign)
                if signs.nunique() > 1:
                    issues.append(f"causal_sign_instability:{predictor}")

    # Stability diagnostics should not be degenerate
    if stability_df.empty:
        issues.append("causal_missing_rank_stability")
    else:
        if "sd_rank" not in stability_df.columns:
            issues.append("causal_missing_stability_sd_rank")
        else:
            mean_sd = pd.to_numeric(stability_df["sd_rank"], errors="coerce").mean()
            if pd.notna(mean_sd) and float(mean_sd) <= 0:
                issues.append("causal_degenerate_rank_stability")

    return {
        "passed": len(issues) == 0,
        "issues": issues,
        "rows": int(len(coeff_df)),
        "n_specs": int(fit_df["spec_id"].nunique()),
    }


def build_manifest_checks(gold_dir: Path) -> Dict[str, object]:
    """Verify that audit manifests exist and are internally consistent."""
    issues: List[str] = []
    manifests_found = 0

    manifest_files = [
        "dclo_state_year_audit_manifest.json",
        "dclo_country_year_audit_manifest.json",
        "dclo_causal_panel_audit_manifest.json",
    ]

    for mf in manifest_files:
        mp = gold_dir / mf
        if not mp.exists():
            issues.append(f"missing_manifest:{mf}")
            continue
        manifests_found += 1
        try:
            manifest = json.loads(mp.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            issues.append(f"invalid_manifest:{mf}")
            continue

        # Verify manifest has required fields
        for field in ["pipeline", "run_id", "environment", "inputs", "stages", "outputs", "status"]:
            if field not in manifest:
                issues.append(f"manifest_missing_field:{mf}:{field}")

        # Check that all inputs had checksums computed
        for input_name, input_info in manifest.get("inputs", {}).items():
            if not input_info.get("exists"):
                issues.append(f"manifest_input_missing:{mf}:{input_name}")
            elif "sha256" not in input_info:
                issues.append(f"manifest_input_no_checksum:{mf}:{input_name}")

        # Check that all outputs had checksums computed
        for output_name, output_info in manifest.get("outputs", {}).items():
            if not output_info.get("exists"):
                issues.append(f"manifest_output_missing:{mf}:{output_name}")
            elif "sha256" not in output_info:
                issues.append(f"manifest_output_no_checksum:{mf}:{output_name}")

        # Check for accounting warnings in stages
        for stage in manifest.get("stages", []):
            if "accounting_warning" in stage:
                issues.append(f"manifest_accounting_warning:{mf}:{stage['stage']}")

        # Verify git commit is recorded
        env = manifest.get("environment", {})
        if not env.get("git_commit"):
            issues.append(f"manifest_no_git_commit:{mf}")

        # Verify status is completed
        if manifest.get("status") != "completed":
            issues.append(f"manifest_incomplete:{mf}")

    return {
        "passed": len(issues) == 0,
        "issues": issues,
        "manifests_found": manifests_found,
        "manifests_expected": len(manifest_files),
    }


def build_verification_checks(gold_dir: Path) -> Dict[str, object]:
    """Check that verification reports exist and contain no errors."""
    issues: List[str] = []

    verification_files = [
        "dclo_state_year_verification.json",
        "dclo_country_year_verification.json",
        "dclo_causal_panel_verification.json",
    ]

    for vf in verification_files:
        vp = gold_dir / vf
        if not vp.exists():
            issues.append(f"missing_verification:{vf}")
            continue
        try:
            report = json.loads(vp.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            issues.append(f"invalid_verification:{vf}")
            continue

        # For nested verification reports (state/country), check each section
        if isinstance(report, dict):
            for section_name, section in report.items():
                if isinstance(section, dict) and "issues" in section:
                    errors = [i for i in section.get("issues", []) if i.get("severity") == "error"]
                    if errors:
                        issues.append(f"verification_errors:{vf}:{section_name}:{len(errors)}")

    return {
        "passed": len(issues) == 0,
        "issues": issues,
    }


def run(data_dir: str) -> None:
    root = Path(data_dir)
    state_path = root / "dclo_state_year.csv"
    country_path = root / "dclo_country_year.csv"
    intake_path = root / "dpi_indicator_intake.csv"
    causal_coef_path = root / "dclo_causal_coefficients.csv"
    causal_fit_path = root / "dclo_causal_model_fit.csv"
    rank_stability_path = root / "dclo_rank_stability.csv"

    state_df = pd.read_csv(state_path) if state_path.exists() else pd.DataFrame()
    country_df = pd.read_csv(country_path) if country_path.exists() else pd.DataFrame()
    intake_df = pd.read_csv(intake_path) if intake_path.exists() else pd.DataFrame()
    causal_coef_df = pd.read_csv(causal_coef_path) if causal_coef_path.exists() else pd.DataFrame()
    causal_fit_df = pd.read_csv(causal_fit_path) if causal_fit_path.exists() else pd.DataFrame()
    rank_stability_df = pd.read_csv(rank_stability_path) if rank_stability_path.exists() else pd.DataFrame()

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
    causal_result = build_causal_checks(causal_coef_df, causal_fit_df, rank_stability_df)
    manifest_result = build_manifest_checks(root)
    verification_result = build_verification_checks(root)

    summary: Dict[str, Any] = {
        "state_track": state_result,
        "country_track": country_result,
        "causal_track": causal_result,
        "audit_manifests": manifest_result,
        "verification_reports": verification_result,
        "overall_passed": bool(
            state_result.get("passed", False)
            and country_result.get("passed", False)
            and causal_result.get("passed", False)
            and manifest_result.get("passed", False)
            and verification_result.get("passed", False)
        ),
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
        f"- states: `{state_result.get('n_states', 'n/a')}`",
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
            f"- economies: `{country_result.get('n_economies', 'n/a')}`",
            f"- years: `{country_result.get('year_min', 'n/a')}` to `{country_result.get('year_max', 'n/a')}`",
            f"- panel_balance: `{country_result.get('panel_balance', 'n/a')}`",
            "- issues:",
        ]
    )
    for issue in country_result.get("issues", []):
        lines.append(f"  - {issue}")
    lines.extend(
        [
            "",
            "## Causal Track",
            f"- passed: `{causal_result.get('passed')}`",
            f"- coefficient_rows: `{causal_result.get('rows', 0)}`",
            f"- n_specs: `{causal_result.get('n_specs', 'n/a')}`",
            "- issues:",
        ]
    )
    for issue in causal_result.get("issues", []):
        lines.append(f"  - {issue}")
    lines.extend(
        [
            "",
            "## Audit Manifests",
            f"- passed: `{manifest_result.get('passed')}`",
            f"- manifests_found: `{manifest_result.get('manifests_found', 0)}/{manifest_result.get('manifests_expected', 0)}`",
            "- issues:",
        ]
    )
    for issue in manifest_result.get("issues", []):
        lines.append(f"  - {issue}")
    lines.extend(
        [
            "",
            "## Verification Reports",
            f"- passed: `{verification_result.get('passed')}`",
            "- issues:",
        ]
    )
    for issue in verification_result.get("issues", []):
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
