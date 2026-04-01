import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from quality.audit_logger import AuditLogger
from quality.data_verification import (
    VerificationResult,
    verify_domain_coverage,
    verify_minimum_sample_size,
    verify_no_duplicates,
    verify_no_nulls,
    verify_numeric_coercion,
    verify_outliers_iqr,
    verify_required_columns,
    verify_score_completeness,
    verify_value_ranges,
    verify_year_coverage,
)


DOMAIN_SCORE_COLUMNS = [
    "ACC_score", "SKL_score", "SRV_score",
    "AGR_score", "ECO_score", "OUT_score",
]


def read_config(config_path: str) -> Dict[str, object]:
    with open(config_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def get_enabled_sources(config: Dict[str, object]) -> Dict[str, str]:
    enabled: Dict[str, str] = {}
    for source in config.get("sources", []):
        if source.get("enabled"):
            enabled[source["source_id"]] = source["path"]
    return enabled


def parse_year_from_period(value: object) -> Optional[int]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    if len(text) >= 4 and text[:4].isdigit():
        return int(text[:4])
    return None


def zscore(series: pd.Series) -> pd.Series:
    """Z-score normalization using sample standard deviation (ddof=1).

    Uses Bessel's correction (ddof=1) as standard for sample data per
    Nunnally & Bernstein (1994). Falls back to zero vector when std is
    zero or undefined (constant series).
    """
    std = series.std(ddof=1)
    if std == 0 or pd.isna(std):
        return pd.Series([0.0] * len(series), index=series.index)
    return (series - series.mean()) / std


def apply_dpi_context(df: pd.DataFrame, config: Dict[str, object], audit: AuditLogger) -> pd.DataFrame:
    context_cfg = config.get("dpi_context", {})
    if not context_cfg or not bool(context_cfg.get("enabled", False)):
        return df

    source_path = context_cfg.get("source_path")
    if not source_path:
        return df

    economy_name = str(context_cfg.get("economy_name", "India")).strip().lower()
    merge_fields = context_cfg.get("merge_fields", ["dpi_composite_v2", "dpi_confidence_score", "coverage_ratio"])
    include_context_adjusted = bool(context_cfg.get("include_context_adjusted_score", True))
    alpha = float(context_cfg.get("context_alpha", 0.15))

    audit.record_input("dpi_context_source", source_path)
    audit.record_parameter("dpi_context_alpha", alpha)
    audit.record_parameter("dpi_context_economy", economy_name)

    ctx = pd.read_csv(source_path)
    if "economy" not in ctx.columns or "year" not in ctx.columns:
        return df

    ctx["economy"] = ctx["economy"].astype(str).str.strip().str.lower()
    ctx = ctx[ctx["economy"] == economy_name].copy()
    if ctx.empty:
        return df

    cols = ["year"] + [col for col in merge_fields if col in ctx.columns]
    ctx = ctx[cols].copy()
    ctx["year"] = pd.to_numeric(ctx["year"], errors="coerce").astype("Int64")
    for col in cols:
        if col != "year":
            ctx[f"CTX_{col}"] = pd.to_numeric(ctx[col], errors="coerce")
    ctx = ctx.drop(columns=[col for col in cols if col != "year"])

    out = df.copy()
    out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("Int64")
    rows_before = len(out)
    out = out.merge(ctx, on="year", how="left")
    audit.record_stage("apply_dpi_context", rows_in=rows_before, rows_out=len(out), rows_dropped=0,
                       notes=f"Left-joined DPI context for {economy_name}, alpha={alpha}")

    if "CTX_dpi_composite_v2" in out.columns:
        out["Z_CTX_dpi_composite_v2"] = zscore(out["CTX_dpi_composite_v2"])
        if include_context_adjusted and "DCLO_score" in out.columns:
            out["DCLO_score_context_adjusted"] = out["DCLO_score"] + alpha * out["Z_CTX_dpi_composite_v2"]
    return out


def normalize_column_name(name: str) -> str:
    return (
        str(name)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
        .replace("(", "")
        .replace(")", "")
    )


def find_first_matching_column(columns: List[str], candidates: List[str]) -> Optional[str]:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return None


def find_column_by_tokens(columns: List[str], include_tokens: List[str], exclude_tokens: Optional[List[str]] = None) -> Optional[str]:
    exclude_tokens = exclude_tokens or []
    for col in columns:
        if any(token in col for token in include_tokens) and not any(token in col for token in exclude_tokens):
            return col
    return None


def aggregate_nfis(path: str, audit: AuditLogger) -> pd.DataFrame:
    audit.record_input("nafis", path)
    df = pd.read_csv(path)
    rows_raw = len(df)

    vr = VerificationResult("nafis")
    required = ["year", "state_name", "prop_hh_microfin", "hh_income_monthly", "prop_saving"]
    vr = verify_required_columns(df, required, vr)
    if not vr.passed:
        raise ValueError(f"NAFIS verification failed: {vr.issues}")

    out = df[required].copy()

    # Auditable numeric coercion
    numeric_cols = ["prop_hh_microfin", "hh_income_monthly", "prop_saving"]
    out, vr = verify_numeric_coercion(out, numeric_cols, vr)

    out["year"] = out["year"].map(parse_year_from_period)
    rows_before_drop = len(out)
    out = out.dropna(subset=["state_name", "year"])
    n_dropped_null_keys = rows_before_drop - len(out)

    out["state_name"] = out["state_name"].astype(str).str.strip()
    out["ECO_prop_hh_microfin"] = out["prop_hh_microfin"]
    out["OUT_hh_income_monthly"] = out["hh_income_monthly"]
    out["OUT_prop_saving"] = out["prop_saving"]

    # Range validation
    out, vr = verify_value_ranges(out, vr)
    vr = verify_outliers_iqr(out, ["ECO_prop_hh_microfin", "OUT_hh_income_monthly", "OUT_prop_saving"], vr)

    grouped = (
        out.groupby(["state_name", "year"], as_index=False)[
            ["ECO_prop_hh_microfin", "OUT_hh_income_monthly", "OUT_prop_saving"]
        ]
        .mean()
    )

    audit.record_stage(
        "aggregate_nafis",
        rows_in=rows_raw,
        rows_out=len(grouped),
        rows_dropped=rows_raw - len(out),
        drop_reasons={"null_keys": n_dropped_null_keys, **{d["reason"]: d["count"] for d in vr.dropped_records}},
        notes=f"Verification: {vr.checks_passed}/{vr.checks_run} checks passed, {vr.n_issues} issues",
    )
    return grouped


def aggregate_nfhs(path: str, audit: AuditLogger) -> pd.DataFrame:
    audit.record_input("nfhs", path)
    df = pd.read_csv(path)
    rows_raw = len(df)

    vr = VerificationResult("nfhs")
    required = ["year", "state_name", "pop_hh_elec", "fem_literacy", "pop_hh_sf", "fem_15_24_hyg_period"]
    vr = verify_required_columns(df, required, vr)
    if not vr.passed:
        raise ValueError(f"NFHS verification failed: {vr.issues}")

    out = df[required].copy()
    numeric_cols = ["pop_hh_elec", "fem_literacy", "pop_hh_sf", "fem_15_24_hyg_period"]
    out, vr = verify_numeric_coercion(out, numeric_cols, vr)

    out["year"] = out["year"].map(parse_year_from_period)
    rows_before_drop = len(out)
    out = out.dropna(subset=["state_name", "year"])
    n_dropped_null_keys = rows_before_drop - len(out)

    out["state_name"] = out["state_name"].astype(str).str.strip()
    out["ACC_pop_hh_elec"] = out["pop_hh_elec"]
    out["SKL_fem_literacy"] = out["fem_literacy"]
    out["SRV_pop_hh_sf"] = out["pop_hh_sf"]
    out["AGR_fem_15_24_hyg_period"] = out["fem_15_24_hyg_period"]

    # Range validation (percentage indicators)
    out, vr = verify_value_ranges(out, vr)
    vr = verify_outliers_iqr(out, ["ACC_pop_hh_elec", "SKL_fem_literacy", "SRV_pop_hh_sf", "AGR_fem_15_24_hyg_period"], vr)

    grouped = (
        out.groupby(["state_name", "year"], as_index=False)[
            ["ACC_pop_hh_elec", "SKL_fem_literacy", "SRV_pop_hh_sf", "AGR_fem_15_24_hyg_period"]
        ]
        .mean()
    )

    audit.record_stage(
        "aggregate_nfhs",
        rows_in=rows_raw,
        rows_out=len(grouped),
        rows_dropped=rows_raw - len(out),
        drop_reasons={"null_keys": n_dropped_null_keys, **{d["reason"]: d["count"] for d in vr.dropped_records}},
        notes=f"Verification: {vr.checks_passed}/{vr.checks_run} checks passed, {vr.n_issues} issues",
    )
    return grouped


def aggregate_shg_chunked(path: str, audit: AuditLogger, chunksize: int = 200000) -> pd.DataFrame:
    audit.record_input("shg_profile", path)
    aggregations: List[pd.DataFrame] = []
    total_rows_read = 0
    header_df = pd.read_csv(path, nrows=0)
    normalized_to_original: Dict[str, str] = {
        normalize_column_name(col): col for col in header_df.columns
    }
    normalized_cols = list(normalized_to_original.keys())

    state_col_norm = find_first_matching_column(
        normalized_cols,
        ["state_name", "state", "state_ut", "state_ut_name", "state_name_ut", "state_or_ut"],
    )
    year_col_norm = find_first_matching_column(
        normalized_cols,
        ["year", "financial_year", "fy", "month", "date", "as_on_date"],
    )
    member_col_norm = find_column_by_tokens(normalized_cols, ["member"], ["id"])
    credit_col_norm = find_column_by_tokens(normalized_cols, ["loan", "credit", "saving", "finance", "amount"], ["id"])

    if state_col_norm is None:
        audit.record_stage("aggregate_shg_chunked", rows_in=0, rows_out=0, rows_dropped=0,
                           notes="No state column found in SHG file, skipping")
        return pd.DataFrame(columns=["state_name", "year", "AGR_shg_member_scale", "ECO_shg_credit_scale"])

    selected_norm_cols = [state_col_norm]
    for col in [year_col_norm, member_col_norm, credit_col_norm]:
        if col and col not in selected_norm_cols:
            selected_norm_cols.append(col)
    usecols = [normalized_to_original[col] for col in selected_norm_cols]

    for chunk in pd.read_csv(path, chunksize=chunksize, low_memory=False, usecols=usecols):
        chunk.columns = [normalize_column_name(col) for col in chunk.columns]
        total_rows_read += len(chunk)

        work = pd.DataFrame()
        work["state_name"] = chunk[state_col_norm].astype(str).str.strip()
        if year_col_norm and year_col_norm in chunk.columns:
            work["year"] = chunk[year_col_norm].map(parse_year_from_period)
        else:
            work["year"] = pd.NA

        metric_cols: List[str] = []
        if member_col_norm and member_col_norm in chunk.columns:
            work["AGR_shg_member_scale"] = pd.to_numeric(chunk[member_col_norm], errors="coerce")
            metric_cols.append("AGR_shg_member_scale")
        if credit_col_norm and credit_col_norm in chunk.columns and credit_col_norm != member_col_norm:
            work["ECO_shg_credit_scale"] = pd.to_numeric(chunk[credit_col_norm], errors="coerce")
            metric_cols.append("ECO_shg_credit_scale")

        if not metric_cols:
            continue

        grouped = work.dropna(subset=["state_name"]).groupby(
            ["state_name", "year"], dropna=False, as_index=False
        )[metric_cols].sum()
        aggregations.append(grouped)

    if not aggregations:
        audit.record_stage("aggregate_shg_chunked", rows_in=total_rows_read, rows_out=0, rows_dropped=total_rows_read,
                           notes="No metric columns found in SHG chunks")
        return pd.DataFrame(columns=["state_name", "year", "AGR_shg_member_scale", "ECO_shg_credit_scale"])

    out = pd.concat(aggregations, ignore_index=True)
    metric_cols = [col for col in ["AGR_shg_member_scale", "ECO_shg_credit_scale"] if col in out.columns]
    out = out.groupby(["state_name", "year"], dropna=False, as_index=False)[metric_cols].sum()
    out["year"] = out["year"].map(parse_year_from_period)

    audit.record_stage(
        "aggregate_shg_chunked",
        rows_in=total_rows_read,
        rows_out=len(out),
        rows_dropped=total_rows_read - len(out),
        notes=f"Processed in {chunksize}-row chunks",
    )
    return out


def aggregate_national_upi(path: str, audit: AuditLogger) -> pd.DataFrame:
    audit.record_input("upi_transactions", path)
    df = pd.read_csv(path)
    rows_raw = len(df)

    vr = VerificationResult("upi_transactions")
    required = ["month", "total_vol", "total_val", "p2p_vol", "p2m_vol"]
    vr = verify_required_columns(df, required, vr)
    if not vr.passed:
        raise ValueError(f"UPI verification failed: {vr.issues}")

    out = df[required].copy()
    numeric_cols = ["total_vol", "total_val", "p2p_vol", "p2m_vol"]
    out, vr = verify_numeric_coercion(out, numeric_cols, vr)

    out["year"] = out["month"].map(parse_year_from_period)
    out, vr = verify_value_ranges(out, vr)

    grouped = (
        out.groupby("year", as_index=False)[numeric_cols]
        .sum()
        .rename(columns={
            "total_vol": "NAT_upi_total_vol",
            "total_val": "NAT_upi_total_val",
            "p2p_vol": "NAT_upi_p2p_vol",
            "p2m_vol": "NAT_upi_p2m_vol",
        })
    )

    audit.record_stage("aggregate_national_upi", rows_in=rows_raw, rows_out=len(grouped),
                       rows_dropped=rows_raw - len(grouped),
                       notes=f"Verification: {vr.checks_passed}/{vr.checks_run} checks passed")
    return grouped


def aggregate_national_internet_banking(path: str, audit: AuditLogger) -> pd.DataFrame:
    audit.record_input("internet_banking", path)
    df = pd.read_csv(path)
    rows_raw = len(df)

    vr = VerificationResult("internet_banking")
    required = ["month", "no_of_transactions", "amt_of_transactions", "active_users"]
    vr = verify_required_columns(df, required, vr)
    if not vr.passed:
        raise ValueError(f"Internet banking verification failed: {vr.issues}")

    out = df[required].copy()
    numeric_cols = ["no_of_transactions", "amt_of_transactions", "active_users"]
    out, vr = verify_numeric_coercion(out, numeric_cols, vr)

    out["year"] = out["month"].map(parse_year_from_period)
    out, vr = verify_value_ranges(out, vr)

    grouped = (
        out.groupby("year", as_index=False)[numeric_cols]
        .sum()
        .rename(columns={
            "no_of_transactions": "NAT_ib_no_of_transactions",
            "amt_of_transactions": "NAT_ib_amt_of_transactions",
            "active_users": "NAT_ib_active_users",
        })
    )

    audit.record_stage("aggregate_national_internet_banking", rows_in=rows_raw, rows_out=len(grouped),
                       rows_dropped=rows_raw - len(grouped),
                       notes=f"Verification: {vr.checks_passed}/{vr.checks_run} checks passed")
    return grouped


def compute_domain_scores(df: pd.DataFrame, audit: AuditLogger) -> pd.DataFrame:
    rows_in = len(df)
    indicator_columns: List[str] = [
        "ACC_pop_hh_elec",
        "SKL_fem_literacy",
        "SRV_pop_hh_sf",
        "AGR_fem_15_24_hyg_period",
        "AGR_shg_member_scale",
        "ECO_prop_hh_microfin",
        "ECO_shg_credit_scale",
        "OUT_hh_income_monthly",
        "OUT_prop_saving",
    ]
    for column in indicator_columns:
        if column in df.columns:
            df[f"Z_{column}"] = zscore(df[column])

    domain_map = {
        "ACC_score": ["Z_ACC_pop_hh_elec"],
        "SKL_score": ["Z_SKL_fem_literacy"],
        "SRV_score": ["Z_SRV_pop_hh_sf"],
        "AGR_score": ["Z_AGR_fem_15_24_hyg_period", "Z_AGR_shg_member_scale"],
        "ECO_score": ["Z_ECO_prop_hh_microfin", "Z_ECO_shg_credit_scale"],
        "OUT_score": ["Z_OUT_hh_income_monthly", "Z_OUT_prop_saving"],
    }

    for domain_score, cols in domain_map.items():
        available = [col for col in cols if col in df.columns]
        if not available:
            df[domain_score] = pd.NA
            continue
        df[domain_score] = df[available].mean(axis=1, skipna=True)

    domain_scores = list(domain_map.keys())
    df["DCLO_score"] = df[domain_scores].mean(axis=1, skipna=True)

    # Record which indicators actually contributed
    indicators_used = [col for col in indicator_columns if col in df.columns]
    audit.record_parameter("indicators_used_in_scoring", indicators_used)
    audit.record_parameter("n_indicators_used", len(indicators_used))
    audit.record_parameter("normalization_method", "z-score (ddof=1, Bessel corrected)")
    audit.record_parameter("domain_aggregation", "equal-weighted mean of z-scored indicators")
    audit.record_parameter("composite_aggregation", "equal-weighted mean of domain scores")

    audit.record_stage("compute_domain_scores", rows_in=rows_in, rows_out=len(df), rows_dropped=0,
                       notes=f"Used {len(indicators_used)}/{len(indicator_columns)} indicators across {len(domain_map)} domains")
    return df


def run(config_path: str) -> None:
    config = read_config(config_path)
    enabled_sources = get_enabled_sources(config)

    audit = AuditLogger(pipeline_name="build_dclo_local")
    audit.record_config(config)

    if "nafis" not in enabled_sources or "nfhs" not in enabled_sources:
        raise ValueError("Both 'nafis' and 'nfhs' sources must be enabled for baseline DCLO")

    base = aggregate_nfis(enabled_sources["nafis"], audit)
    nfhs = aggregate_nfhs(enabled_sources["nfhs"], audit)

    merged = base.merge(nfhs, on=["state_name", "year"], how="outer")
    audit.record_stage("merge_nafis_nfhs", rows_in=len(base) + len(nfhs), rows_out=len(merged), rows_dropped=0,
                       notes="Outer join on (state_name, year)")

    if "upi_transactions" in enabled_sources:
        upi = aggregate_national_upi(enabled_sources["upi_transactions"], audit)
        rows_before = len(merged)
        merged = merged.merge(upi, on="year", how="left")
        audit.record_stage("merge_upi", rows_in=rows_before, rows_out=len(merged), rows_dropped=0,
                           notes="Left join national UPI on year")

    if "internet_banking" in enabled_sources:
        ib = aggregate_national_internet_banking(enabled_sources["internet_banking"], audit)
        rows_before = len(merged)
        merged = merged.merge(ib, on="year", how="left")
        audit.record_stage("merge_internet_banking", rows_in=rows_before, rows_out=len(merged), rows_dropped=0,
                           notes="Left join national internet banking on year")

    if "shg_profile" in enabled_sources:
        shg = aggregate_shg_chunked(enabled_sources["shg_profile"], audit)
        shg = shg.dropna(subset=["state_name"])
        shg["state_name"] = shg["state_name"].astype(str).str.strip()
        shg["year"] = pd.to_numeric(shg["year"], errors="coerce").astype("Int64")
        merged["year"] = pd.to_numeric(merged["year"], errors="coerce").astype("Int64")
        rows_before = len(merged)
        merged = merged.merge(shg, on=["state_name", "year"], how="left")
        shg_metric_cols = [col for col in ["AGR_shg_member_scale", "ECO_shg_credit_scale"] if col in shg.columns]
        if shg_metric_cols:
            shg_state = shg.groupby("state_name", as_index=False)[shg_metric_cols].sum()
            rename_map = {col: f"{col}_state_fallback" for col in shg_metric_cols}
            shg_state = shg_state.rename(columns=rename_map)
            merged = merged.merge(shg_state, on="state_name", how="left")
            for col in shg_metric_cols:
                fallback_col = f"{col}_state_fallback"
                if fallback_col in merged.columns:
                    merged[col] = merged[col].fillna(merged[fallback_col])
                    merged = merged.drop(columns=[fallback_col])
        audit.record_stage("merge_shg", rows_in=rows_before, rows_out=len(merged), rows_dropped=0,
                           notes="Left join SHG with state-level fallback for missing years")

    # Pre-scoring verification on merged data
    vr = VerificationResult("merged_pre_scoring")
    vr = verify_minimum_sample_size(merged, min_rows=10, result=vr, context="merged panel before scoring")
    merged, vr = verify_no_duplicates(merged, ["state_name", "year"], vr)
    vr = verify_year_coverage(merged, "year", expected_min=2000, expected_max=2030, result=vr)
    vr = verify_no_nulls(merged, ["state_name", "year"], vr)

    scored = compute_domain_scores(merged, audit)

    # Post-scoring verification
    vr_post = VerificationResult("post_scoring")
    vr_post = verify_score_completeness(scored, "DCLO_score", "state_name", vr_post)
    vr_post = verify_domain_coverage(scored, DOMAIN_SCORE_COLUMNS, vr_post, min_domains_available=3)
    scored, vr_post = verify_value_ranges(scored, vr_post)

    scored = apply_dpi_context(scored, config, audit)
    scored = scored.sort_values(["year", "state_name"]).reset_index(drop=True)

    output_cfg = config.get("output", {})
    data_dir = Path(output_cfg.get("data_dir", "./data"))
    output_name = output_cfg.get("output_file", "dclo_state_year.csv")

    gold_dir = data_dir / "gold"
    gold_dir.mkdir(parents=True, exist_ok=True)
    out_path = gold_dir / output_name
    scored.to_csv(out_path, index=False)
    audit.record_output("dclo_state_year", str(out_path))

    # Write verification reports alongside output
    verification_report = {
        "pre_scoring": vr.to_dict(),
        "post_scoring": vr_post.to_dict(),
    }
    verification_path = gold_dir / "dclo_state_year_verification.json"
    import json
    verification_path.write_text(json.dumps(verification_report, indent=2, default=str), encoding="utf-8")
    audit.record_output("verification_report", str(verification_path))

    # Write audit manifest
    manifest_path = gold_dir / "dclo_state_year_audit_manifest.json"
    audit.write_manifest(str(manifest_path))

    print(f"Wrote {len(scored)} rows to {out_path}")
    print(f"Wrote verification report: {verification_path}")
    print(f"Wrote audit manifest: {manifest_path}")
    for line in audit.get_summary_lines():
        print(f"  {line}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build DCLO from local downloaded files")
    parser.add_argument("--config", required=True, help="Path to local sources YAML config")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
