import argparse
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import yaml


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
    std = series.std(ddof=0)
    if std == 0 or pd.isna(std):
        return pd.Series([0.0] * len(series), index=series.index)
    return (series - series.mean()) / std


def apply_dpi_context(df: pd.DataFrame, config: Dict[str, object]) -> pd.DataFrame:
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
    out = out.merge(ctx, on="year", how="left")

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


def aggregate_nfis(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = ["year", "state_name", "prop_hh_microfin", "hh_income_monthly", "prop_saving"]
    for column in required:
        if column not in df.columns:
            raise ValueError(f"Missing expected NAFIS column: {column}")

    out = df[required].copy()
    out["year"] = out["year"].map(parse_year_from_period)
    out = out.dropna(subset=["state_name", "year"])
    out["state_name"] = out["state_name"].astype(str).str.strip()
    out["ECO_prop_hh_microfin"] = pd.to_numeric(out["prop_hh_microfin"], errors="coerce")
    out["OUT_hh_income_monthly"] = pd.to_numeric(out["hh_income_monthly"], errors="coerce")
    out["OUT_prop_saving"] = pd.to_numeric(out["prop_saving"], errors="coerce")

    grouped = (
        out.groupby(["state_name", "year"], as_index=False)[
            ["ECO_prop_hh_microfin", "OUT_hh_income_monthly", "OUT_prop_saving"]
        ]
        .mean()
    )
    return grouped


def aggregate_nfhs(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = [
        "year",
        "state_name",
        "pop_hh_elec",
        "fem_literacy",
        "pop_hh_sf",
        "fem_15_24_hyg_period",
    ]
    for column in required:
        if column not in df.columns:
            raise ValueError(f"Missing expected NFHS column: {column}")

    out = df[required].copy()
    out["year"] = out["year"].map(parse_year_from_period)
    out = out.dropna(subset=["state_name", "year"])
    out["state_name"] = out["state_name"].astype(str).str.strip()

    out["ACC_pop_hh_elec"] = pd.to_numeric(out["pop_hh_elec"], errors="coerce")
    out["SKL_fem_literacy"] = pd.to_numeric(out["fem_literacy"], errors="coerce")
    out["SRV_pop_hh_sf"] = pd.to_numeric(out["pop_hh_sf"], errors="coerce")
    out["AGR_fem_15_24_hyg_period"] = pd.to_numeric(out["fem_15_24_hyg_period"], errors="coerce")

    grouped = (
        out.groupby(["state_name", "year"], as_index=False)[
            ["ACC_pop_hh_elec", "SKL_fem_literacy", "SRV_pop_hh_sf", "AGR_fem_15_24_hyg_period"]
        ]
        .mean()
    )
    return grouped


def aggregate_shg_chunked(path: str, chunksize: int = 200000) -> pd.DataFrame:
    aggregations: List[pd.DataFrame] = []
    header_df = pd.read_csv(path, nrows=0)
    normalized_to_original: Dict[str, str] = {
        normalize_column_name(col): col for col in header_df.columns
    }
    normalized_cols = list(normalized_to_original.keys())

    state_col_norm = find_first_matching_column(
        normalized_cols,
        [
            "state_name",
            "state",
            "state_ut",
            "state_ut_name",
            "state_name_ut",
            "state_or_ut",
        ],
    )
    year_col_norm = find_first_matching_column(
        normalized_cols,
        ["year", "financial_year", "fy", "month", "date", "as_on_date"],
    )
    member_col_norm = find_column_by_tokens(normalized_cols, ["member"], ["id"])
    credit_col_norm = find_column_by_tokens(normalized_cols, ["loan", "credit", "saving", "finance", "amount"], ["id"])

    if state_col_norm is None:
        return pd.DataFrame(columns=["state_name", "year", "AGR_shg_member_scale", "ECO_shg_credit_scale"])

    selected_norm_cols = [state_col_norm]
    for col in [year_col_norm, member_col_norm, credit_col_norm]:
        if col and col not in selected_norm_cols:
            selected_norm_cols.append(col)
    usecols = [normalized_to_original[col] for col in selected_norm_cols]

    for chunk in pd.read_csv(path, chunksize=chunksize, low_memory=False, usecols=usecols):
        chunk.columns = [normalize_column_name(col) for col in chunk.columns]

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
        return pd.DataFrame(columns=["state_name", "year", "AGR_shg_member_scale", "ECO_shg_credit_scale"])

    out = pd.concat(aggregations, ignore_index=True)
    metric_cols = [col for col in ["AGR_shg_member_scale", "ECO_shg_credit_scale"] if col in out.columns]
    out = out.groupby(["state_name", "year"], dropna=False, as_index=False)[metric_cols].sum()
    out["year"] = out["year"].map(parse_year_from_period)
    return out


def aggregate_national_upi(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = ["month", "total_vol", "total_val", "p2p_vol", "p2m_vol"]
    for column in required:
        if column not in df.columns:
            raise ValueError(f"Missing expected UPI column: {column}")

    out = df[required].copy()
    out["year"] = out["month"].map(parse_year_from_period)
    for col in ["total_vol", "total_val", "p2p_vol", "p2m_vol"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    grouped = (
        out.groupby("year", as_index=False)[["total_vol", "total_val", "p2p_vol", "p2m_vol"]]
        .sum()
        .rename(
            columns={
                "total_vol": "NAT_upi_total_vol",
                "total_val": "NAT_upi_total_val",
                "p2p_vol": "NAT_upi_p2p_vol",
                "p2m_vol": "NAT_upi_p2m_vol",
            }
        )
    )
    return grouped


def aggregate_national_internet_banking(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = ["month", "no_of_transactions", "amt_of_transactions", "active_users"]
    for column in required:
        if column not in df.columns:
            raise ValueError(f"Missing expected internet banking column: {column}")

    out = df[required].copy()
    out["year"] = out["month"].map(parse_year_from_period)
    for col in ["no_of_transactions", "amt_of_transactions", "active_users"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    grouped = (
        out.groupby("year", as_index=False)[["no_of_transactions", "amt_of_transactions", "active_users"]]
        .sum()
        .rename(
            columns={
                "no_of_transactions": "NAT_ib_no_of_transactions",
                "amt_of_transactions": "NAT_ib_amt_of_transactions",
                "active_users": "NAT_ib_active_users",
            }
        )
    )
    return grouped


def compute_domain_scores(df: pd.DataFrame) -> pd.DataFrame:
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

    domain_scores = [key for key in domain_map.keys()]
    df["DCLO_score"] = df[domain_scores].mean(axis=1, skipna=True)
    return df


def run(config_path: str) -> None:
    config = read_config(config_path)
    enabled_sources = get_enabled_sources(config)

    if "nafis" not in enabled_sources or "nfhs" not in enabled_sources:
        raise ValueError("Both 'nafis' and 'nfhs' sources must be enabled for baseline DCLO")

    base = aggregate_nfis(enabled_sources["nafis"])
    nfhs = aggregate_nfhs(enabled_sources["nfhs"])

    merged = base.merge(nfhs, on=["state_name", "year"], how="outer")

    if "upi_transactions" in enabled_sources:
        upi = aggregate_national_upi(enabled_sources["upi_transactions"])
        merged = merged.merge(upi, on="year", how="left")

    if "internet_banking" in enabled_sources:
        ib = aggregate_national_internet_banking(enabled_sources["internet_banking"])
        merged = merged.merge(ib, on="year", how="left")

    if "shg_profile" in enabled_sources:
        shg = aggregate_shg_chunked(enabled_sources["shg_profile"])
        shg = shg.dropna(subset=["state_name"])
        shg["state_name"] = shg["state_name"].astype(str).str.strip()
        shg["year"] = pd.to_numeric(shg["year"], errors="coerce").astype("Int64")
        merged["year"] = pd.to_numeric(merged["year"], errors="coerce").astype("Int64")
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

    scored = compute_domain_scores(merged)
    scored = apply_dpi_context(scored, config)
    scored = scored.sort_values(["year", "state_name"]).reset_index(drop=True)

    output_cfg = config.get("output", {})
    data_dir = Path(output_cfg.get("data_dir", "./data"))
    output_name = output_cfg.get("output_file", "dclo_state_year.csv")

    gold_dir = data_dir / "gold"
    gold_dir.mkdir(parents=True, exist_ok=True)
    out_path = gold_dir / output_name
    scored.to_csv(out_path, index=False)
    print(f"Wrote {len(scored)} rows to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build DCLO from local downloaded files")
    parser.add_argument("--config", required=True, help="Path to local sources YAML config")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
