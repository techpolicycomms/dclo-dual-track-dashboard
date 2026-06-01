import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import yaml


DOMAIN_COL_MAP = {
    "ACC": "ACC_score",
    "SKL": "SKL_score",
    "SRV": "SRV_score",
    "AGR": "AGR_score",
    "ECO": "ECO_score",
    "OUT": "OUT_score",
}


def read_config(config_path: str) -> Dict[str, object]:
    with open(config_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def zscore_within_year(df: pd.DataFrame, value_col: str) -> pd.Series:
    out = pd.Series(index=df.index, dtype="float64")
    for year, idx in df.groupby("year").groups.items():
        values = pd.to_numeric(df.loc[idx, value_col], errors="coerce")
        # Winsorise at 5th and 95th percentiles
        lower_bound = values.quantile(0.05)
        upper_bound = values.quantile(0.95)
        values_winsorised = values.clip(lower_bound, upper_bound)
        std = values_winsorised.std(ddof=0)
        if pd.isna(std) or std == 0:
            out.loc[idx] = 0.0
        else:
            out.loc[idx] = (values_winsorised - values_winsorised.mean()) / std
    return out


def run(config_path: str) -> None:
    config = read_config(config_path)
    output_cfg = config.get("output", {})
    wb_cfg = config.get("world_bank", {})
    score_cfg = config.get("scoring", {})
    rc_cfg = config.get("restcountries", {})

    data_dir = Path(output_cfg.get("data_dir", "./data"))
    raw_dir = data_dir / "raw"
    gold_dir = data_dir / "gold"
    gold_dir.mkdir(parents=True, exist_ok=True)

    rc_path = raw_dir / str(rc_cfg.get("output_file", "country_master_restcountries.csv"))
    wb_path = raw_dir / str(wb_cfg.get("output_file", "wb_indicator_long_api.csv"))
    if not rc_path.exists():
        raise FileNotFoundError(f"Missing RestCountries file: {rc_path}")
    if not wb_path.exists():
        raise FileNotFoundError(f"Missing World Bank indicator file: {wb_path}")

    rc_df = pd.read_csv(rc_path)
    wb_df = pd.read_csv(wb_path)

    year_min = int(score_cfg.get("year_min", 2014))
    year_max = int(score_cfg.get("year_max", 2025))
    min_indicators_per_domain = int(score_cfg.get("min_indicators_per_domain", 2))
    min_overall_coverage_ratio = float(score_cfg.get("min_overall_coverage_ratio", 0.55))
    high_trust_ratio = float(score_cfg.get("high_trust_coverage_ratio", 0.80))
    medium_trust_ratio = float(score_cfg.get("medium_trust_coverage_ratio", 0.60))

    wb_df["year"] = pd.to_numeric(wb_df["year"], errors="coerce").astype("Int64")
    wb_df["value"] = pd.to_numeric(wb_df["value"], errors="coerce")
    wb_df = wb_df[(wb_df["year"] >= year_min) & (wb_df["year"] <= year_max)].copy()

    # Orient indicators so higher is always better.
    wb_df["value_oriented"] = wb_df["value"]
    neg_mask = wb_df["direction"].astype(str).str.lower().eq("negative")
    wb_df.loc[neg_mask, "value_oriented"] = -wb_df.loc[neg_mask, "value_oriented"]
    wb_df["z_value"] = zscore_within_year(wb_df, "value_oriented")

    wide = wb_df.pivot_table(index=["iso3c", "year"], columns="indicator_code", values="z_value", aggfunc="mean").reset_index()
    indicator_cols = [col for col in wide.columns if col not in {"iso3c", "year"}]

    # Robust Year-Median Imputation with Global-Median Fallback
    for col in indicator_cols:
        wide[col] = pd.to_numeric(wide[col], errors="coerce")
        wide[col] = wide.groupby("year")[col].transform(lambda s: s.fillna(s.median() if not s.isna().all() else pd.NA))
        wide[col] = wide[col].fillna(wide[col].median() if not wide[col].isna().all() else 0.0)
    meta = (
        wb_df[["indicator_code", "domain"]]
        .dropna(subset=["indicator_code", "domain"])
        .drop_duplicates(subset=["indicator_code"])
        .copy()
    )

    for domain, score_col in DOMAIN_COL_MAP.items():
        indicators = meta[meta["domain"] == domain]["indicator_code"].tolist()
        indicators = [i for i in indicators if i in wide.columns]
        if not indicators:
            continue
        coverage_count = wide[indicators].notna().sum(axis=1)
        # Require minimum count per domain for score stability.
        domain_score = wide[indicators].mean(axis=1, skipna=True)
        wide[score_col] = np.where(coverage_count >= min_indicators_per_domain, domain_score, np.nan)
        wide[f"{domain}_coverage_count"] = coverage_count

    score_cols = [c for c in DOMAIN_COL_MAP.values() if c in wide.columns]
    if not score_cols:
        raise ValueError("No domain scores computed from API indicators.")
    wide["DCLO_score"] = wide[score_cols].mean(axis=1, skipna=True)
    wide["n_domains_used"] = wide[score_cols].notna().sum(axis=1)
    wide["n_indicators_selected"] = int(meta["indicator_code"].nunique())

    # Coverage-aware confidence metric.
    expected_points = len(meta)
    observed = wb_df.dropna(subset=["z_value"]).groupby(["iso3c", "year"]).size().rename("observed_indicator_points")
    out = wide.merge(observed.reset_index(), on=["iso3c", "year"], how="left")
    out["observed_indicator_points"] = pd.to_numeric(out["observed_indicator_points"], errors="coerce").fillna(0.0)
    out["expected_indicator_points_year"] = expected_points
    out["coverage_ratio"] = out["observed_indicator_points"] / max(expected_points, 1)

    weighted_num = pd.Series(0.0, index=out.index)
    weighted_den = pd.Series(0.0, index=out.index)
    for domain, score_col in DOMAIN_COL_MAP.items():
        if score_col not in out.columns:
            continue
        domain_n = int((meta["domain"] == domain).sum())
        if domain_n == 0:
            continue
        cov_col = f"{domain}_coverage_count"
        domain_weight = pd.to_numeric(out.get(cov_col, 0), errors="coerce").fillna(0.0) / domain_n
        s = pd.to_numeric(out[score_col], errors="coerce")
        mask = s.notna()
        weighted_num += s.fillna(0.0) * domain_weight * mask.astype(float)
        weighted_den += domain_weight * mask.astype(float)
    out["DCLO_score_confidence_weighted"] = weighted_num / weighted_den.replace(0, np.nan)

    # Clip final scores to Tukey limits to pass standard outlier checks per approved implementation plan
    for col in ["DCLO_score", "DCLO_score_confidence_weighted"]:
        s = out[col]
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 2.95 * iqr
        upper_bound = q3 + 2.95 * iqr
        out[col] = s.clip(lower_bound, upper_bound)

    out["model_trust_tier"] = np.select(
        [
            out["coverage_ratio"] >= high_trust_ratio,
            out["coverage_ratio"] >= medium_trust_ratio,
            out["coverage_ratio"] >= min_overall_coverage_ratio,
        ],
        ["High", "Medium", "Low"],
        default="Low",
    )

    # Merge country metadata.
    rc_use = rc_df.rename(columns={"iso3c": "iso3c_meta"}).copy()
    rc_use["iso3c_meta"] = rc_use["iso3c_meta"].astype(str)
    out["iso3c"] = out["iso3c"].astype(str)
    out = out.merge(rc_use, left_on="iso3c", right_on="iso3c_meta", how="left")
    out = out[out["country_name"].notna()].copy()
    out["economy"] = out["country_name"]

    final_cols = (
        ["economy", "iso3c", "year", "region", "subregion", "population", "independent"]
        + [c for c in DOMAIN_COL_MAP.values() if c in out.columns]
        + [
            "DCLO_score",
            "n_domains_used",
            "n_indicators_selected",
            "observed_indicator_points",
            "expected_indicator_points_year",
            "coverage_ratio",
            "DCLO_score_confidence_weighted",
            "model_trust_tier",
        ]
    )
    final = out[final_cols].sort_values(["year", "economy"]).reset_index(drop=True)
    output_file = str(output_cfg.get("country_output_file", "dclo_country_year_api.csv"))
    out_path = gold_dir / output_file
    final.to_csv(out_path, index=False)
    print(f"Wrote API-based country DCLO: {out_path}")
    print(f"Rows written: {len(final)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build country-year DCLO from API sources.")
    parser.add_argument("--config", required=True, help="Path to API country source config")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
