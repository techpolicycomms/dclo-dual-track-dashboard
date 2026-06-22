"""
Shared normaliser for all six DCLO weekly secondary loops.

Reads the latest raw CSV for a given loop, applies within-period z-scoring
(replicating the methodology in build_dclo_country_api.py), direction-flips
negative indicators, and writes a gold CSV with (entity, period, value,
z_value, <PILLAR>_score, coverage_ratio, model_trust_tier).
"""

import argparse
from pathlib import Path
from typing import Dict

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

RAW_FILE_MAP = {
    "acc": "cloudflare_radar_latest.csv",
    "skl": "wikimedia_pageviews_latest.csv",
    "srv": "dpg_registry_latest.csv",
    "agr": "gdelt_discourse_latest.csv",
    "eco": "upi_datagov_latest.csv",
    "out": "openaq_latest.csv",
}


def read_config(config_path: str) -> Dict:
    with open(config_path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def zscore_across(series: pd.Series) -> pd.Series:
    """Population z-score with 5th–95th percentile winsorisation (OECD 2008)."""
    s = pd.to_numeric(series, errors="coerce")
    lo, hi = s.quantile(0.05), s.quantile(0.95)
    clipped = s.clip(lo, hi)
    std = clipped.std(ddof=0)
    if pd.isna(std) or std == 0:
        return pd.Series(0.0, index=series.index)
    return (clipped - clipped.mean()) / std


def build(loop_name: str, cfg: Dict, raw_dir: Path, gold_dir: Path) -> pd.DataFrame:
    loop = cfg["loops"][loop_name]
    domain = loop["domain"]
    direction = loop.get("direction", "positive")
    entity_col = loop["entity_col"]
    value_col = loop["value_col"]
    output_file = loop["output_file"]
    score_col = DOMAIN_COL_MAP[domain]

    raw_file = raw_dir / RAW_FILE_MAP[loop_name]
    if not raw_file.exists():
        raise FileNotFoundError(f"Raw file not found: {raw_file}. Run the ingestion step first.")

    df = pd.read_csv(raw_file)
    if entity_col not in df.columns:
        raise ValueError(f"entity_col '{entity_col}' not in {list(df.columns)}")
    if value_col not in df.columns:
        raise ValueError(f"value_col '{value_col}' not in {list(df.columns)}")

    df["value"] = pd.to_numeric(df[value_col], errors="coerce")
    df = df.dropna(subset=["value"]).copy()

    # Direction harmonisation: negative indicators are flipped so higher z → better
    value_oriented = -df["value"] if direction == "negative" else df["value"]
    df["z_value"] = zscore_across(value_oriented)

    # Coverage: fraction of entities with non-null observations
    n_total = len(df)
    n_observed = int(df["value"].notna().sum())
    coverage_ratio = round(n_observed / max(n_total, 1), 4)

    df[score_col] = df["z_value"]
    df["coverage_ratio"] = coverage_ratio
    df["model_trust_tier"] = np.where(
        coverage_ratio >= 0.80, "High",
        np.where(coverage_ratio >= 0.60, "Medium", "Low")
    )
    df["entity"] = df[entity_col].astype(str)
    df["domain"] = domain

    # Pull period col if present
    if "period" not in df.columns:
        from datetime import datetime, timezone
        df["period"] = datetime.now(timezone.utc).strftime("%Y-%W")

    keep_cols = (
        ["entity", "period", "domain", "value", "z_value", score_col,
         "coverage_ratio", "model_trust_tier"]
        + [c for c in ["avg_tone", "active_editors", "n_readings", "wiki_label",
                        "entity_type", "fetched_at_utc"]
           if c in df.columns]
    )
    final = df[[c for c in keep_cols if c in df.columns]].sort_values(
        ["period", "entity"]
    ).reset_index(drop=True)

    out_path = gold_dir / output_file
    final.to_csv(out_path, index=False)
    print(f"Wrote {len(final)} rows → {out_path}")
    return final


def run(loop_name: str, config_path: str) -> pd.DataFrame:
    cfg = read_config(config_path)
    data_dir = Path(cfg["output"]["data_dir"])
    raw_dir = data_dir / "raw"
    gold_dir = data_dir / "gold"
    gold_dir.mkdir(parents=True, exist_ok=True)
    return build(loop_name, cfg, raw_dir, gold_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalise a DCLO weekly secondary loop")
    parser.add_argument("--loop", required=True, choices=list(RAW_FILE_MAP.keys()),
                        help="Which secondary loop to build")
    parser.add_argument("--config", default="config/weekly_secondary_loops.yml")
    args = parser.parse_args()
    run(args.loop, args.config)


if __name__ == "__main__":
    main()
