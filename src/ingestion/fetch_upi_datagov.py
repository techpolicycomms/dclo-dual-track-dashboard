"""
Fetch UPI transaction statistics from data.gov.in (ECO pillar).

Reuses the existing DataGovInClient. Pulls the most recent monthly UPI stats
and writes them in a normalised long format for the ECO loop.

Writes:
  data/raw/upi_datagov_<timestamp>.csv
  data/raw/upi_datagov_latest.csv
"""

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

import pandas as pd
import yaml

from data_gov_in_client import get_client_from_env


def read_config(config_path: str) -> Dict:
    with open(config_path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def fetch(cfg: Dict, raw_dir: Path) -> pd.DataFrame:
    loop = cfg["loops"]["eco"]
    resource_id = loop["resource_id"]
    now = datetime.now(timezone.utc)
    period_label = now.strftime("%Y-%W")

    client = get_client_from_env()
    records = client.fetch_records(resource_id=resource_id, offset=0, limit=200)

    if not records:
        raise ValueError(f"No records returned from data.gov.in resource {resource_id}")

    df = pd.DataFrame(records)

    # Standardise column names (the resource may vary slightly)
    rename_map = {}
    for col in df.columns:
        cl = col.lower().replace(" ", "_").replace("-", "_")
        if "volume" in cl or "no_of" in cl or "transactions" in cl:
            rename_map[col] = "volume_mn_transactions"
        elif "value" in cl or "amount" in cl:
            rename_map[col] = "value_cr_inr"
        elif "month" in cl or "date" in cl:
            rename_map[col] = "month_year"
    df = df.rename(columns=rename_map)

    df["period"] = period_label
    df["fetched_at_utc"] = now.isoformat()

    ts = now.strftime("%Y%m%dT%H%M%SZ")
    timestamped = raw_dir / f"upi_datagov_{ts}.csv"
    latest = raw_dir / "upi_datagov_latest.csv"
    df.to_csv(timestamped, index=False)
    df.to_csv(latest, index=False)
    print(f"Wrote {len(df)} UPI rows from data.gov.in → {timestamped}")
    return df


def run(config_path: str) -> pd.DataFrame:
    cfg = read_config(config_path)
    raw_dir = Path(cfg["output"]["data_dir"]) / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    return fetch(cfg, raw_dir)


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config/weekly_secondary_loops.yml"
    run(config_path)
