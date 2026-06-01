import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd
import requests
import yaml


def read_config(config_path: str) -> Dict[str, object]:
    with open(config_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def safe_get(d: Dict[str, object], keys: List[str], default: str = "") -> str:
    cur: object = d
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return str(cur) if cur is not None else default


def run(config_path: str) -> None:
    config = read_config(config_path)
    rc_cfg = config.get("restcountries", {})
    output_cfg = config.get("output", {})
    data_dir = Path(output_cfg.get("data_dir", "./data"))
    raw_dir = data_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    url = str(rc_cfg.get("url"))
    timeout_seconds = int(rc_cfg.get("timeout_seconds", 30))
    output_file = str(rc_cfg.get("output_file", "country_master_restcountries.csv"))
    out_path = raw_dir / output_file

    response = requests.get(url, timeout=timeout_seconds)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, list):
        raise ValueError("Unexpected RestCountries response; expected a list.")

    rows: List[Dict[str, object]] = []
    for row in payload:
        if not isinstance(row, dict):
            continue
        latlng = row.get("latlng", [])
        lat = latlng[0] if isinstance(latlng, list) and len(latlng) >= 1 else None
        lon = latlng[1] if isinstance(latlng, list) and len(latlng) >= 2 else None
        capital = ""
        if isinstance(row.get("capital"), list) and row["capital"]:
            capital = str(row["capital"][0])
        rows.append(
            {
                "iso3c": row.get("cca3"),
                "country_name": safe_get(row, ["name", "common"]),
                "country_official_name": safe_get(row, ["name", "official"]),
                "region": row.get("region", ""),
                "subregion": row.get("subregion", ""),
                "population": row.get("population"),
                "independent": row.get("independent"),
                "capital": capital,
                "lat": lat,
                "lon": lon,
            }
        )

    out_df = pd.DataFrame(rows)
    out_df = out_df.dropna(subset=["iso3c"]).drop_duplicates(subset=["iso3c"]).sort_values("country_name")
    out_df.to_csv(out_path, index=False)
    print(f"Wrote RestCountries master: {out_path}")
    print(f"Countries written: {len(out_df)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch RestCountries metadata for DCLO API pipeline.")
    parser.add_argument("--config", required=True, help="Path to API country source config")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
