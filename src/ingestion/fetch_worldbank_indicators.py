import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd
import requests
import yaml


def read_config(config_path: str) -> Dict[str, object]:
    with open(config_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def fetch_indicator(base_url: str, indicator_code: str, fmt: str, per_page: int, timeout_seconds: int) -> List[Dict[str, object]]:
    url = f"{base_url}/{indicator_code}?format={fmt}&per_page={per_page}"
    response = requests.get(url, timeout=timeout_seconds)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, list):
        raise ValueError(f"Unexpected World Bank response for {indicator_code}")
    if len(payload) < 2 or not isinstance(payload[1], list):
        message = payload[0] if payload else {}
        print(f"[worldbank] skipping archived or missing indicator {indicator_code}: {message}")
        return []
    return payload[1]


def run(config_path: str) -> None:
    config = read_config(config_path)
    wb_cfg = config.get("world_bank", {})
    output_cfg = config.get("output", {})

    data_dir = Path(output_cfg.get("data_dir", "./data"))
    raw_dir = data_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    base_url = str(wb_cfg.get("base_url"))
    fmt = str(wb_cfg.get("format", "json"))
    per_page = int(wb_cfg.get("per_page", 20000))
    timeout_seconds = int(wb_cfg.get("timeout_seconds", 30))
    indicators = wb_cfg.get("indicators", [])
    out_path = raw_dir / str(wb_cfg.get("output_file", "wb_indicator_long_api.csv"))

    rows: List[Dict[str, object]] = []
    for indicator in indicators:
        code = str(indicator.get("code"))
        domain = str(indicator.get("domain", ""))
        direction = str(indicator.get("direction", "positive"))
        name = str(indicator.get("name", code))
        records = fetch_indicator(base_url, code, fmt, per_page, timeout_seconds)
        for record in records:
            country = record.get("country", {})
            rows.append(
                {
                    "indicator_code": code,
                    "indicator_name": name,
                    "domain": domain,
                    "direction": direction,
                    "iso3c": record.get("countryiso3code"),
                    "country_name_wb": country.get("value") if isinstance(country, dict) else None,
                    "year": pd.to_numeric(record.get("date"), errors="coerce"),
                    "value": pd.to_numeric(record.get("value"), errors="coerce"),
                }
            )

    out_df = pd.DataFrame(rows)
    out_df = out_df.dropna(subset=["iso3c", "year"]).sort_values(["indicator_code", "iso3c", "year"])
    out_df.to_csv(out_path, index=False)
    print(f"Wrote World Bank indicator long file: {out_path}")
    print(f"Rows written: {len(out_df)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch World Bank indicators for DCLO API pipeline.")
    parser.add_argument("--config", required=True, help="Path to API country source config")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
