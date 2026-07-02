import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional

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


def write_country_master(rows: List[Dict[str, object]], out_path: Path, source: str) -> None:
    out_df = pd.DataFrame(rows)
    out_df = out_df.dropna(subset=["iso3c"]).drop_duplicates(subset=["iso3c"]).sort_values("country_name")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"Wrote country master from {source}: {out_path}")
    print(f"Countries written: {len(out_df)}")


def fetch_restcountries_v3(url: str, timeout_seconds: int) -> List[Dict[str, object]]:
    response = requests.get(url, timeout=timeout_seconds)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, list):
        raise ValueError("Unexpected RestCountries v3 response; expected a list.")

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
    return rows


def fetch_restcountries_v5(api_key: str, timeout_seconds: int) -> List[Dict[str, object]]:
    base_url = "https://api.restcountries.com/countries/v5"
    headers = {"Authorization": f"Bearer {api_key}"}
    params = {
        "response_fields": "names.common,names.official,codes.alpha_3,region,subregion,population,independent,capitals",
        "limit": 100,
        "offset": 0,
    }

    rows: List[Dict[str, object]] = []
    while True:
        response = requests.get(base_url, headers=headers, params=params, timeout=timeout_seconds)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, list):
            raise ValueError("Unexpected RestCountries v5 response; expected a list.")

        if not payload:
            break

        for row in payload:
            if not isinstance(row, dict):
                continue
            names = row.get("names", {})
            codes = row.get("codes", {})
            capitals = row.get("capitals", [])
            capital = ""
            lat = None
            lon = None
            if isinstance(capitals, list) and capitals:
                first_capital = capitals[0]
                if isinstance(first_capital, dict):
                    capital = str(first_capital.get("name", ""))
                    coordinates = first_capital.get("coordinates", {})
                    if isinstance(coordinates, dict):
                        lat = coordinates.get("lat")
                        lon = coordinates.get("lng")
            rows.append(
                {
                    "iso3c": codes.get("alpha_3") if isinstance(codes, dict) else None,
                    "country_name": names.get("common") if isinstance(names, dict) else "",
                    "country_official_name": names.get("official") if isinstance(names, dict) else "",
                    "region": row.get("region", ""),
                    "subregion": row.get("subregion", ""),
                    "population": row.get("population"),
                    "independent": row.get("independent"),
                    "capital": capital,
                    "lat": lat,
                    "lon": lon,
                }
            )

        if len(payload) < int(params["limit"]):
            break
        params["offset"] = int(params["offset"]) + int(params["limit"])

    return rows


def fetch_world_bank_countries(timeout_seconds: int) -> List[Dict[str, object]]:
    url = "https://api.worldbank.org/v2/country?format=json&per_page=400"
    response = requests.get(url, timeout=timeout_seconds)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, list) or len(payload) < 2 or not isinstance(payload[1], list):
        raise ValueError("Unexpected World Bank country response.")

    rows: List[Dict[str, object]] = []
    for row in payload[1]:
        if not isinstance(row, dict):
            continue
        iso3c = row.get("id")
        if not iso3c or str(iso3c).startswith("X"):
            continue
        name = row.get("name", {})
        country_name = name.get("value") if isinstance(name, dict) else str(name)
        region = row.get("region", {})
        adminregion = row.get("adminregion", {})
        rows.append(
            {
                "iso3c": iso3c,
                "country_name": country_name,
                "country_official_name": country_name,
                "region": region.get("value", "") if isinstance(region, dict) else "",
                "subregion": adminregion.get("value", "") if isinstance(adminregion, dict) else "",
                "population": None,
                "independent": None,
                "capital": row.get("capitalCity", ""),
                "lat": row.get("latitude"),
                "lon": row.get("longitude"),
            }
        )
    return rows


def run(config_path: str) -> None:
    config = read_config(config_path)
    rc_cfg = config.get("restcountries", {})
    output_cfg = config.get("output", {})
    data_dir = Path(output_cfg.get("data_dir", "./data"))
    raw_dir = data_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    url = str(rc_cfg.get("url", ""))
    timeout_seconds = int(rc_cfg.get("timeout_seconds", 30))
    output_file = str(rc_cfg.get("output_file", "country_master_restcountries.csv"))
    out_path = raw_dir / output_file
    api_key = os.getenv("RESTCOUNTRIES_API_KEY", "").strip()

    errors: List[str] = []
    rows: Optional[List[Dict[str, object]]] = None

    if api_key:
        try:
            rows = fetch_restcountries_v5(api_key=api_key, timeout_seconds=timeout_seconds)
            write_country_master(rows, out_path, source="RestCountries v5")
            return
        except Exception as exc:  # noqa: BLE001 - collect and try fallbacks
            errors.append(f"restcountries_v5:{exc}")

    if url:
        try:
            rows = fetch_restcountries_v3(url=url, timeout_seconds=timeout_seconds)
            write_country_master(rows, out_path, source="RestCountries v3")
            return
        except Exception as exc:  # noqa: BLE001 - collect and try fallbacks
            errors.append(f"restcountries_v3:{exc}")

    try:
        rows = fetch_world_bank_countries(timeout_seconds=timeout_seconds)
        write_country_master(rows, out_path, source="World Bank country catalog")
        return
    except Exception as exc:  # noqa: BLE001 - collect and try fallbacks
        errors.append(f"world_bank_countries:{exc}")

    if out_path.exists():
        print("[fetch_restcountries] all live sources failed; keeping existing cached file:")
        for item in errors:
            print(f" - {item}")
        print(f"[fetch_restcountries] using cached: {out_path}")
        return

    joined = "; ".join(errors) if errors else "unknown"
    raise RuntimeError(f"Unable to fetch country metadata and no cached file exists: {joined}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch country metadata for DCLO API pipeline.")
    parser.add_argument("--config", required=True, help="Path to API country source config")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
