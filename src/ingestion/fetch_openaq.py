"""
Fetch OpenAQ v3 PM2.5 measurements near Bihar study districts (OUT pillar).

Finds stations within radius_km of each study location, retrieves the latest
readings, and computes a weekly mean.

Writes:
  data/raw/openaq_<timestamp>.csv
  data/raw/openaq_latest.csv
"""

import math
import os
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import requests
import yaml


def read_config(config_path: str) -> Dict:
    with open(config_path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _get_headers(env_var: Optional[str]) -> Dict[str, str]:
    token = os.getenv(env_var or "", "").strip() if env_var else ""
    if not token:
        raise ValueError(
            f"Environment variable {env_var} is missing or empty. "
            "Register a free key at explore.openaq.org."
        )
    return {"X-API-Key": token}


def fetch_locations_near(lat: float, lon: float, radius_km: int, headers: Dict) -> List[Dict]:
    params = {
        "coordinates": f"{lat},{lon}",
        "radius": radius_km * 1000,
        "parameter_id": 2,  # PM2.5 parameter ID in OpenAQ v3
        "limit": 50,
    }
    resp = requests.get("https://api.openaq.org/v3/locations", headers=headers,
                        params=params, timeout=30)
    resp.raise_for_status()
    return resp.json().get("results", [])


def fetch_measurements(location_id: int, headers: Dict, days_back: int = 7) -> List[float]:
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days_back)
    params = {
        "location_id": location_id,
        "parameter_id": 2,
        "date_from": start.isoformat(),
        "date_to": end.isoformat(),
        "limit": 500,
    }
    resp = requests.get("https://api.openaq.org/v3/measurements", headers=headers,
                        params=params, timeout=30)
    resp.raise_for_status()
    results = resp.json().get("results", [])
    return [r["value"] for r in results if r.get("value") is not None]


def fetch(cfg: Dict, raw_dir: Path) -> pd.DataFrame:
    loop = cfg["loops"]["out"]
    headers = _get_headers(loop.get("auth_env_var"))
    now = datetime.now(timezone.utc)
    period_label = now.strftime("%Y-%W")
    radius_km = int(loop.get("radius_km", 50))

    rows = []
    for loc in loop["study_locations"]:
        label = loc["label"]
        lat, lon = float(loc["lat"]), float(loc["lon"])
        print(f"  Fetching OpenAQ stations near {label} ({lat}, {lon})")
        try:
            stations = fetch_locations_near(lat, lon, radius_km, headers)
            all_values: List[float] = []
            for station in stations:
                sid = station.get("id")
                if sid:
                    vals = fetch_measurements(sid, headers)
                    all_values.extend(vals)
            pm25_avg = round(sum(all_values) / len(all_values), 2) if all_values else None
            n_readings = len(all_values)
        except Exception as exc:
            print(f"    Warning: {exc}")
            pm25_avg = None
            n_readings = 0

        rows.append(
            {
                "period": period_label,
                "location_label": label,
                "lat": lat,
                "lon": lon,
                "pm25_avg": pm25_avg,
                "n_readings": n_readings,
                "fetched_at_utc": now.isoformat(),
            }
        )
        print(f"    → pm25_avg={pm25_avg}, n_readings={n_readings}")

    df = pd.DataFrame(rows)
    ts = now.strftime("%Y%m%dT%H%M%SZ")
    timestamped = raw_dir / f"openaq_{ts}.csv"
    latest = raw_dir / "openaq_latest.csv"
    df.to_csv(timestamped, index=False)
    df.to_csv(latest, index=False)
    print(f"Wrote {len(df)} OpenAQ rows → {timestamped}")
    return df


def run(config_path: str) -> pd.DataFrame:
    cfg = read_config(config_path)
    raw_dir = Path(cfg["output"]["data_dir"]) / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    return fetch(cfg, raw_dir)


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config/weekly_secondary_loops.yml"
    run(config_path)
