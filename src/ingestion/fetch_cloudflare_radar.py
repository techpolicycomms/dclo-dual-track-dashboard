"""
Fetch Cloudflare Radar internet speed / quality data (ACC pillar).

Writes:
  data/raw/cloudflare_radar_<timestamp>.csv
  data/raw/cloudflare_radar_latest.csv
"""

import os
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict

import pandas as pd
import requests
import yaml


def read_config(config_path: str) -> Dict:
    with open(config_path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def fetch(cfg: Dict, raw_dir: Path) -> pd.DataFrame:
    loop = cfg["loops"]["acc"]
    token = os.getenv(loop.get("auth_env_var", ""), "").strip()
    if not token:
        raise ValueError(
            f"Environment variable {loop['auth_env_var']} is missing or empty. "
            "Register a free token at dash.cloudflare.com → Radar API."
        )

    headers = {"Authorization": f"Bearer {token}"}
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=7)
    params = {
        **loop.get("params", {}),
        "dateStart": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "dateEnd": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    resp = requests.get(loop["endpoint"], headers=headers, params=params, timeout=30)
    resp.raise_for_status()
    payload = resp.json()

    result = payload.get("result", {})
    top = result.get("top_0", [])
    if not top:
        raise ValueError(f"Unexpected Cloudflare Radar response shape: {list(result.keys())}")

    rows = []
    period_label = end.strftime("%Y-%W")
    for entry in top:
        rows.append(
            {
                "period": period_label,
                "location": entry.get("clientCountryAlpha2", ""),
                "bandwidth_download": pd.to_numeric(entry.get("bandwidthDownload"), errors="coerce"),
                "bandwidth_upload": pd.to_numeric(entry.get("bandwidthUpload"), errors="coerce"),
                "latency_idle_ms": pd.to_numeric(entry.get("latencyIdle"), errors="coerce"),
                "jitter_ms": pd.to_numeric(entry.get("jitter"), errors="coerce"),
                "fetched_at_utc": end.isoformat(),
            }
        )

    df = pd.DataFrame(rows)
    ts = end.strftime("%Y%m%dT%H%M%SZ")
    timestamped = raw_dir / f"cloudflare_radar_{ts}.csv"
    latest = raw_dir / "cloudflare_radar_latest.csv"
    df.to_csv(timestamped, index=False)
    df.to_csv(latest, index=False)
    print(f"Wrote {len(df)} Cloudflare Radar rows → {timestamped}")
    return df


def run(config_path: str) -> pd.DataFrame:
    cfg = read_config(config_path)
    raw_dir = Path(cfg["output"]["data_dir"]) / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    return fetch(cfg, raw_dir)


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config/weekly_secondary_loops.yml"
    run(config_path)
