"""
Fetch Digital Public Goods / GovTech ecosystem data (SRV pillar).

Strategy (two sources, in priority order):
  1. DPGA REST API  https://api.digitalpublicgoods.net/dpgs  (no auth, fails gracefully)
  2. GitHub Search API for repositories tagged with govtech / DPI topics
     — requires GITHUB_TOKEN (automatically provided in every GitHub Actions run)

Aggregates by SDG category and deployment sector, then writes counts.

Writes:
  data/raw/dpg_registry_<timestamp>.csv
  data/raw/dpg_registry_latest.csv
"""

import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import requests
import yaml


DPGA_ENDPOINT = "https://api.digitalpublicgoods.net/dpgs"
GITHUB_SEARCH_ENDPOINT = "https://api.github.com/search/repositories"
GOVTECH_TOPICS = [
    "digital-public-infrastructure",
    "digital-public-goods",
    "govtech",
    "e-government",
    "open-government",
    "digital-government",
]


def read_config(config_path: str) -> Dict:
    with open(config_path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _try_dpga_api(timeout: int = 20) -> Optional[List[Dict]]:
    try:
        resp = requests.get(DPGA_ENDPOINT, timeout=timeout,
                            headers={"User-Agent": "dclo-research/1.0"})
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, list) and len(data) > 0:
            print(f"  DPGA API: {len(data)} DPGs fetched")
            return data
    except Exception as exc:
        print(f"  DPGA API unavailable ({exc}), falling back to GitHub Search")
    return None


def _aggregate_dpgs(dpgs: List[Dict], period_label: str, now: datetime) -> List[Dict]:
    sdg_counts: Dict[str, int] = defaultdict(int)
    country_counts: Dict[str, int] = defaultdict(int)
    sector_counts: Dict[str, int] = defaultdict(int)

    for dpg in dpgs:
        for sdg_entry in dpg.get("sdgs", []) or []:
            goal = sdg_entry.get("SDGNumber") or sdg_entry.get("goal", "")
            if goal:
                sdg_counts[f"SDG_{goal}"] += 1
        for c in dpg.get("deploymentCountries", []) or []:
            country_counts[c.get("name", c) if isinstance(c, dict) else c] += 1
        for cat in dpg.get("categories", []) or []:
            sector_counts[cat.get("category", cat) if isinstance(cat, dict) else cat] += 1

    rows = []
    for sdg, count in sorted(sdg_counts.items()):
        rows.append({"period": period_label, "sdg": sdg, "dpg_count": count,
                     "source": "dpga_api", "entity_type": "sdg",
                     "fetched_at_utc": now.isoformat()})
    for country, count in sorted(country_counts.items(), key=lambda x: -x[1])[:30]:
        rows.append({"period": period_label, "sdg": country, "dpg_count": count,
                     "source": "dpga_api", "entity_type": "country",
                     "fetched_at_utc": now.isoformat()})
    for sector, count in sorted(sector_counts.items(), key=lambda x: -x[1]):
        rows.append({"period": period_label, "sdg": sector, "dpg_count": count,
                     "source": "dpga_api", "entity_type": "sector",
                     "fetched_at_utc": now.isoformat()})
    rows.append({"period": period_label, "sdg": "TOTAL", "dpg_count": len(dpgs),
                 "source": "dpga_api", "entity_type": "aggregate",
                 "fetched_at_utc": now.isoformat()})
    return rows


def _fetch_github_govtech(period_label: str, now: datetime) -> List[Dict]:
    token = os.getenv("GITHUB_TOKEN", "").strip()
    headers = {"User-Agent": "dclo-research/1.0", "Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    rows = []
    for topic in GOVTECH_TOPICS:
        params = {
            "q": f"topic:{topic}",
            "sort": "stars",
            "order": "desc",
            "per_page": 1,
        }
        try:
            resp = requests.get(GITHUB_SEARCH_ENDPOINT, headers=headers,
                                params=params, timeout=20)
            resp.raise_for_status()
            count = resp.json().get("total_count", 0)
        except Exception as exc:
            print(f"  GitHub topic {topic}: {exc}")
            count = 0
        rows.append({
            "period": period_label,
            "sdg": topic,
            "dpg_count": count,
            "source": "github_search",
            "entity_type": "topic",
            "fetched_at_utc": now.isoformat(),
        })
        print(f"  GitHub topic '{topic}': {count} repos")

    total = sum(r["dpg_count"] for r in rows)
    rows.append({"period": period_label, "sdg": "TOTAL",
                 "dpg_count": total, "source": "github_search",
                 "entity_type": "aggregate", "fetched_at_utc": now.isoformat()})
    return rows


def fetch(cfg: Dict, raw_dir: Path) -> pd.DataFrame:
    now = datetime.now(timezone.utc)
    period_label = now.strftime("%Y-%W")

    dpgs = _try_dpga_api()
    if dpgs is not None:
        rows = _aggregate_dpgs(dpgs, period_label, now)
    else:
        print("  Using GitHub Search fallback for SRV loop")
        rows = _fetch_github_govtech(period_label, now)

    df = pd.DataFrame(rows)
    ts = now.strftime("%Y%m%dT%H%M%SZ")
    timestamped = raw_dir / f"dpg_registry_{ts}.csv"
    latest = raw_dir / "dpg_registry_latest.csv"
    df.to_csv(timestamped, index=False)
    df.to_csv(latest, index=False)
    print(f"Wrote {len(df)} rows → {timestamped}")
    return df


def run(config_path: str) -> pd.DataFrame:
    cfg = read_config(config_path)
    raw_dir = Path(cfg["output"]["data_dir"]) / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    return fetch(cfg, raw_dir)


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config/weekly_secondary_loops.yml"
    run(config_path)
