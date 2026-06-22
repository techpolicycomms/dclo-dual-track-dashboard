"""
Fetch Wikimedia pageview counts for low-resource-language wikis (SKL pillar).

Calls the Wikimedia REST API per-project for the previous ISO week.
No auth required.

Writes:
  data/raw/wikimedia_pageviews_<timestamp>.csv
  data/raw/wikimedia_pageviews_latest.csv
"""

import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List

import pandas as pd
import requests
import yaml


WIKIMEDIA_BASE = "https://wikimedia.org/api/rest_v1"
UA = "dclo-research/1.0 (research; rahul.jha@policyresearch.example)"


def read_config(config_path: str) -> Dict:
    with open(config_path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def fetch_project_pageviews(project: str, start: datetime, end: datetime) -> int:
    """Return total article pageviews for a project over [start, end) (daily sum)."""
    total = 0
    headers = {"User-Agent": UA}
    day = start
    while day < end:
        date_str = day.strftime("%Y/%m/%d")
        url = f"{WIKIMEDIA_BASE}/metrics/pageviews/aggregate/{project}/all-access/all-agents/daily/{date_str}/{date_str}"
        try:
            resp = requests.get(url, headers=headers, timeout=20)
            resp.raise_for_status()
            items = resp.json().get("items", [])
            total += sum(int(item.get("views", 0)) for item in items)
        except Exception:
            pass
        day += timedelta(days=1)
    return total


def fetch_active_editors(project: str, year: int, month: int) -> int:
    """Return count of active editors for a wiki project in a given year-month."""
    headers = {"User-Agent": UA}
    url = (
        f"{WIKIMEDIA_BASE}/metrics/editors/aggregate/{project}"
        f"/all-editor-types/content/{year}/{month:02d}01/{year}/{month:02d}28"
    )
    try:
        resp = requests.get(url, headers=headers, timeout=20)
        resp.raise_for_status()
        items = resp.json().get("items", [])
        return sum(int(item.get("editors", 0)) for item in items)
    except Exception:
        return 0


def fetch(cfg: Dict, raw_dir: Path) -> pd.DataFrame:
    loop = cfg["loops"]["skl"]
    wikis: List[Dict] = loop.get("wikis", [])

    now = datetime.now(timezone.utc)
    end = now - timedelta(days=1)
    start = end - timedelta(days=6)
    period_label = now.strftime("%Y-%W")

    rows = []
    for wiki in wikis:
        code = wiki["code"]
        label = wiki["label"]
        project = f"{code}.wikipedia.org"
        total_views = fetch_project_pageviews(project, start, end)
        active_eds = fetch_active_editors(project, end.year, end.month)
        rows.append(
            {
                "period": period_label,
                "wiki_code": code,
                "wiki_label": label,
                "total_pageviews": total_views,
                "active_editors": active_eds,
                "fetched_at_utc": now.isoformat(),
            }
        )
        print(f"  {code}: {total_views:,} pageviews, {active_eds} active editors")

    df = pd.DataFrame(rows)
    ts = now.strftime("%Y%m%dT%H%M%SZ")
    timestamped = raw_dir / f"wikimedia_pageviews_{ts}.csv"
    latest = raw_dir / "wikimedia_pageviews_latest.csv"
    df.to_csv(timestamped, index=False)
    df.to_csv(latest, index=False)
    print(f"Wrote {len(df)} Wikimedia rows → {timestamped}")
    return df


def run(config_path: str) -> pd.DataFrame:
    cfg = read_config(config_path)
    raw_dir = Path(cfg["output"]["data_dir"]) / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    return fetch(cfg, raw_dir)


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config/weekly_secondary_loops.yml"
    run(config_path)
