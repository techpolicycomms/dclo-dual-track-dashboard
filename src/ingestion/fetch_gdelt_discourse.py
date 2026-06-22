"""
Fetch GDELT 2.0 DOC API — DPI/digital-rights news volume and tone (AGR pillar).

Queries the GDELT Document API for each configured query term over the past 7 days,
capturing article count and average tone (positive = pro-digital rights).
No auth required.

Writes:
  data/raw/gdelt_discourse_<timestamp>.csv
  data/raw/gdelt_discourse_latest.csv
"""

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


def _gdelt_date(dt: datetime) -> str:
    return dt.strftime("%Y%m%d%H%M%S")


def fetch_query(endpoint: str, query_term: str, start: datetime, end: datetime) -> Dict:
    params = {
        "query": query_term,
        "mode": "artlist",
        "maxrecords": 250,
        "startdatetime": _gdelt_date(start),
        "enddatetime": _gdelt_date(end),
        "format": "json",
        "TIMESPAN": "1week",
        "sort": "DateDesc",
    }
    resp = requests.get(endpoint, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    articles = data.get("articles", []) or []
    tones = [float(a["tone"]) for a in articles if "tone" in a and a["tone"] is not None]
    return {
        "article_count": len(articles),
        "avg_tone": round(sum(tones) / len(tones), 4) if tones else None,
    }


def fetch(cfg: Dict, raw_dir: Path) -> pd.DataFrame:
    loop = cfg["loops"]["agr"]
    now = datetime.now(timezone.utc)
    start = now - timedelta(days=7)
    period_label = now.strftime("%Y-%W")

    rows = []
    for q in loop.get("queries", []):
        label = q["label"]
        term = q["term"]
        print(f"  GDELT query: {label!r}")
        try:
            result = fetch_query(loop["endpoint"], term, start, now)
        except Exception as exc:
            print(f"    Warning: GDELT query failed ({exc}); recording null")
            result = {"article_count": 0, "avg_tone": None}
        rows.append(
            {
                "period": period_label,
                "query_label": label,
                "query_term": term,
                "article_count": result["article_count"],
                "avg_tone": result["avg_tone"],
                "fetched_at_utc": now.isoformat(),
            }
        )
        print(f"    → {result['article_count']} articles, tone={result['avg_tone']}")

    df = pd.DataFrame(rows)
    ts = now.strftime("%Y%m%dT%H%M%SZ")
    timestamped = raw_dir / f"gdelt_discourse_{ts}.csv"
    latest = raw_dir / "gdelt_discourse_latest.csv"
    df.to_csv(timestamped, index=False)
    df.to_csv(latest, index=False)
    print(f"Wrote {len(df)} GDELT rows → {timestamped}")
    return df


def run(config_path: str) -> pd.DataFrame:
    cfg = read_config(config_path)
    raw_dir = Path(cfg["output"]["data_dir"]) / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    return fetch(cfg, raw_dir)


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config/weekly_secondary_loops.yml"
    run(config_path)
