#!/usr/bin/env python3
"""
dclo_news_agent.py
------------------
Fetches top digital/tech news from global and regional-language publishers,
maps headlines to DCLO domains, and writes gold intake artifacts.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import yaml

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "src"))

from ingestion.fetch_digital_news import run as fetch_news  # noqa: E402
from transforms.map_news_to_dclo import build_summaries, map_articles  # noqa: E402

INTAKE_COLUMNS = [
    "article_id",
    "fetched_at_utc",
    "source_id",
    "publisher_name",
    "source_type",
    "country_iso3",
    "language",
    "tier",
    "title",
    "summary",
    "url",
    "published_at",
    "primary_dclo_domain",
    "dclo_confidence",
    "ACC_score",
    "SKL_score",
    "SRV_score",
    "AGR_score",
    "ECO_score",
    "OUT_score",
    "matched_keywords_json",
    "secondary_dclo_domains_json",
]


def merge_intake(existing: pd.DataFrame, incoming: pd.DataFrame, lookback_days: int) -> pd.DataFrame:
    combined = pd.concat([existing, incoming], ignore_index=True)
    if combined.empty:
        return combined
    combined = combined.drop_duplicates(subset=["article_id"], keep="last")
    cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
    if "published_at" in combined.columns:
        published = pd.to_datetime(combined["published_at"], errors="coerce", utc=True)
        recent_mask = published.isna() | (published >= cutoff)
        combined = combined[recent_mask].copy()
    return combined.sort_values(["published_at", "country_iso3"], ascending=[False, True])


def read_config(config_path: Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def run(config_path: str) -> int:
    config = read_config(ROOT_DIR / config_path)
    output_cfg = config.get("output", {})
    data_dir = ROOT_DIR / str(output_cfg.get("data_dir", "./data"))
    gold_dir = data_dir / "gold"
    raw_dir = data_dir / str(output_cfg.get("raw_subdir", "raw/news"))
    gold_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    fetched_at = datetime.now(timezone.utc).isoformat()
    articles = fetch_news(config_path)
    if not articles:
        print("[news-agent] no articles fetched")
        return 1

    raw_path = raw_dir / f"dclo_news_raw_{fetched_at[:10]}.jsonl"
    with open(raw_path, "w", encoding="utf-8") as handle:
        for row in articles:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")
    print(f"[news-agent] wrote raw snapshot: {raw_path}")

    keyword_cfg = config.get("dclo_domain_keywords", {})
    mapped = map_articles(articles, keyword_cfg)
    incoming_df = pd.DataFrame(mapped)

    intake_path = gold_dir / str(output_cfg.get("gold_intake_file", "dclo_news_intake.csv"))
    lookback_days = int(config.get("fetch", {}).get("lookback_days", 14))
    if intake_path.exists():
        existing_df = pd.read_csv(intake_path)
        intake_df = merge_intake(existing_df, incoming_df, lookback_days=lookback_days)
    else:
        intake_df = incoming_df

    summary, country_summary = build_summaries(intake_df.to_dict(orient="records"))

    domain_summary_path = gold_dir / str(output_cfg.get("gold_domain_summary_file", "dclo_news_domain_summary.json"))
    country_summary_path = gold_dir / str(output_cfg.get("gold_country_summary_file", "dclo_news_country_summary.csv"))

    intake_df = intake_df if isinstance(intake_df, pd.DataFrame) else pd.DataFrame(intake_df)
    for col in INTAKE_COLUMNS:
        if col not in intake_df.columns:
            intake_df[col] = ""
    intake_df[INTAKE_COLUMNS].to_csv(intake_path, index=False)

    domain_summary_path.write_text(
        json.dumps(
            {
                "generated_at_utc": fetched_at,
                "config_path": config_path,
                **summary,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    country_summary.to_csv(country_summary_path, index=False)

    totals = summary.get("totals", {})
    print(f"[news-agent] articles: {totals.get('total_articles', 0)}")
    print(f"[news-agent] mapped: {totals.get('mapped_articles', 0)} ({totals.get('mapping_rate', 0)})")
    print(f"[news-agent] wrote intake: {intake_path}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="DCLO global digital news ingestion and mapping agent")
    parser.add_argument("--config", default="config/dclo_news_sources.yml")
    args = parser.parse_args()
    raise SystemExit(run(args.config))


if __name__ == "__main__":
    main()
