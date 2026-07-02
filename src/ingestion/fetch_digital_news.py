"""RSS / Atom ingestion for global digital and technology news publishers."""

from __future__ import annotations

import hashlib
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import quote_plus

import requests
import yaml

ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}


def read_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _clean_text(value: Optional[str]) -> str:
    if not value:
        return ""
    text = re.sub(r"<[^>]+>", " ", value)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _parse_datetime(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    value = value.strip()
    try:
        dt = parsedate_to_datetime(value)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).isoformat()
    except (TypeError, ValueError, IndexError):
        pass
    try:
        normalized = value.replace("Z", "+00:00")
        dt = datetime.fromisoformat(normalized)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).isoformat()
    except ValueError:
        return None


def _article_id(url: str, title: str) -> str:
    material = f"{url}|{title}".encode("utf-8")
    return hashlib.sha256(material).hexdigest()[:24]


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _find_text(node: ET.Element, names: List[str]) -> str:
    for name in names:
        child = node.find(name)
        if child is not None and child.text:
            return _clean_text(child.text)
        child = node.find(f"atom:{name}", ATOM_NS)
        if child is not None and child.text:
            return _clean_text(child.text)
    return ""


def _find_link(node: ET.Element) -> str:
    link = node.find("link")
    if link is not None and link.get("href"):
        return str(link.get("href"))
    if link is not None and link.text:
        return _clean_text(link.text)
    atom_link = node.find("atom:link[@rel='alternate']", ATOM_NS)
    if atom_link is not None and atom_link.get("href"):
        return str(atom_link.get("href"))
    atom_link = node.find("atom:link", ATOM_NS)
    if atom_link is not None and atom_link.get("href"):
        return str(atom_link.get("href"))
    return ""


def parse_feed_xml(content: bytes) -> List[Dict[str, str]]:
    root = ET.fromstring(content)
    tag = _local_name(root.tag).lower()
    entries: List[ET.Element] = []

    if tag == "rss":
        channel = root.find("channel")
        if channel is None:
            return []
        entries = list(channel.findall("item"))
    elif tag == "feed":
        entries = list(root.findall("atom:entry", ATOM_NS))
        if not entries:
            entries = list(root.findall("entry"))

    rows: List[Dict[str, str]] = []
    for entry in entries:
        title = _find_text(entry, ["title"])
        link = _find_link(entry)
        if not title or not link:
            continue
        summary = _find_text(entry, ["description", "summary", "content"])
        published_raw = _find_text(entry, ["pubDate", "published", "updated"])
        rows.append(
            {
                "title": title,
                "url": link,
                "summary": summary,
                "published_at": _parse_datetime(published_raw) or "",
            }
        )
    return rows


def google_news_rss_url(query: str, hl: str, gl: str, ceid: str) -> str:
    encoded = quote_plus(query)
    return f"https://news.google.com/rss/search?q={encoded}&hl={hl}&gl={gl}&ceid={ceid}"


def fetch_feed(
    rss_url: str,
    timeout_seconds: int,
    user_agent: str,
    max_articles: int,
) -> List[Dict[str, str]]:
    headers = {"User-Agent": user_agent, "Accept": "application/rss+xml, application/xml, text/xml"}
    response = requests.get(rss_url, headers=headers, timeout=timeout_seconds)
    response.raise_for_status()
    rows = parse_feed_xml(response.content)
    return rows[:max_articles]


def build_source_record(
    item: Dict[str, str],
    *,
    source_id: str,
    publisher_name: str,
    country_iso3: str,
    language: str,
    tier: str,
    source_type: str,
    fetched_at_utc: str,
) -> Dict[str, Any]:
    title = item["title"]
    url = item["url"]
    return {
        "article_id": _article_id(url, title),
        "fetched_at_utc": fetched_at_utc,
        "source_id": source_id,
        "publisher_name": publisher_name,
        "source_type": source_type,
        "country_iso3": country_iso3,
        "language": language,
        "tier": tier,
        "title": title,
        "summary": item.get("summary", ""),
        "url": url,
        "published_at": item.get("published_at", ""),
    }


def run(config_path: str) -> List[Dict[str, Any]]:
    config = read_config(config_path)
    fetch_cfg = config.get("fetch", {})
    timeout_seconds = int(fetch_cfg.get("timeout_seconds", 25))
    max_articles = int(fetch_cfg.get("max_articles_per_feed", 30))
    user_agent = str(fetch_cfg.get("user_agent", "DCLO-News-Agent/1.0"))
    fetched_at = datetime.now(timezone.utc).isoformat()

    articles: List[Dict[str, Any]] = []
    seen_urls: set[str] = set()

    for publisher in config.get("publishers", []):
        if not isinstance(publisher, dict):
            continue
        rss_url = str(publisher.get("rss_url", "")).strip()
        if not rss_url:
            continue
        source_id = str(publisher.get("id", publisher.get("name", "publisher")))
        try:
            items = fetch_feed(rss_url, timeout_seconds, user_agent, max_articles)
        except Exception as exc:  # noqa: BLE001 - continue other feeds
            print(f"[news-fetch] failed {source_id}: {exc}")
            continue
        for item in items:
            url = item["url"]
            if url in seen_urls:
                continue
            seen_urls.add(url)
            articles.append(
                build_source_record(
                    item,
                    source_id=source_id,
                    publisher_name=str(publisher.get("name", source_id)),
                    country_iso3=str(publisher.get("country_iso3", "")),
                    language=str(publisher.get("language", "en")),
                    tier=str(publisher.get("tier", "national")),
                    source_type="publisher_rss",
                    fetched_at_utc=fetched_at,
                )
            )
        print(f"[news-fetch] {source_id}: {len(items)} items")

    for search in config.get("google_news_searches", []):
        if not isinstance(search, dict):
            continue
        source_id = str(search.get("id", "google_news"))
        query = str(search.get("query", "")).strip()
        if not query:
            continue
        rss_url = google_news_rss_url(
            query=query,
            hl=str(search.get("hl", "en")),
            gl=str(search.get("gl", "US")),
            ceid=str(search.get("ceid", "US:en")),
        )
        try:
            items = fetch_feed(rss_url, timeout_seconds, user_agent, max_articles)
        except Exception as exc:  # noqa: BLE001 - continue other feeds
            print(f"[news-fetch] failed {source_id}: {exc}")
            continue
        for item in items:
            url = item["url"]
            if url in seen_urls:
                continue
            seen_urls.add(url)
            articles.append(
                build_source_record(
                    item,
                    source_id=source_id,
                    publisher_name=str(search.get("publisher_name", "Google News")),
                    country_iso3=str(search.get("country_iso3", "")),
                    language=str(search.get("language", "en")),
                    tier="regional_language",
                    source_type="google_news_rss",
                    fetched_at_utc=fetched_at,
                )
            )
        print(f"[news-fetch] {source_id}: {len(items)} items")

    return articles


def main() -> None:
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Fetch digital/tech news from global RSS publishers.")
    parser.add_argument("--config", default="config/dclo_news_sources.yml")
    parser.add_argument("--output", default=None, help="Optional JSONL output path")
    args = parser.parse_args()

    articles = run(args.config)
    print(f"[news-fetch] total unique articles: {len(articles)}")
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as handle:
            for row in articles:
                handle.write(json.dumps(row, ensure_ascii=False))
                handle.write("\n")
        print(f"[news-fetch] wrote {out_path}")


if __name__ == "__main__":
    main()
