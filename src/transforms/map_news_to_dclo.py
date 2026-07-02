"""Map digital/tech news headlines to DCLO formative domains."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
import yaml

DCLO_DOMAINS = ("ACC", "SKL", "SRV", "AGR", "ECO", "OUT")


def read_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(config)


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip().lower()


def _compile_keyword_index(keyword_cfg: Dict[str, Any]) -> Dict[str, List[Tuple[str, str]]]:
    index: Dict[str, List[Tuple[str, str]]] = {domain: [] for domain in DCLO_DOMAINS}
    for domain in DCLO_DOMAINS:
        domain_cfg = keyword_cfg.get(domain, {})
        if not isinstance(domain_cfg, dict):
            continue
        for language, keywords in domain_cfg.items():
            if not isinstance(keywords, list):
                continue
            for keyword in keywords:
                normalized = _normalize_text(keyword)
                if normalized:
                    index[domain].append((language, normalized))
    return index


def score_article(text: str, keyword_index: Dict[str, List[Tuple[str, str]]]) -> Dict[str, Any]:
    normalized = _normalize_text(text)
    domain_scores: Dict[str, float] = {domain: 0.0 for domain in DCLO_DOMAINS}
    matched: Dict[str, List[str]] = {domain: [] for domain in DCLO_DOMAINS}

    for domain, keywords in keyword_index.items():
        for language, keyword in keywords:
            if keyword in normalized:
                domain_scores[domain] += 1.0
                token = f"{language}:{keyword}"
                if token not in matched[domain]:
                    matched[domain].append(token)

    ranked = sorted(domain_scores.items(), key=lambda item: (-item[1], item[0]))
    primary_domain = ranked[0][0] if ranked[0][1] > 0 else "UNMAPPED"
    confidence = ranked[0][1] / max(sum(domain_scores.values()), 1.0)

    secondary_domains = [domain for domain, score in ranked[1:3] if score > 0]

    return {
        "primary_dclo_domain": primary_domain,
        "dclo_confidence": round(confidence, 4),
        "ACC_score": domain_scores["ACC"],
        "SKL_score": domain_scores["SKL"],
        "SRV_score": domain_scores["SRV"],
        "AGR_score": domain_scores["AGR"],
        "ECO_score": domain_scores["ECO"],
        "OUT_score": domain_scores["OUT"],
        "matched_keywords": matched,
        "secondary_dclo_domains": secondary_domains,
    }


def map_articles(articles: List[Dict[str, Any]], keyword_cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    keyword_index = _compile_keyword_index(keyword_cfg)
    mapped: List[Dict[str, Any]] = []
    for article in articles:
        text = f"{article.get('title', '')} {article.get('summary', '')}"
        scores = score_article(text, keyword_index)
        row = {**article, **scores}
        row["matched_keywords_json"] = json.dumps(scores["matched_keywords"], ensure_ascii=False)
        row["secondary_dclo_domains_json"] = json.dumps(scores["secondary_dclo_domains"], ensure_ascii=False)
        del row["matched_keywords"]
        del row["secondary_dclo_domains"]
        mapped.append(row)
    return mapped


def build_summaries(mapped: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], pd.DataFrame]:
    df = pd.DataFrame(mapped)
    if df.empty:
        domain_summary = {domain: 0 for domain in DCLO_DOMAINS}
        domain_summary["UNMAPPED"] = 0
        country_summary = pd.DataFrame(columns=["country_iso3", "language", "article_count", "mapped_count"])
        return {"totals": domain_summary, "by_publisher": {}, "by_country": {}}, country_summary

    domain_summary = {
        domain: int((df["primary_dclo_domain"] == domain).sum()) for domain in DCLO_DOMAINS
    }
    domain_summary["UNMAPPED"] = int((df["primary_dclo_domain"] == "UNMAPPED").sum())
    domain_summary["total_articles"] = int(len(df))
    domain_summary["mapped_articles"] = int((df["primary_dclo_domain"] != "UNMAPPED").sum())
    domain_summary["mapping_rate"] = round(domain_summary["mapped_articles"] / max(len(df), 1), 4)

    by_publisher = (
        df.groupby(["publisher_name", "primary_dclo_domain"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .to_dict(orient="records")
    )
    by_country = (
        df.groupby(["country_iso3", "primary_dclo_domain"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .to_dict(orient="records")
    )

    country_summary = (
        df.groupby(["country_iso3", "language"])
        .agg(
            article_count=("article_id", "count"),
            mapped_count=("primary_dclo_domain", lambda s: int((s != "UNMAPPED").sum())),
        )
        .reset_index()
        .sort_values(["article_count", "country_iso3"], ascending=[False, True])
    )

    summary = {
        "totals": domain_summary,
        "by_publisher": by_publisher,
        "by_country": by_country,
    }
    return summary, country_summary


def run(config_path: str, articles: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any], pd.DataFrame]:
    config = read_config(config_path)
    keyword_cfg = config.get("dclo_domain_keywords", {})
    mapped = map_articles(articles, keyword_cfg)
    summary, country_summary = build_summaries(mapped)
    return mapped, summary, country_summary


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Map fetched news articles to DCLO domains.")
    parser.add_argument("--config", default="config/dclo_news_sources.yml")
    parser.add_argument("--input", required=True, help="Input JSONL from fetch_digital_news")
    parser.add_argument("--output", required=True, help="Output CSV path")
    args = parser.parse_args()

    input_path = Path(args.input)
    articles: List[Dict[str, Any]] = []
    with open(input_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                articles.append(json.loads(line))

    mapped, summary, country_summary = run(args.config, articles)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(mapped).to_csv(output_path, index=False)
    print(json.dumps(summary["totals"], indent=2))
    print(f"[news-map] wrote {output_path} ({len(mapped)} rows)")


if __name__ == "__main__":
    main()
