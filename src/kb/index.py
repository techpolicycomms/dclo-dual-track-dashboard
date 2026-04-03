"""Maintain the wiki's index files and source catalog.

Provides incremental index updates, catalog generation from raw/
metadata, and statistics about the knowledge base.

Usage:
    python -m src.kb.index                 # Rebuild index
    python -m src.kb.index --catalog       # Rebuild source catalog
    python -m src.kb.index --stats         # Print KB statistics
"""

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import yaml

from .config import load_config


def _collect_wiki_articles(wiki_dir: Path) -> list[dict]:
    """Scan wiki/ for all .md articles and parse front matter."""
    articles = []
    for md_path in sorted(wiki_dir.rglob("*.md")):
        if md_path.name.startswith("_"):
            continue
        text = md_path.read_text(errors="replace")
        # Parse basic metadata from the article
        title = ""
        summary = ""
        tags = []
        category = md_path.parent.name if md_path.parent != wiki_dir else "uncategorized"

        for line in text.splitlines():
            if line.startswith("# ") and not title:
                title = line[2:].strip()
            elif line.startswith("> **Summary**:"):
                summary = line.split(":", 1)[1].strip()
            elif line.startswith("- **Tags**:"):
                tag_str = line.split(":", 1)[1].strip()
                if tag_str and tag_str != "none":
                    tags = [t.strip() for t in tag_str.split(",")]

        # Count links
        import re
        links = re.findall(r"\[\[(.+?)\]\]", text)

        articles.append({
            "path": str(md_path.relative_to(wiki_dir)),
            "title": title or md_path.stem.replace("-", " ").title(),
            "summary": summary,
            "category": category,
            "tags": tags,
            "links": links,
            "word_count": len(text.split()),
            "size_bytes": md_path.stat().st_size,
        })
    return articles


def _collect_raw_meta(raw_dir: Path) -> list[dict]:
    """Collect all raw document metadata."""
    metas = []
    for meta_path in sorted(raw_dir.glob("*.meta.yml")):
        with open(meta_path) as f:
            meta = yaml.safe_load(f) or {}
        meta["_file"] = meta_path.name.replace(".meta.yml", "")
        metas.append(meta)
    return metas


def rebuild_index(cfg: dict) -> None:
    """Rebuild the master _index.md from current wiki state."""
    wiki_dir = Path(cfg["paths"]["wiki"])
    articles = _collect_wiki_articles(wiki_dir)

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    total_words = sum(a["word_count"] for a in articles)

    lines = [
        "# Knowledge Base Index",
        "",
        f"*Auto-generated on {ts}.*",
        "",
        f"**{len(articles)} articles** | **{total_words:,} words**",
        "",
        "---",
        "",
    ]

    by_category = defaultdict(list)
    for a in articles:
        by_category[a["category"]].append(a)

    for cat in sorted(by_category):
        lines.append(f"## {cat.replace('-', ' ').title()}")
        lines.append("")
        for a in sorted(by_category[cat], key=lambda x: x["title"]):
            summary_snippet = a["summary"][:100] + "..." if len(a["summary"]) > 100 else a["summary"]
            lines.append(f"- **[[{a['title']}]]** ({a['word_count']:,} words) — {summary_snippet}")
        lines.append("")

    index_path = wiki_dir / cfg["wiki"]["index_file"]
    index_path.write_text("\n".join(lines))
    print(f"Index rebuilt: {index_path} ({len(articles)} articles)")


def rebuild_catalog(cfg: dict) -> None:
    """Rebuild the _catalog.md from raw/ metadata."""
    raw_dir = Path(cfg["paths"]["raw"])
    wiki_dir = Path(cfg["paths"]["wiki"])
    metas = _collect_raw_meta(raw_dir)

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    lines = [
        "# Source Document Catalog",
        "",
        f"*Auto-generated on {ts}.*",
        "",
        f"**{len(metas)} source documents ingested.**",
        "",
        "| # | Document | Source | Category | Tags | Compiled |",
        "|---|----------|--------|----------|------|----------|",
    ]

    for i, meta in enumerate(metas, 1):
        name = meta.get("original_name", meta.get("_file", "?"))
        source = meta.get("source", "")[:50]
        cat = meta.get("category", "")
        tags = ", ".join(meta.get("tags", []))
        compiled = "yes" if meta.get("compiled") else "no"
        lines.append(f"| {i} | {name} | {source} | {cat} | {tags} | {compiled} |")

    lines.append("")
    catalog_path = wiki_dir / cfg["wiki"]["catalog_file"]
    catalog_path.write_text("\n".join(lines))
    print(f"Catalog rebuilt: {catalog_path} ({len(metas)} sources)")


def print_stats(cfg: dict) -> None:
    """Print knowledge base statistics."""
    raw_dir = Path(cfg["paths"]["raw"])
    wiki_dir = Path(cfg["paths"]["wiki"])

    metas = _collect_raw_meta(raw_dir)
    articles = _collect_wiki_articles(wiki_dir)

    total_words = sum(a["word_count"] for a in articles)
    total_raw_bytes = sum(m.get("size_bytes", 0) for m in metas)
    compiled_count = sum(1 for m in metas if m.get("compiled"))

    tag_counter = Counter()
    for a in articles:
        tag_counter.update(a["tags"])

    cat_counter = Counter(a["category"] for a in articles)

    link_counts = [len(a["links"]) for a in articles]
    orphans = [a for a in articles if not a["links"]]

    # All outbound link targets
    all_targets = set()
    for a in articles:
        all_targets.update(a["links"])
    all_titles = {a["title"] for a in articles}
    broken = all_targets - all_titles

    print("=" * 50)
    print("  KNOWLEDGE BASE STATISTICS")
    print("=" * 50)
    print()
    print(f"  Raw documents:     {len(metas)}")
    print(f"    Compiled:        {compiled_count}")
    print(f"    Pending:         {len(metas) - compiled_count}")
    print(f"    Total size:      {total_raw_bytes / 1024:.1f} KB")
    print()
    print(f"  Wiki articles:     {len(articles)}")
    print(f"    Total words:     {total_words:,}")
    print(f"    Avg words/art:   {total_words // max(len(articles), 1):,}")
    print()
    print(f"  Links:")
    print(f"    Total outbound:  {sum(link_counts)}")
    print(f"    Avg per article: {sum(link_counts) / max(len(articles), 1):.1f}")
    print(f"    Orphan articles: {len(orphans)}")
    print(f"    Broken links:    {len(broken)}")
    print()

    if cat_counter:
        print("  Categories:")
        for cat, count in cat_counter.most_common(10):
            print(f"    {cat}: {count}")
        print()

    if tag_counter:
        print("  Top tags:")
        for tag, count in tag_counter.most_common(10):
            print(f"    {tag}: {count}")

    # Also emit as JSON for programmatic use
    stats = {
        "raw_documents": len(metas),
        "compiled": compiled_count,
        "wiki_articles": len(articles),
        "total_words": total_words,
        "orphans": len(orphans),
        "broken_links": len(broken),
    }
    stats_path = Path(cfg["paths"]["wiki"]) / "_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Manage KB index and catalog")
    parser.add_argument("--catalog", action="store_true", help="Rebuild source catalog")
    parser.add_argument("--stats", action="store_true", help="Print KB statistics")
    args = parser.parse_args()

    cfg = load_config()

    if args.stats:
        print_stats(cfg)
    elif args.catalog:
        rebuild_catalog(cfg)
    else:
        rebuild_index(cfg)


if __name__ == "__main__":
    main()
