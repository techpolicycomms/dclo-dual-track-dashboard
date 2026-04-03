"""Compile raw documents into a structured wiki.

Reads all documents in kb/raw/, extracts content and metadata,
and produces a set of interlinked .md files in kb/wiki/.

The compiler:
1. Reads each raw document and its .meta.yml sidecar
2. Generates a summary and identifies key concepts
3. Creates/updates wiki articles organized by category
4. Builds backlinks and cross-references between articles
5. Rebuilds the master index

Usage:
    python -m src.kb.compile
    python -m src.kb.compile --full    # Force full recompile
"""

import argparse
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import yaml

from .config import load_config


def _load_meta(raw_dir: Path) -> list[dict]:
    """Load all metadata sidecars from raw/."""
    metas = []
    for meta_path in sorted(raw_dir.glob("*.meta.yml")):
        with open(meta_path) as f:
            meta = yaml.safe_load(f)
        meta["_meta_path"] = meta_path
        meta["_doc_path"] = meta_path.with_name(
            meta_path.name.replace(".meta.yml", "")
        )
        # Handle compound extensions like .csv.meta.yml
        stem = meta_path.name
        for _ in range(2):  # strip .meta.yml
            stem = Path(stem).stem
        meta["_stem"] = stem
        metas.append(meta)
    return metas


def _read_document(doc_path: Path) -> str:
    """Read document content, handling various formats."""
    if not doc_path.exists():
        return ""
    suffix = doc_path.suffix.lower()
    if suffix in (".md", ".txt", ".html", ".yml", ".yaml"):
        return doc_path.read_text(errors="replace")
    if suffix == ".csv":
        # Return first 100 lines as preview
        lines = doc_path.read_text(errors="replace").splitlines()
        return "\n".join(lines[:100])
    if suffix == ".json":
        return doc_path.read_text(errors="replace")[:5000]
    return f"[Binary file: {doc_path.name}, {doc_path.stat().st_size} bytes]"


def _extract_title(content: str, meta: dict) -> str:
    """Extract title from content or derive from metadata."""
    # Try markdown H1
    match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
    if match:
        return match.group(1).strip()
    return meta.get("original_name", "Untitled").rsplit(".", 1)[0].replace("-", " ").replace("_", " ").title()


def _extract_summary(content: str, max_words: int = 150) -> str:
    """Extract a brief summary from the first paragraph(s)."""
    # Skip frontmatter
    text = re.sub(r"^---.*?---\s*", "", content, flags=re.DOTALL)
    # Skip headings at start
    text = re.sub(r"^#+\s+.*\n", "", text).strip()
    # Get first paragraph
    paragraphs = re.split(r"\n\s*\n", text)
    for p in paragraphs:
        p = p.strip()
        if len(p) > 30:
            words = p.split()
            if len(words) > max_words:
                return " ".join(words[:max_words]) + "..."
            return p
    return content[:300].strip() + "..." if len(content) > 300 else content.strip()


def _extract_concepts(content: str) -> list[str]:
    """Extract key concepts (bold terms, heading terms, linked terms)."""
    concepts = set()
    # Bold terms
    for m in re.finditer(r"\*\*(.+?)\*\*", content):
        term = m.group(1).strip()
        if 2 < len(term) < 60:
            concepts.add(term)
    # Heading terms
    for m in re.finditer(r"^#{2,4}\s+(.+)$", content, re.MULTILINE):
        term = m.group(1).strip()
        if 2 < len(term) < 60:
            concepts.add(term)
    # Wiki-style links
    for m in re.finditer(r"\[\[(.+?)\]\]", content):
        concepts.add(m.group(1).strip())
    return sorted(concepts)


def _slugify(text: str) -> str:
    """Convert text to a URL/filename-safe slug."""
    slug = text.lower().strip()
    slug = re.sub(r"[^\w\s-]", "", slug)
    slug = re.sub(r"[\s_]+", "-", slug)
    return slug.strip("-")[:80]


def _build_article(
    title: str,
    summary: str,
    content: str,
    meta: dict,
    concepts: list[str],
    backlinks: list[str],
) -> str:
    """Build a wiki article in markdown."""
    lines = [
        f"# {title}",
        "",
        f"> **Summary**: {summary}",
        "",
        f"- **Source**: {meta.get('source', 'unknown')}",
        f"- **Ingested**: {meta.get('ingested_at', 'unknown')}",
        f"- **Tags**: {', '.join(meta.get('tags', [])) or 'none'}",
        f"- **Category**: {meta.get('category', 'uncategorized')}",
        "",
        "---",
        "",
    ]

    # Main content (truncated if very long)
    if len(content) > 20000:
        lines.append(content[:20000])
        lines.append(f"\n\n*[Content truncated — full document: {meta.get('original_name', 'raw')}]*")
    else:
        lines.append(content)

    # Concepts section
    if concepts:
        lines.extend([
            "",
            "---",
            "",
            "## Key Concepts",
            "",
        ])
        for c in concepts[:30]:
            slug = _slugify(c)
            lines.append(f"- [[{c}]]")

    # Backlinks section
    if backlinks:
        lines.extend([
            "",
            "---",
            "",
            "## Backlinks",
            "",
        ])
        for bl in sorted(set(backlinks)):
            lines.append(f"- [[{bl}]]")

    lines.append("")
    return "\n".join(lines)


def compile_wiki(cfg: dict, full: bool = False) -> dict:
    """Compile raw documents into wiki articles. Returns compile stats."""
    raw_dir = Path(cfg["paths"]["raw"])
    wiki_dir = Path(cfg["paths"]["wiki"])
    wiki_dir.mkdir(parents=True, exist_ok=True)

    max_summary = cfg["wiki"]["max_summary_words"]
    use_categories = cfg["compile"]["category_dirs"]

    metas = _load_meta(raw_dir)
    if not metas:
        print("No raw documents found. Ingest some first.")
        return {"articles": 0, "concepts": 0}

    # Phase 1: Read all documents and extract metadata
    articles = []
    all_concepts = defaultdict(list)  # concept -> list of article titles

    for meta in metas:
        if not full and meta.get("compiled"):
            continue

        content = _read_document(meta["_doc_path"])
        if not content:
            continue

        title = _extract_title(content, meta)
        summary = _extract_summary(content, max_summary)
        concepts = _extract_concepts(content)

        for c in concepts:
            all_concepts[c].append(title)

        articles.append({
            "title": title,
            "summary": summary,
            "content": content,
            "meta": meta,
            "concepts": concepts,
            "slug": _slugify(title),
        })

    # Phase 2: Build backlink map
    backlink_map = defaultdict(list)
    for art in articles:
        for other in articles:
            if art["title"] == other["title"]:
                continue
            # Check if this article references the other
            if other["title"].lower() in art["content"].lower():
                backlink_map[other["title"]].append(art["title"])

    # Phase 3: Write wiki articles
    written = 0
    for art in articles:
        category = art["meta"].get("category", "uncategorized") or "uncategorized"
        if use_categories:
            out_dir = wiki_dir / _slugify(category)
        else:
            out_dir = wiki_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        backlinks = backlink_map.get(art["title"], [])
        md = _build_article(
            art["title"], art["summary"], art["content"],
            art["meta"], art["concepts"], backlinks,
        )

        out_path = out_dir / f"{art['slug']}.md"
        out_path.write_text(md)
        written += 1

        # Mark as compiled in meta
        art["meta"]["compiled"] = True
        meta_path = art["meta"]["_meta_path"]
        meta_copy = {k: v for k, v in art["meta"].items() if not k.startswith("_")}
        with open(meta_path, "w") as f:
            yaml.dump(meta_copy, f, default_flow_style=False, sort_keys=False)

    # Phase 4: Generate concept index articles for frequently-referenced concepts
    concept_threshold = cfg["wiki"]["concept_threshold"]
    concept_dir = wiki_dir / "concepts"
    concept_dir.mkdir(exist_ok=True)
    concept_count = 0

    for concept, referencing_articles in all_concepts.items():
        if len(referencing_articles) >= concept_threshold:
            slug = _slugify(concept)
            concept_md = [
                f"# {concept}",
                "",
                f"*Concept referenced in {len(referencing_articles)} article(s).*",
                "",
                "## Referenced In",
                "",
            ]
            for ref in sorted(set(referencing_articles)):
                concept_md.append(f"- [[{ref}]]")
            concept_md.append("")

            (concept_dir / f"{slug}.md").write_text("\n".join(concept_md))
            concept_count += 1

    # Phase 5: Rebuild master index
    if cfg["compile"]["rebuild_index"]:
        _build_master_index(wiki_dir, articles, all_concepts, cfg)

    stats = {
        "articles": written,
        "concepts": concept_count,
        "total_concepts_found": len(all_concepts),
        "backlinks_created": sum(len(v) for v in backlink_map.values()),
    }
    print(f"\nCompile complete:")
    print(f"  Articles written: {written}")
    print(f"  Concept stubs: {concept_count}")
    print(f"  Backlinks: {stats['backlinks_created']}")
    return stats


def _build_master_index(
    wiki_dir: Path,
    articles: list[dict],
    all_concepts: dict,
    cfg: dict,
) -> None:
    """Build the master _index.md file."""
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# Knowledge Base Index",
        "",
        f"*Auto-generated on {ts}. Do not edit manually.*",
        "",
        f"**{len(articles)} articles** | **{len(all_concepts)} concepts**",
        "",
        "---",
        "",
        "## Articles",
        "",
    ]

    # Group by category
    by_category = defaultdict(list)
    for art in articles:
        cat = art["meta"].get("category", "uncategorized") or "uncategorized"
        by_category[cat].append(art)

    for cat in sorted(by_category):
        lines.append(f"### {cat.title()}")
        lines.append("")
        for art in sorted(by_category[cat], key=lambda a: a["title"]):
            lines.append(f"- **[[{art['title']}]]** — {art['summary'][:80]}...")
        lines.append("")

    # Top concepts
    top_concepts = sorted(all_concepts.items(), key=lambda x: -len(x[1]))[:50]
    if top_concepts:
        lines.extend([
            "---",
            "",
            "## Top Concepts",
            "",
        ])
        for concept, refs in top_concepts:
            lines.append(f"- **{concept}** ({len(refs)} refs)")
        lines.append("")

    (wiki_dir / cfg["wiki"]["index_file"]).write_text("\n".join(lines))
    print(f"  Index: {cfg['wiki']['index_file']}")


def main():
    parser = argparse.ArgumentParser(description="Compile wiki from raw documents")
    parser.add_argument("--full", action="store_true", help="Force full recompile")
    args = parser.parse_args()

    cfg = load_config()
    compile_wiki(cfg, full=args.full)


if __name__ == "__main__":
    main()
