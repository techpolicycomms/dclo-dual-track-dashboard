"""Generate outputs from the wiki: reports, slides, and visualizations.

Supports multiple output formats that can be viewed in Obsidian
or other markdown-compatible tools.

Usage:
    python -m src.kb.output report --topic "DCLO methodology"
    python -m src.kb.output slides --topic "causal evidence" --format marp
    python -m src.kb.output chart --type category-treemap
"""

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from .config import load_config
from .search import WikiIndex


def _ensure_output_dir(cfg: dict, subdir: str = "") -> Path:
    out = Path(cfg["paths"]["outputs"])
    if subdir:
        out = out / subdir
    out.mkdir(parents=True, exist_ok=True)
    return out


def generate_report(cfg: dict, topic: str, depth: int = 5) -> Path:
    """Generate a research report by searching the wiki for a topic.

    Assembles relevant article excerpts into a structured report.
    """
    wiki_dir = Path(cfg["paths"]["wiki"])
    idx = WikiIndex(wiki_dir, cfg)
    results = idx.search(topic, top_k=depth)

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    out_dir = _ensure_output_dir(cfg, "reports")

    slug = topic.lower().replace(" ", "-")[:40]
    out_path = out_dir / f"report-{slug}-{datetime.now(timezone.utc).strftime('%Y%m%d')}.md"

    lines = [
        f"# Report: {topic.title()}",
        "",
        f"*Generated on {ts}*",
        "",
        f"Based on {len(results)} relevant article(s) from the knowledge base.",
        "",
        "---",
        "",
    ]

    if not results:
        lines.append("*No matching articles found. Try broadening the search topic.*")
    else:
        for i, r in enumerate(results, 1):
            lines.extend([
                f"## {i}. {r['title']}",
                "",
                f"**Relevance score**: {r['score']:.2f}",
                "",
            ])
            if r["summary"]:
                lines.extend([r["summary"], ""])
            lines.extend([
                "### Excerpt",
                "",
                r["snippet"],
                "",
                f"*Source: {r['path']}*",
                "",
                "---",
                "",
            ])

    lines.extend([
        "## Further Research",
        "",
        "Consider exploring:",
        "",
    ])
    # Suggest related concepts from the results
    all_snippets = " ".join(r["snippet"] for r in results)
    import re
    bold_terms = re.findall(r"\*\*(.+?)\*\*", all_snippets)
    for term in sorted(set(bold_terms))[:10]:
        lines.append(f"- {term}")
    lines.append("")

    out_path.write_text("\n".join(lines))
    print(f"Report generated: {out_path}")
    return out_path


def generate_slides(cfg: dict, topic: str, num_slides: int = 8) -> Path:
    """Generate a Marp-format slide deck from wiki content.

    Marp slides are viewable in Obsidian with the Marp plugin.
    """
    wiki_dir = Path(cfg["paths"]["wiki"])
    idx = WikiIndex(wiki_dir, cfg)
    results = idx.search(topic, top_k=num_slides)

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    out_dir = _ensure_output_dir(cfg, "slides")

    slug = topic.lower().replace(" ", "-")[:40]
    out_path = out_dir / f"slides-{slug}-{ts}.md"

    lines = [
        "---",
        "marp: true",
        "theme: default",
        "paginate: true",
        f"title: {topic.title()}",
        "---",
        "",
        f"# {topic.title()}",
        "",
        f"Knowledge Base Summary — {ts}",
        "",
        "---",
        "",
    ]

    for r in results:
        # Each result becomes a slide
        lines.extend([
            f"## {r['title']}",
            "",
        ])
        if r["summary"]:
            lines.append(r["summary"][:300])
            lines.append("")
        lines.extend([
            f"> {r['snippet'][:200]}",
            "",
            f"*Score: {r['score']:.2f} | Source: {r['path']}*",
            "",
            "---",
            "",
        ])

    # Closing slide
    lines.extend([
        "## Summary",
        "",
        f"- **{len(results)} sources** consulted",
        f"- Topic: **{topic}**",
        f"- Generated: {ts}",
        "",
        "*Auto-generated from the LLM Knowledge Base*",
        "",
    ])

    out_path.write_text("\n".join(lines))
    print(f"Slides generated: {out_path}")
    return out_path


def generate_stats_chart(cfg: dict) -> Path:
    """Generate a matplotlib visualization of KB structure.

    Outputs a Python script + PNG chart of category distribution,
    article sizes, and link density.
    """
    wiki_dir = Path(cfg["paths"]["wiki"])
    out_dir = _ensure_output_dir(cfg, "charts")

    # Collect stats
    articles = []
    for md_path in sorted(wiki_dir.rglob("*.md")):
        if md_path.name.startswith("_"):
            continue
        text = md_path.read_text(errors="replace")
        import re
        title = ""
        for line in text.splitlines():
            if line.startswith("# ") and not title:
                title = line[2:].strip()
                break

        cat = md_path.parent.name if md_path.parent != wiki_dir else "root"
        links = len(re.findall(r"\[\[.+?\]\]", text))
        articles.append({
            "title": title or md_path.stem,
            "category": cat,
            "words": len(text.split()),
            "links": links,
        })

    if not articles:
        print("No articles to chart.")
        return out_dir

    # Generate a self-contained Python script for the chart
    chart_data = json.dumps(articles, indent=2)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d")

    script = f'''"""Auto-generated KB visualization script."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import Counter

data = json.loads("""{chart_data}""")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Knowledge Base Overview", fontsize=14, fontweight="bold")

# 1. Articles per category
cats = Counter(d["category"] for d in data)
axes[0].barh(list(cats.keys()), list(cats.values()), color="#4C78A8")
axes[0].set_title("Articles by Category")
axes[0].set_xlabel("Count")

# 2. Word count distribution
words = [d["words"] for d in data]
axes[1].hist(words, bins=20, color="#F58518", edgecolor="white")
axes[1].set_title("Word Count Distribution")
axes[1].set_xlabel("Words")
axes[1].set_ylabel("Articles")

# 3. Links per article
link_counts = [d["links"] for d in data]
axes[2].hist(link_counts, bins=15, color="#E45756", edgecolor="white")
axes[2].set_title("Links per Article")
axes[2].set_xlabel("Outbound Links")
axes[2].set_ylabel("Articles")

plt.tight_layout()
plt.savefig("{out_dir}/kb-overview-{ts}.png", dpi=150, bbox_inches="tight")
print("Chart saved: {out_dir}/kb-overview-{ts}.png")
'''

    script_path = out_dir / f"kb-overview-{ts}.py"
    script_path.write_text(script)

    # Try to run it
    try:
        import subprocess
        result = subprocess.run(
            ["python", str(script_path)],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0:
            print(result.stdout.strip())
        else:
            print(f"Chart script saved: {script_path}")
            print(f"  Run manually: python {script_path}")
            if result.stderr:
                print(f"  Note: {result.stderr.strip()[:200]}")
    except Exception:
        print(f"Chart script saved: {script_path}")
        print(f"  Run: python {script_path}")

    return script_path


def main():
    parser = argparse.ArgumentParser(description="Generate outputs from wiki")
    sub = parser.add_subparsers(dest="command")

    rp = sub.add_parser("report", help="Generate a research report")
    rp.add_argument("--topic", required=True, help="Report topic")
    rp.add_argument("--depth", type=int, default=5, help="Number of sources")

    sp = sub.add_parser("slides", help="Generate Marp slides")
    sp.add_argument("--topic", required=True, help="Slide deck topic")
    sp.add_argument("--num", type=int, default=8, help="Number of slides")

    cp = sub.add_parser("chart", help="Generate KB visualization")

    args = parser.parse_args()
    cfg = load_config()

    if args.command == "report":
        generate_report(cfg, args.topic, args.depth)
    elif args.command == "slides":
        generate_slides(cfg, args.topic, args.num)
    elif args.command == "chart":
        generate_stats_chart(cfg)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
