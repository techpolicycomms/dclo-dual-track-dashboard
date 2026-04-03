"""Wiki health checks and linting.

Runs a battery of checks against the wiki to find issues like
orphan articles, broken links, missing summaries, stale sources,
and near-duplicate content.

Usage:
    python -m src.kb.lint
    python -m src.kb.lint --check orphan_detection broken_links
    python -m src.kb.lint --fix   # Auto-fix what's possible
"""

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import yaml

from .config import load_config


class LintResult:
    """Single lint finding."""

    def __init__(self, check: str, severity: str, path: str, message: str):
        self.check = check
        self.severity = severity  # "error", "warning", "info"
        self.path = path
        self.message = message

    def to_dict(self) -> dict:
        return {
            "check": self.check,
            "severity": self.severity,
            "path": self.path,
            "message": self.message,
        }

    def __str__(self):
        icon = {"error": "E", "warning": "W", "info": "I"}.get(self.severity, "?")
        return f"  [{icon}] {self.path}: {self.message}"


def _scan_wiki(wiki_dir: Path) -> list[dict]:
    """Scan all wiki articles."""
    articles = []
    for md_path in sorted(wiki_dir.rglob("*.md")):
        if md_path.name.startswith("_"):
            continue
        text = md_path.read_text(errors="replace")
        title = ""
        summary = ""
        has_summary_section = False

        for line in text.splitlines():
            if line.startswith("# ") and not title:
                title = line[2:].strip()
            if line.startswith("> **Summary**:"):
                summary = line.split(":", 1)[1].strip()
                has_summary_section = True

        links = re.findall(r"\[\[(.+?)\]\]", text)
        rel = str(md_path.relative_to(wiki_dir))

        articles.append({
            "path": rel,
            "abs_path": md_path,
            "title": title or md_path.stem.replace("-", " ").title(),
            "summary": summary,
            "has_summary": has_summary_section,
            "links": links,
            "word_count": len(text.split()),
            "content": text,
        })
    return articles


def check_orphans(articles: list[dict]) -> list[LintResult]:
    """Find articles with no inbound links."""
    results = []
    all_link_targets = set()
    for a in articles:
        all_link_targets.update(a["links"])

    titles = {a["title"] for a in articles}
    for a in articles:
        if a["path"].startswith("concepts/"):
            continue
        is_linked = a["title"] in all_link_targets
        if not is_linked:
            results.append(LintResult(
                "orphan_detection", "warning", a["path"],
                f"Orphan article: '{a['title']}' has no inbound links",
            ))
    return results


def check_broken_links(articles: list[dict]) -> list[LintResult]:
    """Find links pointing to non-existent articles."""
    results = []
    titles = {a["title"] for a in articles}

    for a in articles:
        for link in a["links"]:
            if link not in titles:
                results.append(LintResult(
                    "broken_links", "error", a["path"],
                    f"Broken link: [[{link}]] (target not found)",
                ))
    return results


def check_missing_summaries(articles: list[dict]) -> list[LintResult]:
    """Find articles without a summary section."""
    results = []
    for a in articles:
        if a["path"].startswith("concepts/"):
            continue
        if not a["has_summary"]:
            results.append(LintResult(
                "missing_summaries", "warning", a["path"],
                f"Missing summary section in '{a['title']}'",
            ))
    return results


def check_stale_sources(raw_dir: Path, articles: list[dict]) -> list[LintResult]:
    """Find raw documents not reflected in any wiki article."""
    results = []
    raw_names = set()
    for meta_path in raw_dir.glob("*.meta.yml"):
        with open(meta_path) as f:
            meta = yaml.safe_load(f) or {}
        if not meta.get("compiled"):
            raw_names.add(meta.get("original_name", meta_path.stem))

    if raw_names:
        for name in sorted(raw_names):
            results.append(LintResult(
                "stale_sources", "info", f"raw/{name}",
                f"Source '{name}' has not been compiled into wiki",
            ))
    return results


def check_duplicates(articles: list[dict], threshold: float = 0.85) -> list[LintResult]:
    """Find near-duplicate articles using Jaccard similarity on token sets."""
    results = []

    def _token_set(text: str) -> set[str]:
        return set(re.findall(r"\w{3,}", text.lower()))

    seen_pairs = set()
    for i, a in enumerate(articles):
        for j, b in enumerate(articles):
            if i >= j:
                continue
            pair = (a["path"], b["path"])
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)

            set_a = _token_set(a["content"])
            set_b = _token_set(b["content"])
            if not set_a or not set_b:
                continue
            jaccard = len(set_a & set_b) / len(set_a | set_b)
            if jaccard >= threshold:
                results.append(LintResult(
                    "duplicate_content", "warning", a["path"],
                    f"Near-duplicate ({jaccard:.0%} similar) with {b['path']}",
                ))
    return results


def check_category_balance(articles: list[dict]) -> list[LintResult]:
    """Flag categories that are very small or very large."""
    results = []
    cats = Counter(a["path"].split("/")[0] if "/" in a["path"] else "root" for a in articles)

    if len(cats) < 2:
        return results

    avg = sum(cats.values()) / len(cats)
    for cat, count in cats.items():
        if count == 1 and avg > 3:
            results.append(LintResult(
                "category_balance", "info", cat,
                f"Category '{cat}' has only 1 article (avg: {avg:.0f})",
            ))
        elif count > avg * 4 and count > 10:
            results.append(LintResult(
                "category_balance", "info", cat,
                f"Category '{cat}' has {count} articles — consider splitting",
            ))
    return results


_CHECKS = {
    "orphan_detection": check_orphans,
    "broken_links": check_broken_links,
    "missing_summaries": check_missing_summaries,
    "duplicate_content": check_duplicates,
    "category_balance": check_category_balance,
    # stale_sources handled separately (needs raw_dir)
}


def run_lint(cfg: dict, checks: list[str] | None = None) -> list[LintResult]:
    """Run all (or selected) lint checks."""
    wiki_dir = Path(cfg["paths"]["wiki"])
    raw_dir = Path(cfg["paths"]["raw"])
    threshold = cfg["lint"]["similarity_threshold"]

    if not wiki_dir.exists():
        print("Wiki not found. Run compile first.")
        return []

    articles = _scan_wiki(wiki_dir)
    if not articles:
        print("No articles found in wiki.")
        return []

    enabled = checks or cfg["lint"]["checks"]
    all_results: list[LintResult] = []

    for check_name in enabled:
        if check_name == "stale_sources":
            findings = check_stale_sources(raw_dir, articles)
        elif check_name == "duplicate_content":
            findings = check_duplicates(articles, threshold)
        elif check_name in _CHECKS:
            findings = _CHECKS[check_name](articles)
        else:
            print(f"  Unknown check: {check_name}")
            continue
        all_results.extend(findings)

    # Print results
    errors = [r for r in all_results if r.severity == "error"]
    warnings = [r for r in all_results if r.severity == "warning"]
    infos = [r for r in all_results if r.severity == "info"]

    print(f"\nLint: {len(articles)} articles checked")
    print(f"  {len(errors)} error(s), {len(warnings)} warning(s), {len(infos)} info(s)")
    print()

    for r in sorted(all_results, key=lambda x: (x.severity, x.path)):
        print(r)

    # Write report
    report = {
        "articles_checked": len(articles),
        "errors": len(errors),
        "warnings": len(warnings),
        "infos": len(infos),
        "findings": [r.to_dict() for r in all_results],
    }
    report_path = wiki_dir / "_lint_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport: {report_path}")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Lint the wiki")
    parser.add_argument("--check", nargs="+", help="Specific checks to run")
    args = parser.parse_args()

    cfg = load_config()
    run_lint(cfg, checks=args.check)


if __name__ == "__main__":
    main()
