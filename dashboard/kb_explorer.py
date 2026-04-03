"""Knowledge Base Explorer — Streamlit web UI.

Browse, search, and inspect the LLM-compiled wiki from the browser.

Usage:
    streamlit run dashboard/kb_explorer.py
"""

import json
import re
from pathlib import Path

import streamlit as st

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
WIKI_DIR = ROOT / "kb" / "wiki"
STATS_PATH = WIKI_DIR / "_stats.json"
LINT_PATH = WIKI_DIR / "_lint_report.json"
INDEX_PATH = WIKI_DIR / "_index.md"
CATALOG_PATH = WIKI_DIR / "_catalog.md"

st.set_page_config(page_title="Knowledge Base Explorer", page_icon="KB", layout="wide")


# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------
@st.cache_data
def scan_articles() -> list[dict]:
    """Collect metadata from every wiki article."""
    articles = []
    if not WIKI_DIR.exists():
        return articles
    for md_path in sorted(WIKI_DIR.rglob("*.md")):
        if md_path.name.startswith("_"):
            continue
        text = md_path.read_text(errors="replace")
        title = ""
        summary = ""
        tags: list[str] = []
        for line in text.splitlines():
            if line.startswith("# ") and not title:
                title = line[2:].strip()
            elif line.startswith("> **Summary**:") and not summary:
                summary = line.split(":", 1)[1].strip()
            elif line.startswith("- **Tags**:"):
                tag_str = line.split(":", 1)[1].strip()
                if tag_str and tag_str != "none":
                    tags = [t.strip() for t in tag_str.split(",")]

        links = re.findall(r"\[\[(.+?)\]\]", text)
        rel = str(md_path.relative_to(WIKI_DIR))
        category = md_path.parent.name if md_path.parent != WIKI_DIR else "root"

        articles.append(
            {
                "path": rel,
                "abs_path": str(md_path),
                "title": title or md_path.stem.replace("-", " ").title(),
                "summary": summary,
                "category": category,
                "tags": tags,
                "links": links,
                "word_count": len(text.split()),
                "content": text,
            }
        )
    return articles


@st.cache_data
def load_json(path: Path) -> dict | list | None:
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Search (simple TF-IDF inline, avoids importing src.kb at runtime)
# ---------------------------------------------------------------------------
import math
from collections import Counter, defaultdict

_STOP = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "as", "into", "through", "during",
    "before", "after", "and", "but", "or", "not", "so", "yet", "this",
    "that", "these", "those", "it", "its", "he", "she", "they", "them",
    "we", "us", "you", "me", "my", "your", "his", "her", "our", "their",
    "what", "which", "who", "how", "when", "where", "why", "if", "about",
}


def _tokenize(text: str) -> list[str]:
    text = re.sub(r"[^\w\s]", " ", text.lower())
    return [t for t in text.split() if len(t) > 1 and t not in _STOP]


def search_articles(articles: list[dict], query: str, top_k: int = 20) -> list[dict]:
    """Lightweight TF-IDF search over pre-loaded articles."""
    qtokens = _tokenize(query)
    if not qtokens:
        return []

    # Build IDF
    n = len(articles)
    doc_tokens = []
    df_counter: Counter = Counter()
    for a in articles:
        toks = set(_tokenize(a["content"]))
        doc_tokens.append(toks)
        df_counter.update(toks)

    idf = {t: math.log(n / c) for t, c in df_counter.items()}

    scored = []
    for i, a in enumerate(articles):
        tf = Counter(_tokenize(a["content"]))
        title_toks = set(_tokenize(a["title"]))
        score = 0.0
        for qt in qtokens:
            if qt not in tf:
                continue
            base = (1 + math.log(tf[qt])) * idf.get(qt, 0)
            if qt in title_toks:
                base *= 2.0
            score += base
        if score > 0:
            scored.append((score, i))

    scored.sort(key=lambda x: -x[0])
    results = []
    for score, idx in scored[:top_k]:
        a = articles[idx]
        results.append({**a, "score": round(score, 2)})
    return results


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
def main():
    articles = scan_articles()

    if not articles:
        st.warning(
            "No wiki articles found. Run `python -m src.kb seed` to bootstrap the knowledge base."
        )
        return

    st.title("Knowledge Base Explorer")

    # Sidebar: stats + navigation
    stats = load_json(STATS_PATH)
    with st.sidebar:
        st.header("KB Overview")
        if stats:
            c1, c2 = st.columns(2)
            c1.metric("Articles", stats.get("wiki_articles", 0))
            c2.metric("Words", f"{stats.get('total_words', 0):,}")
            c3, c4 = st.columns(2)
            c3.metric("Raw Docs", stats.get("raw_documents", 0))
            c4.metric("Compiled", stats.get("compiled", 0))
            c5, c6 = st.columns(2)
            c5.metric("Orphans", stats.get("orphans", 0))
            c6.metric("Broken Links", stats.get("broken_links", 0))
        else:
            st.caption("Run `python -m src.kb index --stats` to generate stats.")

        st.divider()
        st.header("Categories")
        categories = sorted({a["category"] for a in articles})
        selected_cat = st.radio(
            "Filter by category", ["All"] + categories, label_visibility="collapsed"
        )

    # Tabs
    tab_search, tab_browse, tab_lint, tab_index = st.tabs(
        ["Search", "Browse", "Health", "Index"]
    )

    # ---- SEARCH TAB ----
    with tab_search:
        query = st.text_input(
            "Search the knowledge base",
            placeholder="e.g. causal panel estimation, DCLO methodology...",
        )
        if query:
            results = search_articles(articles, query)
            if not results:
                st.info("No results found.")
            else:
                st.caption(f"{len(results)} result(s)")
                for r in results:
                    with st.expander(
                        f"**{r['title']}** — score {r['score']} | {r['category']} | {r['word_count']:,} words"
                    ):
                        if r["summary"]:
                            st.markdown(f"> {r['summary']}")
                        st.caption(f"Path: `{r['path']}` | Tags: {', '.join(r['tags']) or 'none'}")
                        # Show truncated content
                        st.markdown(r["content"][:3000])
                        if len(r["content"]) > 3000:
                            st.caption("... content truncated")

    # ---- BROWSE TAB ----
    with tab_browse:
        filtered = (
            articles
            if selected_cat == "All"
            else [a for a in articles if a["category"] == selected_cat]
        )

        col_list, col_detail = st.columns([1, 2])

        with col_list:
            st.subheader(f"Articles ({len(filtered)})")
            titles = [a["title"] for a in filtered]
            if not titles:
                st.info("No articles in this category.")
                selected_title = None
            else:
                selected_title = st.radio(
                    "Select article", titles, label_visibility="collapsed"
                )

        with col_detail:
            if selected_title:
                article = next(a for a in filtered if a["title"] == selected_title)
                st.subheader(article["title"])
                st.caption(
                    f"Category: **{article['category']}** | "
                    f"Words: **{article['word_count']:,}** | "
                    f"Tags: {', '.join(article['tags']) or 'none'} | "
                    f"Links: {len(article['links'])}"
                )
                st.divider()
                st.markdown(article["content"])

    # ---- HEALTH TAB ----
    with tab_lint:
        lint_data = load_json(LINT_PATH)
        if not lint_data:
            st.info("No lint report found. Run `python -m src.kb lint` first.")
        else:
            st.subheader("Wiki Health Report")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Articles Checked", lint_data.get("articles_checked", 0))
            c2.metric("Errors", lint_data.get("errors", 0))
            c3.metric("Warnings", lint_data.get("warnings", 0))
            c4.metric("Info", lint_data.get("infos", 0))

            findings = lint_data.get("findings", [])
            if findings:
                severity_filter = st.multiselect(
                    "Filter by severity",
                    ["error", "warning", "info"],
                    default=["error", "warning"],
                )
                check_names = sorted({f["check"] for f in findings})
                check_filter = st.multiselect(
                    "Filter by check",
                    check_names,
                    default=check_names,
                )

                filtered_findings = [
                    f
                    for f in findings
                    if f["severity"] in severity_filter and f["check"] in check_filter
                ]
                st.caption(f"Showing {len(filtered_findings)} of {len(findings)} findings")

                for f in filtered_findings[:100]:
                    icon = {"error": "E", "warning": "W", "info": "I"}.get(
                        f["severity"], "?"
                    )
                    st.markdown(
                        f"`[{icon}]` **{f['path']}** — {f['message']}"
                    )
            else:
                st.success("No issues found!")

    # ---- INDEX TAB ----
    with tab_index:
        col_idx, col_cat = st.columns(2)
        with col_idx:
            st.subheader("Master Index")
            if INDEX_PATH.exists():
                st.markdown(INDEX_PATH.read_text(errors="replace"))
            else:
                st.info("Index not built yet.")
        with col_cat:
            st.subheader("Source Catalog")
            if CATALOG_PATH.exists():
                st.markdown(CATALOG_PATH.read_text(errors="replace"))
            else:
                st.info("Catalog not built yet.")


if __name__ == "__main__":
    main()
