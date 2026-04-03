"""TF-IDF search engine over the wiki.

Builds an inverted index from wiki articles, supports keyword and
phrase search with ranked results. Designed to be used both as a
CLI tool and as a library callable by LLM agents.

Usage:
    python -m src.kb.search "digital capability index"
    python -m src.kb.search "causal panel" --top 5
    python -m src.kb.search "DCLO" --json
"""

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

from .config import load_config


def _tokenize(text: str) -> list[str]:
    """Lowercase tokenization with basic normalization."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    tokens = text.split()
    # Remove very short tokens and stop words
    stop = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been",
        "being", "have", "has", "had", "do", "does", "did", "will",
        "would", "could", "should", "may", "might", "shall", "can",
        "to", "of", "in", "for", "on", "with", "at", "by", "from",
        "as", "into", "through", "during", "before", "after", "and",
        "but", "or", "nor", "not", "so", "yet", "both", "either",
        "neither", "each", "every", "all", "any", "few", "more",
        "most", "other", "some", "such", "no", "only", "own", "same",
        "than", "too", "very", "just", "because", "this", "that",
        "these", "those", "it", "its", "he", "she", "they", "them",
        "we", "us", "you", "i", "me", "my", "your", "his", "her",
        "our", "their", "what", "which", "who", "whom", "how",
        "when", "where", "why", "if", "then", "else", "about",
    }
    return [t for t in tokens if len(t) > 1 and t not in stop]


class WikiIndex:
    """TF-IDF inverted index over wiki articles."""

    def __init__(self, wiki_dir: Path, cfg: dict):
        self.wiki_dir = wiki_dir
        self.cfg = cfg
        self.documents: dict[str, dict] = {}   # path -> {title, content, tokens, ...}
        self.idf: dict[str, float] = {}
        self.inverted: dict[str, set[str]] = defaultdict(set)
        self._build()

    def _build(self):
        """Build the index from wiki .md files."""
        for md_path in self.wiki_dir.rglob("*.md"):
            if md_path.name.startswith("_"):
                continue
            text = md_path.read_text(errors="replace")
            rel = str(md_path.relative_to(self.wiki_dir))

            # Parse title and summary
            title = ""
            summary = ""
            for line in text.splitlines():
                if line.startswith("# ") and not title:
                    title = line[2:].strip()
                elif line.startswith("> **Summary**:") and not summary:
                    summary = line.split(":", 1)[1].strip()

            title = title or md_path.stem.replace("-", " ").title()
            tokens = _tokenize(text)
            title_tokens = _tokenize(title)
            summary_tokens = _tokenize(summary)

            self.documents[rel] = {
                "title": title,
                "summary": summary,
                "content": text,
                "tokens": tokens,
                "title_tokens": title_tokens,
                "summary_tokens": summary_tokens,
                "tf": Counter(tokens),
                "path": rel,
                "abs_path": str(md_path),
            }

            for token in set(tokens):
                self.inverted[token].add(rel)

        # Compute IDF
        n = len(self.documents)
        if n == 0:
            return
        for term, doc_set in self.inverted.items():
            self.idf[term] = math.log(n / len(doc_set))

    def search(self, query: str, top_k: int = 20) -> list[dict]:
        """Search the index. Returns ranked results."""
        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        boost_title = self.cfg["search"]["boost_title"]
        boost_summary = self.cfg["search"]["boost_summary"]
        snippet_len = self.cfg["search"]["snippet_length"]

        scores: dict[str, float] = defaultdict(float)

        for token in query_tokens:
            if token not in self.inverted:
                continue
            idf = self.idf.get(token, 0)
            for doc_path in self.inverted[token]:
                doc = self.documents[doc_path]
                tf = doc["tf"][token]
                # Log-normalized TF
                tf_score = (1 + math.log(tf)) if tf > 0 else 0
                base = tf_score * idf

                # Boost for title/summary matches
                if token in doc["title_tokens"]:
                    base *= boost_title
                if token in doc["summary_tokens"]:
                    base *= boost_summary

                scores[doc_path] += base

        # Rank and return
        ranked = sorted(scores.items(), key=lambda x: -x[1])[:top_k]
        results = []
        for doc_path, score in ranked:
            doc = self.documents[doc_path]
            snippet = self._extract_snippet(doc["content"], query_tokens, snippet_len)
            results.append({
                "path": doc_path,
                "title": doc["title"],
                "summary": doc["summary"],
                "score": round(score, 3),
                "snippet": snippet,
            })
        return results

    def _extract_snippet(self, content: str, query_tokens: list[str], length: int) -> str:
        """Extract a relevant snippet around the first query match."""
        content_lower = content.lower()
        best_pos = len(content)
        for token in query_tokens:
            pos = content_lower.find(token)
            if 0 <= pos < best_pos:
                best_pos = pos

        start = max(0, best_pos - length // 4)
        end = start + length
        snippet = content[start:end].replace("\n", " ").strip()
        if start > 0:
            snippet = "..." + snippet
        if end < len(content):
            snippet = snippet + "..."
        return snippet


def search_wiki(query: str, cfg: dict, top_k: int = 0, as_json: bool = False) -> list[dict]:
    """High-level search function."""
    wiki_dir = Path(cfg["paths"]["wiki"])
    if not wiki_dir.exists():
        print("Wiki directory not found. Run compile first.")
        return []

    idx = WikiIndex(wiki_dir, cfg)
    k = top_k or cfg["search"]["max_results"]
    results = idx.search(query, top_k=k)

    if as_json:
        print(json.dumps(results, indent=2))
    else:
        if not results:
            print(f"No results for: {query}")
        else:
            print(f"\nSearch: \"{query}\" ({len(results)} results)\n")
            for i, r in enumerate(results, 1):
                print(f"  {i}. [{r['score']:.2f}] {r['title']}")
                print(f"     {r['path']}")
                if r["summary"]:
                    print(f"     {r['summary'][:100]}")
                print(f"     {r['snippet']}")
                print()

    return results


def main():
    parser = argparse.ArgumentParser(description="Search the wiki")
    parser.add_argument("query", help="Search query")
    parser.add_argument("--top", type=int, default=0, help="Max results")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    cfg = load_config()
    search_wiki(args.query, cfg, top_k=args.top, as_json=args.json)


if __name__ == "__main__":
    main()
