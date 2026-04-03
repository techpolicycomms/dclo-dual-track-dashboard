"""Unified CLI entry point for the LLM Knowledge Base toolkit.

Usage:
    python -m src.kb ingest <path> [--source ...] [--tag ...] [--category ...]
    python -m src.kb compile [--full]
    python -m src.kb index [--catalog] [--stats]
    python -m src.kb search <query> [--top N] [--json]
    python -m src.kb lint [--check ...]
    python -m src.kb output report --topic "..."
    python -m src.kb output slides --topic "..."
    python -m src.kb output chart
    python -m src.kb seed   # Seed wiki from existing project docs
"""

import sys


def main():
    if len(sys.argv) < 2:
        _print_help()
        return

    command = sys.argv[1]
    # Shift argv so sub-modules see their own args
    sys.argv = [f"kb {command}"] + sys.argv[2:]

    if command == "ingest":
        from .ingest import main as run
        run()
    elif command == "compile":
        from .compile import main as run
        run()
    elif command == "index":
        from .index import main as run
        run()
    elif command == "search":
        from .search import main as run
        run()
    elif command == "lint":
        from .lint import main as run
        run()
    elif command == "output":
        from .output import main as run
        run()
    elif command == "seed":
        from .seed import main as run
        run()
    elif command in ("help", "--help", "-h"):
        _print_help()
    else:
        print(f"Unknown command: {command}")
        _print_help()
        sys.exit(1)


def _print_help():
    print("""
LLM Knowledge Base Toolkit
===========================

Commands:
  ingest   Ingest raw documents into kb/raw/
  compile  Compile raw docs into interlinked wiki articles
  index    Rebuild master index and catalog
  search   Search the wiki (TF-IDF)
  lint     Run health checks on the wiki
  output   Generate reports, slides, or charts
  seed     Seed the wiki from existing project docs/data

Examples:
  python -m src.kb ingest docs/ --tag dclo --category methodology
  python -m src.kb compile --full
  python -m src.kb search "causal panel estimation"
  python -m src.kb lint
  python -m src.kb output report --topic "digital capability"
  python -m src.kb output slides --topic "DCLO index"
  python -m src.kb seed
""".strip())
