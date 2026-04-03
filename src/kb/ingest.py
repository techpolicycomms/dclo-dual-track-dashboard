"""Ingest raw source documents into the knowledge base.

Copies files into kb/raw/ with metadata sidecar files (.meta.yml),
preserving originals and recording provenance.

Usage:
    python -m src.kb.ingest path/to/document.md --source "URL or description"
    python -m src.kb.ingest path/to/folder/ --tag economics --tag india
"""

import argparse
import hashlib
import shutil
from datetime import datetime, timezone
from pathlib import Path

import yaml

from .config import load_config


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_meta(dest: Path, meta: dict) -> Path:
    meta_path = dest.with_suffix(dest.suffix + ".meta.yml")
    with open(meta_path, "w") as f:
        yaml.dump(meta, f, default_flow_style=False, sort_keys=False)
    return meta_path


def ingest_file(
    src: Path,
    raw_dir: Path,
    *,
    source: str = "",
    tags: list[str] | None = None,
    category: str = "",
) -> dict:
    """Ingest a single file into kb/raw/. Returns metadata dict."""
    if not src.exists():
        raise FileNotFoundError(f"Source not found: {src}")

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dest_name = f"{src.stem}_{ts}{src.suffix}"
    dest = raw_dir / dest_name

    shutil.copy2(src, dest)

    meta = {
        "original_name": src.name,
        "ingested_at": ts,
        "source": source or str(src.resolve()),
        "sha256": _sha256(dest),
        "size_bytes": dest.stat().st_size,
        "tags": tags or [],
        "category": category,
        "compiled": False,
    }
    meta_path = _write_meta(dest, meta)

    print(f"  Ingested: {src.name} -> {dest.name}")
    print(f"  Metadata: {meta_path.name}")
    return meta


def ingest_directory(
    src_dir: Path,
    raw_dir: Path,
    *,
    extensions: list[str] | None = None,
    source: str = "",
    tags: list[str] | None = None,
    category: str = "",
    recursive: bool = True,
) -> list[dict]:
    """Ingest all matching files from a directory."""
    results = []
    pattern = "**/*" if recursive else "*"
    for path in sorted(src_dir.glob(pattern)):
        if not path.is_file():
            continue
        if extensions and path.suffix.lower() not in extensions:
            continue
        if path.suffix == ".meta.yml":
            continue
        meta = ingest_file(
            path, raw_dir, source=source, tags=tags, category=category
        )
        results.append(meta)
    return results


def main():
    cfg = load_config()
    raw_dir = Path(cfg["paths"]["raw"])
    raw_dir.mkdir(parents=True, exist_ok=True)
    extensions = cfg["ingest"]["supported_extensions"]

    parser = argparse.ArgumentParser(description="Ingest documents into KB raw/")
    parser.add_argument("path", help="File or directory to ingest")
    parser.add_argument("--source", default="", help="Source URL or description")
    parser.add_argument("--tag", action="append", default=[], help="Tag(s) for the document")
    parser.add_argument("--category", default="", help="Category assignment")
    parser.add_argument("--no-recursive", action="store_true", help="Don't recurse into subdirs")
    args = parser.parse_args()

    src = Path(args.path)
    print(f"Ingesting into: {raw_dir}")

    if src.is_file():
        ingest_file(src, raw_dir, source=args.source, tags=args.tag, category=args.category)
    elif src.is_dir():
        results = ingest_directory(
            src, raw_dir,
            extensions=extensions,
            source=args.source,
            tags=args.tag,
            category=args.category,
            recursive=not args.no_recursive,
        )
        print(f"\nIngested {len(results)} file(s).")
    else:
        print(f"Error: {src} not found.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
