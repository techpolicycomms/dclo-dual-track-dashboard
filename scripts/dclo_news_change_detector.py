#!/usr/bin/env python3
"""Detect new digital/tech news articles for DCLO news loop."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]


def load_state(state_path: Path) -> Dict[str, Any]:
    if not state_path.exists():
        return {"version": 1, "last_checked_utc": None, "article_fingerprint_sha256": None, "article_count": 0}
    return json.loads(state_path.read_text(encoding="utf-8"))


def save_state(state_path: Path, state: Dict[str, Any]) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")


def fingerprint_intake(intake_path: Path) -> Dict[str, Any]:
    if not intake_path.exists():
        return {"exists": False, "article_count": 0, "fingerprint_sha256": None}

    df = pd.read_csv(intake_path)
    if df.empty:
        return {"exists": True, "article_count": 0, "fingerprint_sha256": sha256_text("")}

    use_cols = [col for col in ["article_id", "url", "title", "published_at", "primary_dclo_domain"] if col in df.columns]
    subset = df[use_cols].sort_values(use_cols).astype(str)
    return {
        "exists": True,
        "article_count": int(len(df)),
        "mapped_count": int((df.get("primary_dclo_domain", pd.Series(dtype=str)) != "UNMAPPED").sum())
        if "primary_dclo_domain" in df.columns
        else 0,
        "fingerprint_sha256": sha256_text(subset.to_csv(index=False)),
    }


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def detect_changes(previous: Dict[str, Any], current: Dict[str, Any]) -> Tuple[bool, List[str]]:
    reasons: List[str] = []
    if not current.get("exists"):
        return False, ["no_intake_file"]

    prev_hash = previous.get("article_fingerprint_sha256")
    curr_hash = current.get("fingerprint_sha256")
    prev_count = int(previous.get("article_count", 0) or 0)
    curr_count = int(current.get("article_count", 0) or 0)

    if prev_hash is None:
        if curr_count > 0:
            reasons.append("initial_news_sync")
        return bool(reasons), reasons

    if prev_hash != curr_hash:
        reasons.append("article_fingerprint_changed")
    if curr_count > prev_count:
        reasons.append("new_articles_added")

    return bool(reasons), reasons


def write_github_output(output_path: Path, should_update: bool, reasons: List[str]) -> None:
    lines = [
        f"should_update={'true' if should_update else 'false'}",
        f"change_reasons={json.dumps(reasons)}",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect DCLO news intake changes.")
    parser.add_argument("--intake-file", default="data/gold/dclo_news_intake.csv")
    parser.add_argument("--state-file", default="data/gold/dclo_news_sync_state.json")
    parser.add_argument("--update-state", action="store_true")
    parser.add_argument("--github-output", default=None)
    args = parser.parse_args()

    intake_path = ROOT_DIR / args.intake_file
    state_path = ROOT_DIR / args.state_file
    state = load_state(state_path)
    current = fingerprint_intake(intake_path)
    should_update, reasons = detect_changes(state, current)

    print(f"[news-change-detector] should_update={should_update}")
    for reason in reasons:
        print(f"  - {reason}")

    if args.github_output:
        write_github_output(Path(args.github_output), should_update, reasons)

    if args.update_state:
        state["version"] = 1
        state["last_checked_utc"] = datetime.now(timezone.utc).isoformat()
        state["article_fingerprint_sha256"] = current.get("fingerprint_sha256")
        state["article_count"] = current.get("article_count", 0)
        state["mapped_count"] = current.get("mapped_count", 0)
        if should_update:
            state["last_update_utc"] = state["last_checked_utc"]
            state["last_update_reasons"] = reasons
        save_state(state_path, state)

    raise SystemExit(0)


if __name__ == "__main__":
    main()
