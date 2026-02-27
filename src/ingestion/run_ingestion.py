import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import yaml
from dotenv import load_dotenv

from data_gov_in_client import get_client_from_env


def read_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def filter_fields(records: List[Dict[str, Any]], fields: List[str]) -> List[Dict[str, Any]]:
    if not fields:
        return records
    filtered: List[Dict[str, Any]] = []
    for record in records:
        filtered.append({field: record.get(field) for field in fields})
    return filtered


def write_jsonl(path: Path, records: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True))
            handle.write("\n")


def run(config_path: str) -> None:
    load_dotenv()
    config = read_config(config_path)
    sources = config.get("sources", [])
    if not isinstance(sources, list) or not sources:
        raise ValueError("Config must include a non-empty 'sources' list")

    data_dir = Path(os.getenv("DATA_DIR", "./data"))
    raw_dir = data_dir / "raw"
    curated_dir = data_dir / "curated"
    ensure_dir(raw_dir)
    ensure_dir(curated_dir)

    client = get_client_from_env()
    run_ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    for source in sources:
        name = source.get("name")
        resource_id = source.get("resource_id")
        limit = int(source.get("limit", 100))
        data_format = source.get("format", "json")
        fields = source.get("fields", [])

        if not name or not resource_id:
            raise ValueError("Each source requires 'name' and 'resource_id'")

        records = client.fetch_records(
            resource_id=resource_id,
            offset=0,
            limit=limit,
            data_format=data_format,
        )
        records = filter_fields(records, fields)

        raw_path = raw_dir / f"{name}_{run_ts}.jsonl"
        latest_path = curated_dir / f"{name}_latest.jsonl"
        write_jsonl(raw_path, records)
        write_jsonl(latest_path, records)
        print(f"Ingested {len(records)} records for {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest datasets from data.gov.in")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
