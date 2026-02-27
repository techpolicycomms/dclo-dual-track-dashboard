import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import pandas as pd
import requests
import yaml


def read_config(config_path: str) -> Dict[str, object]:
    with open(config_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_from_local(path: str, data_format: str, sheet_name: str = "Sheet1") -> pd.DataFrame:
    if data_format == "csv":
        return pd.read_csv(path)
    if data_format in {"xlsx", "xls"}:
        return pd.read_excel(path, sheet_name=sheet_name)
    raise ValueError(f"Unsupported format for local file: {data_format}")


def download_to_raw(url: str, destination: Path) -> None:
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    destination.write_bytes(response.content)


def load_from_downloaded_file(path: Path, data_format: str, sheet_name: str = "Sheet1") -> pd.DataFrame:
    if data_format == "csv":
        return pd.read_csv(path)
    if data_format in {"xlsx", "xls"}:
        return pd.read_excel(path, sheet_name=sheet_name)
    raise ValueError(f"Unsupported format for downloaded file: {data_format}")


def apply_optional_transforms(df: pd.DataFrame, dataset_cfg: Dict[str, object]) -> pd.DataFrame:
    field_select = dataset_cfg.get("field_select", [])
    if field_select:
        available = [col for col in field_select if col in df.columns]
        df = df[available].copy()

    rename_map = dataset_cfg.get("rename_map", {})
    if rename_map:
        df = df.rename(columns=rename_map)

    return df


def ingest_one_dataset(dataset_cfg: Dict[str, object], raw_dir: Path, curated_dir: Path, run_ts: str) -> None:
    name = dataset_cfg.get("name")
    source_type = dataset_cfg.get("source_type")
    data_format = str(dataset_cfg.get("format", "csv")).lower()
    sheet_name = str(dataset_cfg.get("sheet_name", "Sheet1"))
    if not name or not source_type:
        raise ValueError("Each dataset requires 'name' and 'source_type'")

    if source_type == "local_file":
        local_path = dataset_cfg.get("path")
        if not local_path:
            raise ValueError(f"{name}: missing 'path' for local_file source")
        df = load_from_local(local_path, data_format, sheet_name=sheet_name)

    elif source_type == "download_url":
        url = dataset_cfg.get("url")
        if not url:
            raise ValueError(f"{name}: missing 'url' for download_url source")
        raw_download_path = raw_dir / f"{name}_{run_ts}.{data_format}"
        download_to_raw(url, raw_download_path)
        df = load_from_downloaded_file(raw_download_path, data_format, sheet_name=sheet_name)
    else:
        raise ValueError(f"{name}: unsupported source_type '{source_type}'")

    df = apply_optional_transforms(df, dataset_cfg)
    df["source_name"] = name
    df["ingested_at_utc"] = run_ts

    raw_csv_path = raw_dir / f"{name}_{run_ts}.csv"
    latest_csv_path = curated_dir / f"{name}_latest.csv"
    df.to_csv(raw_csv_path, index=False)
    df.to_csv(latest_csv_path, index=False)
    print(f"{name}: {len(df)} rows -> {latest_csv_path}")


def run(config_path: str) -> None:
    config = read_config(config_path)
    datasets: List[Dict[str, object]] = config.get("datasets", [])
    datasets = [dataset for dataset in datasets if bool(dataset.get("enabled", True))]
    if not datasets:
        raise ValueError("Config must contain a non-empty 'datasets' list")

    output_cfg = config.get("output", {})
    data_dir = Path(output_cfg.get("data_dir", "./data"))
    raw_dir = data_dir / "raw"
    curated_dir = data_dir / "curated"
    ensure_dir(raw_dir)
    ensure_dir(curated_dir)

    run_ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    for dataset_cfg in datasets:
        ingest_one_dataset(dataset_cfg, raw_dir, curated_dir, run_ts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest RBI DBIE datasets from URLs or local exports")
    parser.add_argument("--config", required=True, help="Path to RBI DBIE ingestion YAML config")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
