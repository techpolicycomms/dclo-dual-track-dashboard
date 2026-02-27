import json
import os
from pathlib import Path
from typing import Dict, List

import pandas as pd


def read_jsonl(path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def safe_to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def build_gold_for_file(input_path: Path, output_dir: Path) -> None:
    rows = read_jsonl(input_path)
    if not rows:
        return
    df = pd.DataFrame(rows)
    dataset_name = input_path.name.replace("_latest.jsonl", "")

    # Generic aggregate for Power BI: row counts by first non-metric dimensions.
    value_cols = []
    for col in df.columns:
        converted = safe_to_numeric(df[col])
        if converted.notna().sum() > 0:
            df[col] = converted
            value_cols.append(col)

    dim_cols = [c for c in df.columns if c not in value_cols]
    if not dim_cols:
        dim_cols = [df.columns[0]]

    base_group = dim_cols[: min(2, len(dim_cols))]
    agg = df.groupby(base_group, dropna=False).size().reset_index(name="record_count")
    out_path = output_dir / f"{dataset_name}_gold.csv"
    agg.to_csv(out_path, index=False)


def main() -> None:
    data_dir = Path(os.getenv("DATA_DIR", "./data"))
    curated_dir = data_dir / "curated"
    gold_dir = data_dir / "gold"
    gold_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(curated_dir.glob("*_latest.jsonl"))
    if not files:
        raise ValueError(f"No curated files found in {curated_dir}")

    for file_path in files:
        build_gold_for_file(file_path, gold_dir)
    print(f"Gold tables built in {gold_dir}")


if __name__ == "__main__":
    main()
