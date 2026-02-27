import json
import os
from pathlib import Path
from typing import Dict, List


def read_jsonl(path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def validate_dataset(path: Path) -> List[str]:
    errors: List[str] = []
    rows = read_jsonl(path)
    if not rows:
        errors.append(f"{path.name}: no rows")
        return errors

    # Basic schema quality gate: column set should be stable in one file.
    expected_keys = set(rows[0].keys())
    for idx, row in enumerate(rows[1:], start=2):
        row_keys = set(row.keys())
        if row_keys != expected_keys:
            errors.append(f"{path.name}: row {idx} has inconsistent keys")

    if len(rows) < 5:
        errors.append(f"{path.name}: low row count ({len(rows)})")
    return errors


def main() -> None:
    data_dir = Path(os.getenv("DATA_DIR", "./data"))
    curated_dir = data_dir / "curated"
    files = sorted(curated_dir.glob("*_latest.jsonl"))
    if not files:
        raise ValueError(f"No curated files found in {curated_dir}")

    all_errors: List[str] = []
    for file_path in files:
        errors = validate_dataset(file_path)
        all_errors.extend(errors)

    if all_errors:
        print("Validation failed:")
        for error in all_errors:
            print(f"- {error}")
        raise SystemExit(1)

    print(f"Validation passed for {len(files)} dataset(s)")


if __name__ == "__main__":
    main()
