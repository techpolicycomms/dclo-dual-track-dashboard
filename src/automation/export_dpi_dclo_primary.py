import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd

from common import load_yaml, read_jsonl


DPI_SCORE_FIELDS = [
    "usefulness_score_1to5",
    "ease_of_use_score_1to5",
    "interpretability_score_1to5",
    "trust_in_results_score_1to5",
    "policy_actionability_score_1to5",
    "would_recommend_1to5",
]

DCLO_DOMAIN_MAP: Dict[str, str] = {
    "acc_connectivity_reliability": "ACC",
    "skl_self_efficacy_digital_tasks": "SKL",
    "eco_digital_payments_use": "ECO",
    "srv_gov_service_completion_digital": "SRV",
    "agr_control_over_phone_account": "AGR",
    "out_perceived_welfare_change": "OUT",
}


def normalize_rows(rows: List[Dict[str, object]]) -> pd.DataFrame:
    records = [row for row in rows if "question_id" not in row]
    if not records:
        return pd.DataFrame()
    frame = pd.DataFrame(records)
    frame["response_id"] = frame.get("response_id", pd.Series(dtype=str)).astype(str)
    frame["campaign_mode"] = frame.get("campaign_mode", "prototype")
    frame["survey_mode"] = frame.get("survey_mode", "voice")
    frame["timestamp_utc"] = frame.get("timestamp_utc", "")
    return frame


def run(config_path: str) -> None:
    root = Path(__file__).resolve().parents[2]
    config = load_yaml(config_path)
    paths = config.get("paths", {})
    if not isinstance(paths, dict):
        raise ValueError("Invalid paths config")

    responses_path = root / str(paths.get("responses_jsonl", "data/primary/survey_responses.jsonl"))
    eligibility_path = root / str(paths.get("eligibility_csv", "data/gold/survey_incentive_eligibility.csv"))
    export_path = root / str(paths.get("dpi_dclo_export_csv", "data/gold/dpi_dclo_primary_export.csv"))
    dpi_long_path = root / str(paths.get("dpi_primary_long_csv", "data/gold/dpi_primary_validation_long.csv"))

    frame = normalize_rows(read_jsonl(responses_path))
    if frame.empty:
        frame = pd.DataFrame(columns=["response_id", "campaign_mode", "survey_mode", "timestamp_utc"])

    for field in DPI_SCORE_FIELDS + list(DCLO_DOMAIN_MAP.keys()):
        if field not in frame.columns:
            frame[field] = pd.NA
        frame[field] = pd.to_numeric(frame[field], errors="coerce")

    frame["dpi_validation_score"] = frame[DPI_SCORE_FIELDS].mean(axis=1, skipna=True)
    for field, domain in DCLO_DOMAIN_MAP.items():
        frame[f"{domain}_score"] = frame[field]
    domain_cols = [f"{domain}_score" for domain in ["ACC", "SKL", "ECO", "SRV", "AGR", "OUT"]]
    frame["DCLO_primary_score"] = frame[domain_cols].mean(axis=1, skipna=True)

    if eligibility_path.exists():
        eligibility = pd.read_csv(eligibility_path)
        frame = frame.merge(
            eligibility[["response_id", "completion_status", "incentive_eligible", "incentive_amount_inr"]],
            on="response_id",
            how="left",
        )
    else:
        frame["completion_status"] = pd.NA
        frame["incentive_eligible"] = pd.NA
        frame["incentive_amount_inr"] = pd.NA

    export_cols = [
        "response_id",
        "timestamp_utc",
        "campaign_mode",
        "survey_mode",
        "country",
        "participant_role",
        "organization_type",
        "dpi_validation_score",
        "ACC_score",
        "SKL_score",
        "ECO_score",
        "SRV_score",
        "AGR_score",
        "OUT_score",
        "DCLO_primary_score",
        "completion_status",
        "incentive_eligible",
        "incentive_amount_inr",
    ]
    for col in export_cols:
        if col not in frame.columns:
            frame[col] = pd.NA
    frame[export_cols].to_csv(export_path, index=False)

    long_rows: List[Dict[str, object]] = []
    score_fields = DPI_SCORE_FIELDS + list(DCLO_DOMAIN_MAP.keys())
    for _, row in frame.iterrows():
        for field in score_fields:
            long_rows.append(
                {
                    "response_id": row.get("response_id"),
                    "timestamp_utc": row.get("timestamp_utc"),
                    "campaign_mode": row.get("campaign_mode"),
                    "survey_mode": row.get("survey_mode"),
                    "field_name": field,
                    "field_value": row.get(field),
                }
            )
    pd.DataFrame(long_rows).to_csv(dpi_long_path, index=False)

    print(f"Wrote DPI/DCLO primary export: {export_path}")
    print(f"Wrote DPI primary long export: {dpi_long_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export schema-aligned primary survey outputs for DPI/DCLO pipelines")
    parser.add_argument("--config", default="config/survey_automation.yml")
    args = parser.parse_args()
    run(config_path=args.config)


if __name__ == "__main__":
    main()
