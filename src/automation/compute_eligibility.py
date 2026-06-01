import argparse
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

from common import get_env, load_yaml, read_jsonl, stable_phone_hash, write_csv


QUESTION_TO_FIELD: Dict[str, str] = {
    "consent": "consent",
    "participant_role": "participant_role",
    "organization_type": "organization_type",
    "country": "country",
    "task_completed": "task_completed",
    "usefulness_score_1to5": "usefulness_score_1to5",
    "ease_of_use_score_1to5": "ease_of_use_score_1to5",
    "interpretability_score_1to5": "interpretability_score_1to5",
    "trust_in_results_score_1to5": "trust_in_results_score_1to5",
    "policy_actionability_score_1to5": "policy_actionability_score_1to5",
    "would_recommend_1to5": "would_recommend_1to5",
    "acc_connectivity_reliability": "acc_connectivity_reliability",
    "skl_self_efficacy_digital_tasks": "skl_self_efficacy_digital_tasks",
    "eco_digital_payments_use": "eco_digital_payments_use",
    "srv_gov_service_completion_digital": "srv_gov_service_completion_digital",
    "agr_control_over_phone_account": "agr_control_over_phone_account",
    "out_perceived_welfare_change": "out_perceived_welfare_change",
}


def _is_blank(value: object) -> bool:
    return value is None or str(value).strip() == ""


def collapse_question_rows(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[str, Dict[str, object]] = defaultdict(dict)
    for row in rows:
        if "question_id" not in row:
            continue
        call_sid = str(row.get("call_sid", "")).strip()
        question_id = str(row.get("question_id", "")).strip()
        value = row.get("response_value")
        if not call_sid or question_id not in QUESTION_TO_FIELD:
            continue
        grouped[call_sid]["response_id"] = call_sid
        grouped[call_sid]["survey_mode"] = "voice"
        grouped[call_sid]["timestamp_utc"] = row.get("timestamp_utc")
        grouped[call_sid][QUESTION_TO_FIELD[question_id]] = value
    return list(grouped.values())


def normalize_responses(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    full_rows = [row for row in rows if "question_id" not in row]
    collapsed = collapse_question_rows(rows)
    return full_rows + collapsed


def classify_row(
    row: Dict[str, object],
    required_fields: List[str],
    min_duration: int,
    max_duration: int,
    seen_phone_hashes: Dict[str, int],
    hash_salt: str,
) -> Tuple[str, bool, str]:
    missing = [field for field in required_fields if _is_blank(row.get(field))]
    consent_value = str(row.get("consent", "")).strip().lower()
    duration = int(float(row.get("interview_duration_sec", 0) or 0))
    breakoff = str(row.get("breakoff_flag", "false")).strip().lower() in {"true", "1", "yes"}

    phone = str(row.get("phone", "")).strip()
    phone_hash = stable_phone_hash(phone, hash_salt) if phone else str(row.get("phone_hash", "")).strip()
    duplicate = False
    if phone_hash:
        seen_phone_hashes[phone_hash] = seen_phone_hashes.get(phone_hash, 0) + 1
        duplicate = seen_phone_hashes[phone_hash] > 1

    invalid_duration = bool(duration and (duration < min_duration or duration > max_duration))
    if consent_value not in {"yes", "1"}:
        return "invalid", False, "consent_not_granted"
    if duplicate:
        return "invalid", False, "duplicate_response"
    if breakoff:
        return "partial", False, "breakoff_detected"
    if missing:
        return "partial", False, "missing_required_fields"
    if invalid_duration:
        return "invalid", False, "duration_outlier"
    return "complete", True, "eligible_full_completion"


def run(config_path: str) -> None:
    root = Path(__file__).resolve().parents[2]
    config = load_yaml(config_path)
    paths = config.get("paths", {})
    survey_cfg = config.get("survey", {})
    incentive_cfg = config.get("incentive", {})
    if not isinstance(paths, dict) or not isinstance(survey_cfg, dict) or not isinstance(incentive_cfg, dict):
        raise ValueError("Invalid config sections for eligibility pipeline")

    responses_path = root / str(paths.get("responses_jsonl", "data/primary/survey_responses.jsonl"))
    eligibility_path = root / str(paths.get("eligibility_csv", "data/gold/survey_incentive_eligibility.csv"))
    payout_path = root / str(paths.get("payout_review_csv", "data/gold/survey_payout_review.csv"))

    rows = normalize_responses(read_jsonl(responses_path))
    required_fields = [str(field) for field in survey_cfg.get("required_fields", [])]
    duration_cfg = survey_cfg.get("expected_duration_sec", {})
    if not isinstance(duration_cfg, dict):
        duration_cfg = {}
    min_duration = int(duration_cfg.get("min", 120))
    max_duration = int(duration_cfg.get("max", 1800))
    hash_salt = get_env("SURVEY_PHONE_HASH_SALT", default="local-dev")
    amount_inr = int(incentive_cfg.get("amount_inr", 200))

    seen_phone_hashes: Dict[str, int] = {}
    scored: List[Dict[str, object]] = []
    for row in rows:
        status, eligible, reason = classify_row(
            row=row,
            required_fields=required_fields,
            min_duration=min_duration,
            max_duration=max_duration,
            seen_phone_hashes=seen_phone_hashes,
            hash_salt=hash_salt,
        )
        phone = str(row.get("phone", "")).strip()
        phone_hash = stable_phone_hash(phone, hash_salt) if phone else str(row.get("phone_hash", "")).strip()
        scored.append(
            {
                "response_id": row.get("response_id", ""),
                "timestamp_utc": row.get("timestamp_utc", datetime.now(timezone.utc).isoformat()),
                "campaign_mode": row.get("campaign_mode", "prototype"),
                "survey_mode": row.get("survey_mode", "voice"),
                "phone_hash": phone_hash,
                "completion_status": status,
                "incentive_eligible": str(eligible).lower(),
                "eligibility_reason": reason,
                "incentive_amount_inr": amount_inr if eligible else 0,
            }
        )

    eligibility_fields = [
        "response_id",
        "timestamp_utc",
        "campaign_mode",
        "survey_mode",
        "phone_hash",
        "completion_status",
        "incentive_eligible",
        "eligibility_reason",
        "incentive_amount_inr",
    ]
    write_csv(eligibility_path, scored, eligibility_fields)

    payout_rows = [row for row in scored if str(row["incentive_eligible"]) == "true"]
    payout_fields = [
        "response_id",
        "timestamp_utc",
        "phone_hash",
        "incentive_amount_inr",
        "eligibility_reason",
    ]
    write_csv(payout_path, payout_rows, payout_fields)
    print(f"Wrote eligibility file: {eligibility_path}")
    print(f"Wrote payout review file: {payout_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute completion and incentive eligibility")
    parser.add_argument("--config", default="config/survey_automation.yml", help="Path to survey automation config")
    args = parser.parse_args()
    run(config_path=args.config)


if __name__ == "__main__":
    main()
