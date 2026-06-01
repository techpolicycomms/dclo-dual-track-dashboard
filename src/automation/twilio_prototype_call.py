import argparse
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

from common import append_jsonl, ensure_dir, get_env, load_yaml, stable_phone_hash

try:
    from twilio.rest import Client  # type: ignore
except ImportError:  # pragma: no cover
    Client = None


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def recipients_for_mode(config: Dict[str, object], mode: str, root: Path) -> List[str]:
    survey_cfg = config.get("survey", {})
    if not isinstance(survey_cfg, dict):
        return []

    if mode == "prototype":
        recipients = survey_cfg.get("prototype_allowlist", [])
        return [str(item) for item in recipients] if isinstance(recipients, list) else []

    queue_path = root / "data" / "primary" / "outreach_queue.csv"
    if not queue_path.exists():
        return []

    rows = queue_path.read_text(encoding="utf-8").splitlines()
    if len(rows) <= 1:
        return []
    phones: List[str] = []
    for line in rows[1:]:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        phone = parts[1]
        if phone:
            phones.append(phone)
    return phones


def max_recipients(config: Dict[str, object], mode: str) -> int:
    survey_cfg = config.get("survey", {})
    if not isinstance(survey_cfg, dict):
        return 1
    mapping = survey_cfg.get("max_recipients_per_run", {})
    if not isinstance(mapping, dict):
        return 1
    return int(mapping.get(mode, 1))


def build_simulated_response(phone: str, mode: str) -> Dict[str, object]:
    return {
        "response_id": str(uuid.uuid4()),
        "phone": phone,
        "survey_mode": "voice",
        "campaign_mode": mode,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "consent": "yes",
        "participant_role": "researcher",
        "organization_type": "academia",
        "country": "Switzerland",
        "task_completed": "yes",
        "usefulness_score_1to5": 5,
        "ease_of_use_score_1to5": 4,
        "interpretability_score_1to5": 4,
        "trust_in_results_score_1to5": 4,
        "policy_actionability_score_1to5": 4,
        "would_recommend_1to5": 5,
        "key_insight": "Prototype run confirms strong feasibility for mixed-mode outreach.",
        "top_issue": "Need tighter wording for compensation clarification.",
        "acc_connectivity_reliability": 4,
        "skl_self_efficacy_digital_tasks": 4,
        "eco_digital_payments_use": 5,
        "srv_gov_service_completion_digital": 4,
        "agr_control_over_phone_account": 5,
        "out_perceived_welfare_change": 4,
        "call_attempt_count": 1,
        "interview_duration_sec": 480,
        "breakoff_flag": False,
        "follow_up_consent": "yes",
    }


def run(config_path: str, mode: str) -> None:
    root = Path(__file__).resolve().parents[2]
    config = load_yaml(config_path)
    path_cfg = config.get("paths", {})
    twilio_cfg = config.get("twilio", {})
    if not isinstance(path_cfg, dict) or not isinstance(twilio_cfg, dict):
        raise ValueError("Invalid survey automation config sections")

    responses_path = root / str(path_cfg.get("responses_jsonl", "data/primary/survey_responses.jsonl"))
    events_path = root / str(path_cfg.get("survey_events_jsonl", "data/primary/survey_events.jsonl"))
    ensure_dir(responses_path.parent)
    ensure_dir(events_path.parent)

    recipients = recipients_for_mode(config, mode=mode, root=root)[: max_recipients(config, mode)]
    if not recipients:
        raise ValueError(f"No recipients available for mode: {mode}")

    dry_run = bool(twilio_cfg.get("dry_run", False))
    voice_webhook_url = str(twilio_cfg.get("voice_webhook_url", "")).strip()
    status_callback_url = get_env("TWILIO_STATUS_CALLBACK_URL")
    hash_salt = get_env("SURVEY_PHONE_HASH_SALT", default="local-dev")

    if dry_run:
        print("[twilio] dry-run enabled; writing simulated responses")
        rows = [build_simulated_response(phone, mode=mode) for phone in recipients]
        append_jsonl(responses_path, rows)
        append_jsonl(
            events_path,
            [
                {
                    "event_type": "simulated_call_complete",
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "campaign_mode": mode,
                    "recipient_count": len(rows),
                }
            ],
        )
        print(f"[twilio] wrote {len(rows)} simulated responses to {responses_path}")
        return

    if Client is None:
        raise RuntimeError("Twilio SDK missing. Install requirements to run live calls.")
    if not voice_webhook_url:
        raise ValueError("twilio.voice_webhook_url is required for live mode")

    account_sid = os.getenv("TWILIO_ACCOUNT_SID", "").strip()
    auth_token = os.getenv("TWILIO_AUTH_TOKEN", "").strip()
    from_number = os.getenv("TWILIO_FROM_NUMBER", "").strip()
    if not account_sid or not auth_token or not from_number:
        raise ValueError("TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, and TWILIO_FROM_NUMBER must be set")

    client = Client(account_sid, auth_token)
    created_events: List[Dict[str, object]] = []
    for phone in recipients:
        call = client.calls.create(
            to=phone,
            from_=from_number,
            url=voice_webhook_url,
            status_callback=status_callback_url or None,
            status_callback_method=str(twilio_cfg.get("status_callback_method", "POST")),
            machine_detection=str(twilio_cfg.get("machine_detection", "DetectMessageEnd")),
            timeout=int(twilio_cfg.get("timeout_seconds", 40)),
        )
        created_events.append(
            {
                "event_type": "twilio_call_created",
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "campaign_mode": mode,
                "call_sid": call.sid,
                "phone_hash": stable_phone_hash(phone, hash_salt),
                "from_number": from_number,
                "status": "queued",
                "run_id": _utc_timestamp(),
            }
        )
    append_jsonl(events_path, created_events)
    print(f"[twilio] created {len(created_events)} calls. Events written to {events_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Trigger Twilio prototype and campaign calls")
    parser.add_argument("--config", default="config/survey_automation.yml", help="Path to survey automation config")
    parser.add_argument(
        "--mode",
        default="prototype",
        choices=["prototype", "cohort1", "cohort2", "full"],
        help="Campaign mode",
    )
    args = parser.parse_args()
    run(config_path=args.config, mode=args.mode)


if __name__ == "__main__":
    main()
