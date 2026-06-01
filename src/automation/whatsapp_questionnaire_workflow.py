import argparse
import csv
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import requests

from common import ensure_dir, load_yaml

DEBUG_LOG_PATH = Path("/Users/rahuljha/Desktop/coding projects/.cursor/debug-d98a23.log")
DEBUG_SESSION_ID = "d98a23"


def _debug_log(run_id: str, hypothesis_id: str, location: str, message: str, data: Dict[str, object]) -> None:
    payload = {
        "sessionId": DEBUG_SESSION_ID,
        "id": f"log_{uuid.uuid4().hex}",
        "timestamp": int(datetime.now(timezone.utc).timestamp() * 1000),
        "runId": run_id,
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
    }
    with open(DEBUG_LOG_PATH, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True))
        handle.write("\n")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv_rows(path: Path, rows: List[Dict[str, str]], fieldnames: List[str]) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def load_questionnaire_flow(root: Path) -> Dict[str, object]:
    path = root / "config" / "whatsapp_questionnaire_flow.json"
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("Invalid whatsapp questionnaire flow")
    return data


def ensure_state(config_path: str, mode: str, reset_progress: bool = False) -> Path:
    run_id = os.getenv("WA_DEBUG_RUN_ID", f"run_{uuid.uuid4().hex[:8]}")
    root = Path(__file__).resolve().parents[2]
    config = load_yaml(config_path)
    paths = config.get("paths", {})
    if not isinstance(paths, dict):
        raise ValueError("Invalid paths config")

    queue_path = root / str(paths.get("outreach_queue_csv", "data/primary/outreach_queue.csv"))
    state_path = root / str(paths.get("whatsapp_questionnaire_state_csv", "data/primary/whatsapp_questionnaire_state.csv"))
    queue_rows = read_csv_rows(queue_path)
    state_rows = read_csv_rows(state_path)
    state_by_id = {row.get("respondent_id", ""): row for row in state_rows}
    # region agent log
    _debug_log(
        run_id=run_id,
        hypothesis_id="H1",
        location="whatsapp_questionnaire_workflow.py:ensure_state",
        message="Queue and existing state loaded",
        data={"mode": mode, "queue_count": len(queue_rows), "state_count": len(state_rows)},
    )
    # endregion agent log

    updated: List[Dict[str, str]] = []
    for row in queue_rows:
        if row.get("campaign_mode", "prototype") != mode and mode != "full":
            continue
        if str(row.get("opt_in_status", "")).lower() != "yes":
            continue
        rid = row.get("respondent_id", "")
        existing = state_by_id.get(rid, {})
        question_index = existing.get("question_index", "0")
        status = existing.get("status", "ready")
        last_prompt_id = existing.get("last_prompt_id", "")
        last_sent_at_utc = existing.get("last_sent_at_utc", "")
        last_response_text = existing.get("last_response_text", "")
        completed_at_utc = existing.get("completed_at_utc", "")
        if reset_progress:
            question_index = "0"
            status = "ready"
            last_prompt_id = ""
            last_sent_at_utc = ""
            last_response_text = ""
            completed_at_utc = ""
        updated.append(
            {
                "respondent_id": rid,
                "name": row.get("name", "Participant"),
                "phone": row.get("phone", ""),
                "campaign_mode": row.get("campaign_mode", mode),
                "question_index": question_index,
                "status": status,
                "last_prompt_id": last_prompt_id,
                "last_sent_at_utc": last_sent_at_utc,
                "last_response_text": last_response_text,
                "completed_at_utc": completed_at_utc,
            }
        )

    fieldnames = [
        "respondent_id",
        "name",
        "phone",
        "campaign_mode",
        "question_index",
        "status",
        "last_prompt_id",
        "last_sent_at_utc",
        "last_response_text",
        "completed_at_utc",
    ]
    write_csv_rows(state_path, updated, fieldnames)
    # region agent log
    _debug_log(
        run_id=run_id,
        hypothesis_id="H1",
        location="whatsapp_questionnaire_workflow.py:ensure_state",
        message="State file written after eligibility filtering",
        data={"mode": mode, "eligible_state_rows": len(updated), "reset_progress": reset_progress},
    )
    # endregion agent log
    print(f"[wa-workflow] state prepared: {state_path}")
    return state_path


def compose_question_message(flow: Dict[str, object], respondent_name: str, question_index: int) -> str:
    intro = str(flow.get("intro_message", "")).format(name=respondent_name)
    questions = flow.get("questions", [])
    if not isinstance(questions, list):
        raise ValueError("Flow questions must be a list")
    if question_index == 0:
        first_prompt = str(questions[0].get("prompt", ""))
        return f"{intro}\n\nQ1/{len(questions)}: {first_prompt}"
    if question_index < len(questions):
        prompt = str(questions[question_index].get("prompt", ""))
        return f"Q{question_index + 1}/{len(questions)}: {prompt}"
    return str(flow.get("closing_message", "")).format(name=respondent_name)


def send_via_business_api(config: Dict[str, object], to_phone: str, body: str, dry_run: bool) -> None:
    run_id = os.getenv("WA_DEBUG_RUN_ID", f"run_{uuid.uuid4().hex[:8]}")
    whatsapp_cfg = config.get("whatsapp", {})
    if not isinstance(whatsapp_cfg, dict):
        raise ValueError("Missing whatsapp config")
    business = whatsapp_cfg.get("business_api", {})
    if not isinstance(business, dict):
        raise ValueError("Missing whatsapp.business_api config")

    phone_id_env = str(business.get("phone_number_id_env", "WHATSAPP_PHONE_NUMBER_ID"))
    token_env = str(business.get("token_env", "WHATSAPP_PERMANENT_TOKEN"))
    api_version = str(business.get("api_version", "v20.0"))
    phone_number_id = os.getenv(phone_id_env, "").strip()
    token = os.getenv(token_env, "").strip()
    # region agent log
    _debug_log(
        run_id=run_id,
        hypothesis_id="H2",
        location="whatsapp_questionnaire_workflow.py:send_via_business_api",
        message="Prepared WhatsApp Business API send context",
        data={
            "dry_run": dry_run,
            "has_phone_number_id": bool(phone_number_id),
            "has_token": bool(token),
            "msg_len": len(body),
            "target_suffix": to_phone[-2:] if to_phone else "",
        },
    )
    # endregion agent log

    if dry_run:
        print(f"[wa-workflow] dry-run send -> {to_phone}: {body[:100]}...")
        return
    if not phone_number_id or not token:
        raise ValueError(f"Missing env vars {phone_id_env} and/or {token_env}")

    endpoint = f"https://graph.facebook.com/{api_version}/{phone_number_id}/messages"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    payload = {
        "messaging_product": "whatsapp",
        "to": to_phone.replace("+", ""),
        "type": "text",
        "text": {"body": body},
    }
    response = requests.post(endpoint, headers=headers, json=payload, timeout=30)
    response.raise_for_status()


def send_next_question(
    config_path: str,
    mode: str,
    dry_run: bool,
    reset_progress: bool = False,
    allow_awaiting_retry: bool = False,
) -> None:
    run_id = os.getenv("WA_DEBUG_RUN_ID", f"run_{uuid.uuid4().hex[:8]}")
    root = Path(__file__).resolve().parents[2]
    config = load_yaml(config_path)
    state_path = ensure_state(config_path=config_path, mode=mode, reset_progress=reset_progress)
    flow = load_questionnaire_flow(root)
    questions = flow.get("questions", [])
    if not isinstance(questions, list):
        raise ValueError("Invalid questionnaire questions")

    rows = read_csv_rows(state_path)
    # region agent log
    _debug_log(
        run_id=run_id,
        hypothesis_id="H3",
        location="whatsapp_questionnaire_workflow.py:send_next_question",
        message="Loaded workflow state rows before iteration",
        data={
            "mode": mode,
            "dry_run": dry_run,
            "rows_total": len(rows),
            "ready_rows": sum(1 for r in rows if r.get("status") in {"ready", "awaiting_reply"}),
        },
    )
    # endregion agent log
    for row in rows:
        status = row.get("status", "")
        if status not in {"ready", "awaiting_reply"}:
            continue
        if status == "awaiting_reply" and not allow_awaiting_retry:
            # region agent log
            _debug_log(
                run_id=run_id,
                hypothesis_id="H8",
                location="whatsapp_questionnaire_workflow.py:send_next_question",
                message="Skipped send because awaiting reply and retry not allowed",
                data={"respondent_id": row.get("respondent_id", ""), "status": status},
            )
            # endregion agent log
            continue
        phone = row.get("phone", "")
        if not phone:
            continue
        q_idx = int(row.get("question_index", "0") or 0)
        message = compose_question_message(flow=flow, respondent_name=row.get("name", "Participant"), question_index=q_idx)
        send_via_business_api(config=config, to_phone=phone, body=message, dry_run=dry_run)
        # region agent log
        _debug_log(
            run_id=run_id,
            hypothesis_id="H3",
            location="whatsapp_questionnaire_workflow.py:send_next_question",
            message="Question sent and state advanced",
            data={
                "respondent_id": row.get("respondent_id", ""),
                "previous_question_index": q_idx,
                "new_question_index": q_idx + 1,
                "status_before": row.get("status", ""),
                "questions_total": len(questions),
            },
        )
        # endregion agent log
        row["last_sent_at_utc"] = utc_now()
        row["status"] = "awaiting_reply"
        if q_idx < len(questions):
            row["last_prompt_id"] = str(questions[q_idx].get("id", ""))
            row["question_index"] = str(q_idx + 1)
        else:
            row["status"] = "completed"
            row["completed_at_utc"] = utc_now()

    fieldnames = [
        "respondent_id",
        "name",
        "phone",
        "campaign_mode",
        "question_index",
        "status",
        "last_prompt_id",
        "last_sent_at_utc",
        "last_response_text",
        "completed_at_utc",
    ]
    write_csv_rows(state_path, rows, fieldnames)
    print(f"[wa-workflow] updated state: {state_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="WhatsApp Business questionnaire workflow")
    parser.add_argument("--config", default="config/survey_automation.yml")
    parser.add_argument("--mode", default="prototype", choices=["prototype", "cohort1", "cohort2", "full"])
    parser.add_argument("--action", default="send-next", choices=["prepare-state", "send-next"])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--reset-progress", action="store_true")
    parser.add_argument("--allow-awaiting-retry", action="store_true")
    args = parser.parse_args()

    if args.action == "prepare-state":
        ensure_state(config_path=args.config, mode=args.mode, reset_progress=args.reset_progress)
        return
    send_next_question(
        config_path=args.config,
        mode=args.mode,
        dry_run=args.dry_run,
        reset_progress=args.reset_progress,
        allow_awaiting_retry=args.allow_awaiting_retry,
    )


if __name__ == "__main__":
    main()
