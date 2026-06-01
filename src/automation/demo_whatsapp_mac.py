import argparse
import csv
import json
import os
import subprocess
import time
import urllib.parse
import uuid
from hashlib import sha256
from pathlib import Path
from typing import Dict, List

from common import load_yaml

DEBUG_LOG_PATH = Path("/Users/rahuljha/Desktop/coding projects/.cursor/debug-d98a23.log")
DEBUG_SESSION_ID = "d98a23"
DEMO_STATE_PATH = Path("/Users/rahuljha/Desktop/coding projects/data pipelines/projects/india-open-data-powerbi/data/primary/whatsapp_demo_state.csv")
DEMO_LOCK_PATH = Path("/Users/rahuljha/Desktop/coding projects/data pipelines/projects/india-open-data-powerbi/data/primary/whatsapp_demo.lock")


def _debug_log(run_id: str, hypothesis_id: str, location: str, message: str, data: Dict[str, object]) -> None:
    payload = {
        "sessionId": DEBUG_SESSION_ID,
        "id": f"log_{uuid.uuid4().hex}",
        "timestamp": int(time.time() * 1000),
        "runId": run_id,
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
    }
    with open(DEBUG_LOG_PATH, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True))
        handle.write("\n")


def load_questions(root: Path) -> List[Dict[str, str]]:
    import json

    flow_path = root / "config" / "whatsapp_questionnaire_flow.json"
    with open(flow_path, "r", encoding="utf-8") as handle:
        flow = json.load(handle)
    questions = flow.get("questions", [])
    if not isinstance(questions, list):
        return []
    out: List[Dict[str, str]] = []
    for q in questions:
        if isinstance(q, dict):
            out.append({"id": str(q.get("id", "")), "prompt": str(q.get("prompt", ""))})
    return out


def _read_demo_state() -> List[Dict[str, str]]:
    if not DEMO_STATE_PATH.exists():
        return []
    with open(DEMO_STATE_PATH, "r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_demo_state(rows: List[Dict[str, str]]) -> None:
    DEMO_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["phone", "next_message_index", "last_message_hash", "last_sent_epoch", "completed"]
    with open(DEMO_STATE_PATH, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def _load_phone_state(phone: str) -> Dict[str, str]:
    rows = _read_demo_state()
    for row in rows:
        if row.get("phone", "") == phone:
            return row
    return {
        "phone": phone,
        "next_message_index": "0",
        "last_message_hash": "",
        "last_sent_epoch": "0",
        "completed": "false",
    }


def _save_phone_state(updated: Dict[str, str]) -> None:
    rows = _read_demo_state()
    found = False
    for idx, row in enumerate(rows):
        if row.get("phone", "") == updated.get("phone", ""):
            rows[idx] = updated
            found = True
            break
    if not found:
        rows.append(updated)
    _write_demo_state(rows)


def _should_skip_duplicate(state: Dict[str, str], message: str, dedupe_seconds: int) -> bool:
    message_hash = sha256(message.encode("utf-8")).hexdigest()
    last_hash = state.get("last_message_hash", "")
    last_epoch = float(state.get("last_sent_epoch", "0") or 0)
    now_epoch = time.time()
    return message_hash == last_hash and (now_epoch - last_epoch) < dedupe_seconds


def _acquire_lock(run_id: str) -> bool:
    now = time.time()
    stale_after = 1800
    if DEMO_LOCK_PATH.exists():
        age = now - DEMO_LOCK_PATH.stat().st_mtime
        if age > stale_after:
            DEMO_LOCK_PATH.unlink(missing_ok=True)
        else:
            # region agent log
            _debug_log(
                run_id=run_id,
                hypothesis_id="H9",
                location="demo_whatsapp_mac.py:_acquire_lock",
                message="Active lock detected; refusing concurrent run",
                data={"lock_age_seconds": age},
            )
            # endregion agent log
            return False
    DEMO_LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    DEMO_LOCK_PATH.write_text(f"{run_id}|{int(now)}", encoding="utf-8")
    return True


def _release_lock() -> None:
    DEMO_LOCK_PATH.unlink(missing_ok=True)


def send_via_whatsapp_app(phone: str, message: str, auto_send: bool) -> None:
    run_id = os.getenv("WA_DEBUG_RUN_ID", f"run_{uuid.uuid4().hex[:8]}")
    phone_no_plus = phone.replace("+", "")
    encoded = urllib.parse.quote(message)
    uri = f"whatsapp://send?phone={phone_no_plus}&text={encoded}"
    open_result = subprocess.run(["open", uri], check=False)
    # region agent log
    _debug_log(
        run_id=run_id,
        hypothesis_id="H4",
        location="demo_whatsapp_mac.py:send_via_whatsapp_app",
        message="Opened WhatsApp URI for message draft",
        data={
            "open_returncode": open_result.returncode,
            "auto_send": auto_send,
            "target_suffix": phone[-2:] if phone else "",
            "msg_len": len(message),
        },
    )
    # endregion agent log
    time.sleep(1.2)
    if not auto_send:
        return
    script = """
tell application "WhatsApp" to activate
delay 0.4
tell application "System Events"
  keystroke return
end tell
"""
    result = subprocess.run(["osascript", "-e", script], check=False)
    # region agent log
    _debug_log(
        run_id=run_id,
        hypothesis_id="H5",
        location="demo_whatsapp_mac.py:send_via_whatsapp_app",
        message="Attempted auto-send keystroke via osascript",
        data={"osascript_returncode": result.returncode},
    )
    # endregion agent log
    if result.returncode != 0:
        print(
            "[wa-mac-demo] auto-send blocked by macOS permissions. "
            "Enable Accessibility for Terminal/Cursor in System Settings > Privacy & Security > Accessibility."
        )


def run(
    config_path: str,
    phone: str,
    question_count: int,
    gap_seconds: float,
    auto_send: bool,
    dry_run: bool,
    reset_demo_state: bool,
) -> None:
    run_id = os.getenv("WA_DEBUG_RUN_ID", f"run_{uuid.uuid4().hex[:8]}")
    root = Path(__file__).resolve().parents[2]
    _ = load_yaml(config_path)
    questions = load_questions(root)
    if not questions:
        raise ValueError("No WhatsApp questionnaire questions found")

    selected = questions[: max(1, min(question_count, len(questions)))]
    intro = (
        "Hi Rahul - this is the ITU WhatsApp questionnaire demo. "
        "Please reply naturally in text or with score where useful."
    )
    messages = [intro]
    for idx, q in enumerate(selected, start=1):
        messages.append(f"Q{idx}: {q['prompt']}")
    # region agent log
    _debug_log(
        run_id=run_id,
        hypothesis_id="H3",
        location="demo_whatsapp_mac.py:run",
        message="Prepared WhatsApp demo message set",
        data={
            "question_count_requested": question_count,
            "question_count_selected": len(selected),
            "messages_total": len(messages),
            "dry_run": dry_run,
            "auto_send": auto_send,
        },
    )
    # endregion agent log

    if not _acquire_lock(run_id=run_id):
        print("[wa-mac-demo] another run is active; skipping to prevent duplicate sends")
        return
    print(f"[wa-mac-demo] target: {phone}")
    print(f"[wa-mac-demo] messages: {len(messages)} | auto_send={auto_send} | dry_run={dry_run}")
    state = _load_phone_state(phone)
    if reset_demo_state:
        state = {
            "phone": phone,
            "next_message_index": "0",
            "last_message_hash": "",
            "last_sent_epoch": "0",
            "completed": "false",
        }
        _save_phone_state(state)
    start_idx = int(state.get("next_message_index", "0") or 0)
    # region agent log
    _debug_log(
        run_id=run_id,
        hypothesis_id="H6",
        location="demo_whatsapp_mac.py:run",
        message="Loaded demo state for resume",
        data={
            "start_idx": start_idx,
            "completed": state.get("completed", "false"),
            "last_sent_epoch": state.get("last_sent_epoch", ""),
        },
    )
    # endregion agent log
    try:
        for idx, message in enumerate(messages[start_idx:], start=start_idx + 1):
            if _should_skip_duplicate(state=state, message=message, dedupe_seconds=120):
                # region agent log
                _debug_log(
                    run_id=run_id,
                    hypothesis_id="H7",
                    location="demo_whatsapp_mac.py:run",
                    message="Skipped duplicate message due dedupe window",
                    data={"message_index": idx, "msg_len": len(message)},
                )
                # endregion agent log
                continue
            print(f"[wa-mac-demo] {idx}/{len(messages)} -> {message[:90]}...")
            if not dry_run:
                send_via_whatsapp_app(phone=phone, message=message, auto_send=auto_send)
            state["next_message_index"] = str(idx)
            state["last_message_hash"] = sha256(message.encode("utf-8")).hexdigest()
            state["last_sent_epoch"] = str(time.time())
            state["completed"] = "true" if idx >= len(messages) else "false"
            _save_phone_state(state)
            time.sleep(max(0.5, gap_seconds))
        print("[wa-mac-demo] complete")
    finally:
        _release_lock()


def main() -> None:
    parser = argparse.ArgumentParser(description="Demo WhatsApp questionnaire via WhatsApp Mac app")
    parser.add_argument("--config", default="config/survey_automation.yml")
    parser.add_argument("--phone", default="+41788186778")
    parser.add_argument("--question-count", type=int, default=3)
    parser.add_argument("--gap-seconds", type=float, default=3.0)
    parser.add_argument("--auto-send", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--reset-demo-state", action="store_true")
    args = parser.parse_args()
    run(
        config_path=args.config,
        phone=args.phone,
        question_count=args.question_count,
        gap_seconds=args.gap_seconds,
        auto_send=args.auto_send,
        dry_run=args.dry_run,
        reset_demo_state=args.reset_demo_state,
    )


if __name__ == "__main__":
    main()
