import argparse
import csv
import os
from pathlib import Path
from typing import Dict, List

import requests

from common import load_yaml


def read_queue(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def run(config_path: str, dry_run: bool) -> None:
    root = Path(__file__).resolve().parents[2]
    config = load_yaml(config_path)
    paths = config.get("paths", {})
    whatsapp_cfg = config.get("whatsapp", {})
    if not isinstance(paths, dict) or not isinstance(whatsapp_cfg, dict):
        raise ValueError("Invalid config for WhatsApp API sender")

    queue_path = root / str(paths.get("outreach_queue_csv", "data/primary/outreach_queue.csv"))
    queue = read_queue(queue_path)
    business_cfg = whatsapp_cfg.get("business_api", {})
    if not isinstance(business_cfg, dict):
        raise ValueError("Missing whatsapp.business_api config")

    phone_id_env = str(business_cfg.get("phone_number_id_env", "WHATSAPP_PHONE_NUMBER_ID"))
    token_env = str(business_cfg.get("token_env", "WHATSAPP_PERMANENT_TOKEN"))
    api_version = str(business_cfg.get("api_version", "v20.0"))
    phone_number_id = os.getenv(phone_id_env, "").strip()
    token = os.getenv(token_env, "").strip()

    if dry_run:
        print(f"[whatsapp-api] dry-run: would send {len(queue)} messages from queue {queue_path}")
        return
    if not phone_number_id or not token:
        raise ValueError(f"Missing WhatsApp API credentials in env vars {phone_id_env} and/or {token_env}")

    endpoint = f"https://graph.facebook.com/{api_version}/{phone_number_id}/messages"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    sent = 0
    for row in queue:
        if str(row.get("opt_in_status", "")).lower() != "yes":
            continue
        payload = {
            "messaging_product": "whatsapp",
            "to": str(row.get("phone", "")).replace("+", ""),
            "type": "text",
            "text": {"body": str(row.get("invite_message", ""))},
        }
        response = requests.post(endpoint, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        sent += 1
    print(f"[whatsapp-api] sent {sent} messages")


def main() -> None:
    parser = argparse.ArgumentParser(description="Send outreach messages with WhatsApp Business API")
    parser.add_argument("--config", default="config/survey_automation.yml")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    run(config_path=args.config, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
