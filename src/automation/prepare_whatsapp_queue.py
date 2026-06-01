import argparse
import csv
import urllib.parse
from pathlib import Path
from typing import Dict, List

from common import ensure_dir, load_yaml


OPT_IN_TEMPLATE = (
    "Hello {name}, we are running an academic phone survey on digital capability and policy outcomes. "
    "You will receive INR 200 after successfully completing the full survey. "
    "Compensation delivery method and timing will be shared after completion verification. "
    "Reply YES to opt in."
)

INVITE_TEMPLATE = (
    "Thank you for opting in, {name}. Please complete your survey here: {survey_link}. "
    "You will receive INR 200 after successful full completion."
)

REMINDER_TEMPLATE = (
    "Reminder for {name}: your survey is still pending. Full completion qualifies for INR 200 compensation. "
    "Continue here: {survey_link}"
)

COMPLETE_TEMPLATE = (
    "Thanks {name}, your survey is marked complete. INR 200 compensation is approved subject to verification. "
    "Payment method/timing details will be shared separately."
)


def read_contacts(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def run(config_path: str, mode: str) -> None:
    root = Path(__file__).resolve().parents[2]
    config = load_yaml(config_path)
    paths = config.get("paths", {})
    if not isinstance(paths, dict):
        raise ValueError("Invalid paths config")

    contacts_path = root / "data" / "primary" / "whatsapp_contacts.csv"
    queue_path = root / str(paths.get("outreach_queue_csv", "data/primary/outreach_queue.csv"))
    ensure_dir(queue_path.parent)

    contacts = read_contacts(contacts_path)
    if not contacts:
        contacts = [
            {
                "respondent_id": "prototype_001",
                "name": "Prototype User",
                "phone": "+41788186778",
                "language": "en",
                "campaign_mode": "prototype",
                "opt_in_status": "yes",
            }
        ]

    queue_rows: List[Dict[str, str]] = []
    for item in contacts:
        campaign_mode = item.get("campaign_mode", "prototype")
        if mode != "full" and campaign_mode != mode:
            continue
        name = item.get("name", "Participant")
        phone = item.get("phone", "")
        respondent_id = item.get("respondent_id", "")
        if not phone:
            continue
        survey_link = f"https://example.com/survey?respondent_id={urllib.parse.quote(respondent_id)}"
        invite_text = INVITE_TEMPLATE.format(name=name, survey_link=survey_link)
        row = {
            "respondent_id": respondent_id,
            "phone": phone,
            "name": name,
            "campaign_mode": campaign_mode,
            "opt_in_status": item.get("opt_in_status", "pending"),
            "opt_in_message": OPT_IN_TEMPLATE.format(name=name),
            "invite_message": invite_text,
            "reminder_message": REMINDER_TEMPLATE.format(name=name, survey_link=survey_link),
            "completion_message": COMPLETE_TEMPLATE.format(name=name),
            "whatsapp_web_link": f"https://web.whatsapp.com/send?phone={phone.replace('+', '')}&text={urllib.parse.quote(invite_text)}",
        }
        queue_rows.append(row)

    fieldnames = [
        "respondent_id",
        "phone",
        "name",
        "campaign_mode",
        "opt_in_status",
        "opt_in_message",
        "invite_message",
        "reminder_message",
        "completion_message",
        "whatsapp_web_link",
    ]
    with open(queue_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in queue_rows:
            writer.writerow(row)
    print(f"Wrote outreach queue: {queue_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare WhatsApp outreach queue with incentive messaging")
    parser.add_argument("--config", default="config/survey_automation.yml")
    parser.add_argument(
        "--mode",
        default="prototype",
        choices=["prototype", "cohort1", "cohort2", "full"],
    )
    args = parser.parse_args()
    run(config_path=args.config, mode=args.mode)


if __name__ == "__main__":
    main()
