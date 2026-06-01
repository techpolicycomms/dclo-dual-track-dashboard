import argparse
import csv
import time
import webbrowser
from pathlib import Path
from typing import Dict, List

from common import load_yaml


def read_queue(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def run(config_path: str, auto_advance_seconds: int) -> None:
    root = Path(__file__).resolve().parents[2]
    config = load_yaml(config_path)
    paths = config.get("paths", {})
    if not isinstance(paths, dict):
        raise ValueError("Invalid paths config")
    queue_path = root / str(paths.get("outreach_queue_csv", "data/primary/outreach_queue.csv"))
    queue = read_queue(queue_path)
    if not queue:
        print("[whatsapp-web] queue is empty")
        return

    print("[whatsapp-web] Opening links in browser with manual approval checkpoints")
    for idx, row in enumerate(queue, start=1):
        if str(row.get("opt_in_status", "")).lower() != "yes":
            continue
        link = str(row.get("whatsapp_web_link", "")).strip()
        if not link:
            continue
        print(f"[{idx}] {row.get('name', 'Participant')} -> {row.get('phone', '')}")
        webbrowser.open(link, new=2)
        if auto_advance_seconds > 0:
            time.sleep(auto_advance_seconds)
            continue
        answer = input("Send this message and continue? [y/N]: ").strip().lower()
        if answer != "y":
            print("[whatsapp-web] stopping at user request")
            break
    print("[whatsapp-web] run complete")


def main() -> None:
    parser = argparse.ArgumentParser(description="Open WhatsApp Web invite links with manual approval")
    parser.add_argument("--config", default="config/survey_automation.yml")
    parser.add_argument(
        "--auto-advance-seconds",
        type=int,
        default=0,
        help="If set > 0, automatically proceeds to next message after N seconds",
    )
    args = parser.parse_args()
    run(config_path=args.config, auto_advance_seconds=args.auto_advance_seconds)


if __name__ == "__main__":
    main()
