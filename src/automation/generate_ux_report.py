import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd

from common import load_yaml, read_jsonl


UX_FIELDS = [
    "usefulness_score_1to5",
    "ease_of_use_score_1to5",
    "interpretability_score_1to5",
    "trust_in_results_score_1to5",
    "policy_actionability_score_1to5",
]


def _safe_mean(frame: pd.DataFrame, column: str) -> float:
    if column not in frame.columns:
        return 0.0
    return float(pd.to_numeric(frame[column], errors="coerce").mean())


def run(config_path: str, mode: str) -> None:
    root = Path(__file__).resolve().parents[2]
    config = load_yaml(config_path)
    paths = config.get("paths", {})
    if not isinstance(paths, dict):
        raise ValueError("Invalid paths config")

    responses_path = root / str(paths.get("responses_jsonl", "data/primary/survey_responses.jsonl"))
    eligibility_path = root / str(paths.get("eligibility_csv", "data/gold/survey_incentive_eligibility.csv"))
    report_path = root / str(paths.get("ux_report_md", "data/gold/prototype_ux_report.md"))

    rows = [row for row in read_jsonl(responses_path) if "question_id" not in row]
    frame = pd.DataFrame(rows) if rows else pd.DataFrame()
    if not frame.empty and "campaign_mode" in frame.columns:
        frame = frame[frame["campaign_mode"].fillna("prototype") == mode].copy()

    completion_rate = 0.0
    eligible_count = 0
    if eligibility_path.exists():
        eligibility = pd.read_csv(eligibility_path)
        if "campaign_mode" in eligibility.columns:
            eligibility = eligibility[eligibility["campaign_mode"].fillna("prototype") == mode]
        total = len(eligibility)
        if total:
            completion_rate = float((eligibility["completion_status"] == "complete").mean()) * 100.0
            eligible_count = int((eligibility["incentive_eligible"].astype(str) == "true").sum())

    avg_duration = float(pd.to_numeric(frame.get("interview_duration_sec", pd.Series(dtype=float)), errors="coerce").mean())
    if pd.isna(avg_duration):
        avg_duration = 0.0

    ux_lines: List[str] = [
        "# Prototype UX Report",
        "",
        f"- Campaign mode: `{mode}`",
        f"- Responses analyzed: `{len(frame)}`",
        f"- Completion rate: `{completion_rate:.1f}%`",
        f"- Incentive-eligible responses: `{eligible_count}`",
        f"- Average interview duration (sec): `{avg_duration:.0f}`",
        "",
        "## UX Score Snapshot (1-5)",
    ]
    for field in UX_FIELDS:
        ux_lines.append(f"- `{field}`: `{_safe_mean(frame, field):.2f}`")

    ux_lines.extend(
        [
            "",
            "## Findings",
            "- Compensation message is included in the call flow and should be audited for clarity.",
            "- Required-field completeness should be verified against completion_status in eligibility outputs.",
            "- Use this report as a go/no-go checkpoint before expanding cohort size.",
        ]
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(ux_lines), encoding="utf-8")
    print(f"Wrote UX report: {report_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate prototype and cohort UX markdown report")
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
