from pathlib import Path

import json
import pandas as pd
import streamlit as st


st.set_page_config(page_title="Survey Live Dashboard", page_icon="📞", layout="wide")

ROOT = Path(__file__).resolve().parents[1]
RESPONSES_PATH = ROOT / "data" / "primary" / "survey_responses.jsonl"
EVENTS_PATH = ROOT / "data" / "primary" / "survey_events.jsonl"
ELIGIBILITY_PATH = ROOT / "data" / "gold" / "survey_incentive_eligibility.csv"
EXPORT_PATH = ROOT / "data" / "gold" / "dpi_dclo_primary_export.csv"


def read_jsonl(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return pd.DataFrame(rows)


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def auto_refresh_section() -> None:
    st.sidebar.header("Refresh")
    auto_refresh = st.sidebar.toggle("Auto refresh", value=True)
    seconds = st.sidebar.slider("Refresh every (seconds)", min_value=5, max_value=60, value=15, step=5)
    if auto_refresh:
        st.markdown(f"<meta http-equiv='refresh' content='{seconds}'>", unsafe_allow_html=True)
    st.sidebar.caption("Dashboard reads data directly from incoming survey files.")


def main() -> None:
    auto_refresh_section()
    st.title("Live Phone Survey Monitoring")
    st.caption("Incoming survey data, eligibility status, and DPI/DCLO exports update automatically.")

    responses = read_jsonl(RESPONSES_PATH)
    events = read_jsonl(EVENTS_PATH)
    eligibility = read_csv(ELIGIBILITY_PATH)
    exports = read_csv(EXPORT_PATH)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Raw responses", int(len(responses)))
    c2.metric("Survey events", int(len(events)))
    c3.metric("Eligibility rows", int(len(eligibility)))
    c4.metric("Export rows", int(len(exports)))

    if not eligibility.empty and "completion_status" in eligibility.columns:
        status_counts = eligibility["completion_status"].value_counts(dropna=False).rename_axis("status").reset_index(name="count")
        st.subheader("Completion status")
        st.bar_chart(status_counts.set_index("status"))

    if not eligibility.empty and "incentive_eligible" in eligibility.columns:
        eligible_count = int((eligibility["incentive_eligible"].astype(str) == "true").sum())
        st.info(f"INR 200 payout-eligible completions: {eligible_count}")

    st.subheader("Latest call events")
    if events.empty:
        st.warning("No events found yet.")
    else:
        show_events = events.copy()
        if "timestamp_utc" in show_events.columns:
            show_events = show_events.sort_values("timestamp_utc", ascending=False)
        st.dataframe(show_events.head(30), use_container_width=True)

    st.subheader("Latest responses (mixed-mode)")
    if responses.empty:
        st.warning("No responses found yet.")
    else:
        show_responses = responses.copy()
        if "timestamp_utc" in show_responses.columns:
            show_responses = show_responses.sort_values("timestamp_utc", ascending=False)
        st.dataframe(show_responses.head(30), use_container_width=True)

    st.subheader("DPI/DCLO export preview")
    if exports.empty:
        st.warning("No export rows found yet.")
    else:
        st.dataframe(exports.head(30), use_container_width=True)

    st.caption(f"Data files: {RESPONSES_PATH} | {EVENTS_PATH} | {ELIGIBILITY_PATH} | {EXPORT_PATH}")


if __name__ == "__main__":
    main()
