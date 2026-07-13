from pathlib import Path

import json
import sqlite3
import pandas as pd
import plotly.express as px
import streamlit as st


st.set_page_config(page_title="DCLO Dashboard", page_icon="📊", layout="wide")

STATE_DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "gold" / "dclo_state_year.csv"
COUNTRY_DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "gold" / "dclo_country_year.csv"
COUNTRY_API_DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "gold" / "dclo_country_year_api.csv"
EXPLAINER_PATH = Path(__file__).resolve().parents[1] / "docs" / "dashboard-explainer.md"
STANDARD_CHECKS_PATH = Path(__file__).resolve().parents[1] / "data" / "gold" / "dclo_standard_checks_summary.json"
CAUSAL_COEF_PATH = Path(__file__).resolve().parents[1] / "data" / "gold" / "dclo_causal_coefficients.csv"
CAUSAL_FIT_PATH = Path(__file__).resolve().parents[1] / "data" / "gold" / "dclo_causal_model_fit.csv"
RANK_STABILITY_PATH = Path(__file__).resolve().parents[1] / "data" / "gold" / "dclo_rank_stability.csv"
METHOD_COMPARISON_PATH = Path(__file__).resolve().parents[1] / "data" / "gold" / "dclo_method_comparison.csv"
STATE_MANIFEST_PATH = Path(__file__).resolve().parents[1] / "data" / "gold" / "dclo_state_year_audit_manifest.json"
COUNTRY_MANIFEST_PATH = Path(__file__).resolve().parents[1] / "data" / "gold" / "dclo_country_year_audit_manifest.json"
CAUSAL_MANIFEST_PATH = Path(__file__).resolve().parents[1] / "data" / "gold" / "dclo_causal_panel_audit_manifest.json"
STATE_VERIFICATION_PATH = Path(__file__).resolve().parents[1] / "data" / "gold" / "dclo_state_year_verification.json"
COUNTRY_VERIFICATION_PATH = Path(__file__).resolve().parents[1] / "data" / "gold" / "dclo_country_year_verification.json"
LONGITUDINAL_DB_PATH = Path(__file__).resolve().parents[1] / "data" / "gold" / "dclo_longitudinal.db"
STATE_DOMAIN_SCORE_COLUMNS = [
    "ACC_score",
    "SKL_score",
    "SRV_score",
    "AGR_score",
    "ECO_score",
    "OUT_score",
]
COUNTRY_DOMAIN_SCORE_COLUMNS = STATE_DOMAIN_SCORE_COLUMNS
STATE_CENTROIDS = {
    "Andaman And Nicobar Islands": (11.7401, 92.6586),
    "Andhra Pradesh": (15.9129, 79.7400),
    "Arunachal Pradesh": (28.2180, 94.7278),
    "Assam": (26.2006, 92.9376),
    "Bihar": (25.0961, 85.3131),
    "Chandigarh": (30.7333, 76.7794),
    "Chhattisgarh": (21.2787, 81.8661),
    "Dadra And Nagar Haveli": (20.1809, 73.0169),
    "Delhi": (28.7041, 77.1025),
    "Goa": (15.2993, 74.1240),
    "Gujarat": (22.2587, 71.1924),
    "Haryana": (29.0588, 76.0856),
    "Himachal Pradesh": (31.1048, 77.1734),
    "Jammu And Kashmir": (33.7782, 76.5762),
    "Jharkhand": (23.6102, 85.2799),
    "Karnataka": (15.3173, 75.7139),
    "Kerala": (10.8505, 76.2711),
    "Ladakh": (34.1526, 77.5770),
    "Madhya Pradesh": (22.9734, 78.6569),
    "Maharashtra": (19.7515, 75.7139),
    "Manipur": (24.6637, 93.9063),
    "Meghalaya": (25.4670, 91.3662),
    "Mizoram": (23.1645, 92.9376),
    "Nagaland": (26.1584, 94.5624),
    "Odisha": (20.9517, 85.0985),
    "Puducherry": (11.9416, 79.8083),
    "Punjab": (31.1471, 75.3412),
    "Rajasthan": (27.0238, 74.2179),
    "Sikkim": (27.5330, 88.5122),
    "Tamil Nadu": (11.1271, 78.6569),
    "The Dadra And Nagar Haveli And Daman And Diu": (20.3974, 72.8328),
    "Tripura": (23.9408, 91.9882),
    "Uttar Pradesh": (26.8467, 80.9462),
    "Uttarakhand": (30.0668, 79.0193),
    "West Bengal": (22.9868, 87.8550),
}


@st.cache_data
def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset: {path}")
    df = pd.read_csv(path)
    if "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    for col in ["DCLO_score", "DCLO_score_context_adjusted"] + STATE_DOMAIN_SCORE_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


@st.cache_data
def load_explainer(path: Path) -> str:
    if not path.exists():
        return "Explainer file not found."
    return path.read_text(encoding="utf-8")


@st.cache_data
def load_standard_checks(path: Path) -> dict:
    if not path.exists():
        return {"overall_passed": None, "state_track": {}, "country_track": {}, "causal_track": {}}
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


@st.cache_data
def load_optional_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data
def load_audit_manifest(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (json.JSONDecodeError, OSError):
        return {}


@st.cache_data
def load_verification_report(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (json.JSONDecodeError, OSError):
        return {}


def render_provenance(manifest: dict, verification: dict, track_name: str) -> None:
    """Render audit provenance and data verification details."""
    if not manifest:
        st.info(f"No audit manifest available for {track_name}.")
        return

    st.subheader("Pipeline Run Metadata")
    col1, col2, col3 = st.columns(3)
    col1.metric("Pipeline", manifest.get("pipeline", "n/a"))
    col2.metric("Run ID", manifest.get("run_id", "n/a"))
    col3.metric("Status", manifest.get("status", "n/a"))

    env = manifest.get("environment", {})
    st.markdown(f"**Git Commit:** `{env.get('git_commit', 'not recorded')}`")
    st.markdown(f"**Python:** `{env.get('python_version', 'n/a')[:60]}`")
    st.markdown(f"**Platform:** `{env.get('platform', 'n/a')}`")

    packages = env.get("packages", {})
    if packages:
        pkg_str = ", ".join(f"{k} {v}" for k, v in sorted(packages.items()))
        st.markdown(f"**Key packages:** {pkg_str}")

    # Input file checksums
    inputs = manifest.get("inputs", {})
    if inputs:
        st.subheader("Input File Integrity (SHA-256)")
        input_rows = []
        for name, info in inputs.items():
            input_rows.append({
                "Source": name,
                "Exists": info.get("exists", False),
                "SHA-256": info.get("sha256", "n/a")[:16] + "..." if info.get("sha256") else "n/a",
                "Size (bytes)": info.get("size_bytes", "n/a"),
            })
        st.dataframe(pd.DataFrame(input_rows), width="stretch")

    # Pipeline stages
    stages = manifest.get("stages", [])
    if stages:
        st.subheader("Pipeline Stage Accounting")
        stage_rows = []
        for s in stages:
            stage_rows.append({
                "Stage": s.get("stage", ""),
                "Rows In": s.get("rows_in", ""),
                "Rows Out": s.get("rows_out", ""),
                "Dropped": s.get("rows_dropped", 0),
                "Notes": s.get("notes", "")[:80],
            })
        st.dataframe(pd.DataFrame(stage_rows), width="stretch")

        # Flag accounting warnings
        warnings = [s for s in stages if "accounting_warning" in s]
        if warnings:
            st.warning(f"{len(warnings)} stage(s) have row-accounting discrepancies.")

    # Output checksums
    outputs = manifest.get("outputs", {})
    if outputs:
        st.subheader("Output File Integrity (SHA-256)")
        out_rows = []
        for name, info in outputs.items():
            out_rows.append({
                "Output": name,
                "SHA-256": info.get("sha256", "n/a")[:16] + "..." if info.get("sha256") else "n/a",
                "Size (bytes)": info.get("size_bytes", "n/a"),
            })
        st.dataframe(pd.DataFrame(out_rows), width="stretch")

    # Analytical parameters
    params = manifest.get("parameters", {})
    if params:
        st.subheader("Analytical Parameters")
        for key, value in params.items():
            if isinstance(value, (dict, list)):
                st.markdown(f"**{key}:** `{json.dumps(value)[:100]}`")
            else:
                st.markdown(f"**{key}:** `{value}`")

    # Random seeds
    seeds = manifest.get("random_seeds", {})
    if seeds:
        st.subheader("Random Seeds (Reproducibility)")
        for name, seed in seeds.items():
            st.markdown(f"- **{name}:** `{seed}`")

    # Verification report
    if verification:
        st.subheader("Data Verification Results")
        for section_name, section in verification.items():
            if isinstance(section, dict) and "checks_run" in section:
                passed = section.get("passed", False)
                icon = "PASS" if passed else "ISSUES"
                st.markdown(
                    f"**{section_name}:** {icon} "
                    f"({section.get('checks_passed', 0)}/{section.get('checks_run', 0)} checks passed)"
                )
                issues = section.get("issues", [])
                if issues:
                    for issue in issues[:10]:
                        severity = issue.get("severity", "info") if isinstance(issue, dict) else "info"
                        msg = issue.get("message", str(issue)) if isinstance(issue, dict) else str(issue)
                        st.markdown(f"  - [{severity}] {msg}")


def render_kpis(df_year: pd.DataFrame, full_df: pd.DataFrame, selected_year: int, entity_col: str, score_col: str) -> None:
    mean_score = df_year[score_col].mean(skipna=True)
    entity_count = df_year[entity_col].nunique()
    top_entity = df_year.sort_values(score_col, ascending=False).iloc[0][entity_col]
    bottom_entity = df_year.sort_values(score_col, ascending=True).iloc[0][entity_col]

    prev_year_df = full_df[full_df["year"] == (selected_year - 1)]
    prev_mean = prev_year_df[score_col].mean(skipna=True) if not prev_year_df.empty else None
    delta = None if prev_mean is None or pd.isna(prev_mean) else mean_score - prev_mean

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mean DCLO", f"{mean_score:.3f}", None if delta is None else f"{delta:.3f}")
    c2.metric("Entities Covered", f"{entity_count}")
    c3.metric("Top Entity", f"{top_entity}")
    c4.metric("Bottom Entity", f"{bottom_entity}")


def render_ranking(df_year: pd.DataFrame, entity_col: str, score_col: str, title: str) -> None:
    rank_df = df_year.sort_values(score_col, ascending=False)
    fig = px.bar(
        rank_df,
        x=score_col,
        y=entity_col,
        orientation="h",
        title=title,
        labels={entity_col: "Entity", score_col: "DCLO Score"},
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=700)
    st.plotly_chart(fig, use_container_width=True)


def render_trends(full_df: pd.DataFrame, selected_entities: list[str], entity_col: str, score_col: str, title: str) -> None:
    if not selected_entities:
        return
    trend_df = full_df[full_df[entity_col].isin(selected_entities)].copy()
    trend_df = trend_df.dropna(subset=[score_col, "year"])
    fig = px.line(
        trend_df,
        x="year",
        y=score_col,
        color=entity_col,
        markers=True,
        title=title,
        labels={"year": "Year", score_col: "DCLO Score", entity_col: "Entity"},
    )
    st.plotly_chart(fig, use_container_width=True)


def render_domain_heatmap(df_year: pd.DataFrame, entity_col: str, domain_cols: list[str], title: str) -> None:
    available_cols = [col for col in domain_cols if col in df_year.columns]
    if not available_cols:
        st.info("No domain score columns found for heatmap.")
        return

    heatmap_df = (
        df_year[[entity_col] + available_cols]
        .dropna(subset=[entity_col])
        .set_index(entity_col)
        .sort_index()
    )
    fig = px.imshow(
        heatmap_df,
        aspect="auto",
        color_continuous_scale="RdYlGn",
        title=title,
        labels={"x": "Domain", "y": "Entity", "color": "Score"},
    )
    st.plotly_chart(fig, use_container_width=True)


def render_domain_profile(df_year: pd.DataFrame, selected_entity: str, entity_col: str, domain_cols: list[str]) -> None:
    row_df = df_year[df_year[entity_col] == selected_entity]
    if row_df.empty:
        st.info("No data available for selected entity.")
        return

    row = row_df.iloc[0]
    domain_vals = {col: row.get(col, pd.NA) for col in domain_cols}
    profile_df = pd.DataFrame({"domain": list(domain_vals.keys()), "score": list(domain_vals.values())})
    profile_df = profile_df.dropna(subset=["score"])
    if profile_df.empty:
        st.info("No domain scores available for selected entity.")
        return

    fig = px.bar(
        profile_df,
        x="domain",
        y="score",
        title=f"Domain Profile: {selected_entity}",
        labels={"domain": "Domain", "score": "Score"},
    )
    st.plotly_chart(fig, use_container_width=True)


def render_country_coverage_diagnostics(df_year: pd.DataFrame) -> None:
    required = {"coverage_ratio", "imputed_indicator_points", "observed_indicator_points", "model_trust_tier"}
    if not required.issubset(df_year.columns):
        return

    st.subheader("Coverage Diagnostics")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Mean observed coverage", f"{df_year['coverage_ratio'].mean() * 100:.1f}%")
    col2.metric("Median observed indicators", f"{df_year['observed_indicator_points'].median():.0f}")
    col3.metric("Median imputed indicators", f"{df_year['imputed_indicator_points'].median():.0f}")
    high_share = (df_year["model_trust_tier"].astype(str) == "High").mean()
    col4.metric("High-trust rows", f"{high_share * 100:.1f}%")

    trust_counts = (
        df_year["model_trust_tier"]
        .value_counts(dropna=False)
        .rename_axis("Trust tier")
        .reset_index(name="Rows")
    )
    st.dataframe(trust_counts, width="stretch")


def render_state_map(df_year: pd.DataFrame) -> None:
    map_df = df_year.copy()
    map_df["lat"] = map_df["state_name"].map(lambda x: STATE_CENTROIDS.get(x, (None, None))[0])
    map_df["lon"] = map_df["state_name"].map(lambda x: STATE_CENTROIDS.get(x, (None, None))[1])
    map_df = map_df.dropna(subset=["lat", "lon", "DCLO_score"])
    if map_df.empty:
        st.info("Map not available: no centroid mapping found for current states.")
        return
    # Plotly marker size must be non-negative; shift scores into a positive range.
    min_score = map_df["DCLO_score"].min()
    map_df["map_size"] = (map_df["DCLO_score"] - min_score) + 0.05

    fig = px.scatter_geo(
        map_df,
        lat="lat",
        lon="lon",
        color="DCLO_score",
        size="map_size",
        hover_name="state_name",
        color_continuous_scale="Viridis",
        projection="natural earth",
        title="India State Map (DCLO by State)",
    )
    fig.update_geos(scope="asia", lataxis_range=[6, 38], lonaxis_range=[67, 98], showcountries=True)
    fig.update_layout(height=650)
    st.plotly_chart(fig, use_container_width=True)


def render_country_map(df_year: pd.DataFrame, score_col: str) -> None:
    map_df = df_year.dropna(subset=["economy", score_col]).copy()
    if map_df.empty:
        st.info("No country data available for map.")
        return
    if "iso3c" in map_df.columns:
        location_col = "iso3c"
        location_mode = "ISO-3"
    else:
        location_col = "economy"
        location_mode = "country names"
    fig = px.choropleth(
        map_df,
        locations=location_col,
        locationmode=location_mode,
        color=score_col,
        hover_name="economy",
        color_continuous_scale="Viridis",
        title="Country Comparison Map (DCLO)",
    )
    fig.update_layout(height=650)
    st.plotly_chart(fig, use_container_width=True)


def render_downloads(df_year: pd.DataFrame, score_col: str, entity_col: str, suffix: str) -> None:
    sorted_df = df_year.sort_values(score_col, ascending=False)
    csv_bytes = sorted_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download filtered data (CSV)",
        data=csv_bytes,
        file_name=f"dclo_filtered_{suffix}.csv",
        mime="text/csv",
    )

    summary_cols = [entity_col, "year", score_col]
    summary = sorted_df[summary_cols + [c for c in STATE_DOMAIN_SCORE_COLUMNS if c in sorted_df.columns]]
    summary_bytes = summary.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download score summary (CSV)",
        data=summary_bytes,
        file_name=f"dclo_score_summary_{suffix}.csv",
        mime="text/csv",
    )


def render_causal_forest(coeff_df: pd.DataFrame) -> None:
    if coeff_df.empty:
        st.info("Causal coefficient output not available.")
        return
    use = coeff_df.copy()
    use["label"] = use["spec_id"].astype(str) + " | " + use["predictor"].astype(str)
    use = use.sort_values(["spec_kind", "spec_id", "predictor"])
    fig = px.scatter(
        use,
        x="coef",
        y="label",
        color="spec_kind",
        error_x=1.96 * pd.to_numeric(use["std_error"], errors="coerce"),
        title="Causal Evidence: Coefficients with 95% intervals",
        labels={"coef": "Estimated effect", "label": "Specification / Predictor", "spec_kind": "Spec type"},
    )
    fig.add_vline(x=0.0, line_dash="dash")
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)


def render_model_fit_table(fit_df: pd.DataFrame) -> None:
    if fit_df.empty:
        st.info("Causal model-fit output not available.")
        return
    show_cols = [
        "spec_kind",
        "spec_id",
        "outcome",
        "lag",
        "lead",
        "n_obs",
        "n_entities",
        "n_years",
        "r2_within",
        "residual_std",
    ]
    available = [col for col in show_cols if col in fit_df.columns]
    st.dataframe(fit_df[available].sort_values(["spec_kind", "spec_id"]), width="stretch")


def render_rank_stability(stability_df: pd.DataFrame, selected_year: int) -> None:
    if stability_df.empty:
        st.info("Rank stability output not available.")
        return
    if "year" not in stability_df.columns:
        st.info("Rank stability file missing `year` column.")
        return
    use = stability_df[stability_df["year"] == selected_year].copy()
    if use.empty:
        st.info(f"No stability rows available for {selected_year}.")
        return
    top = use.sort_values("top_k_freq", ascending=False).head(20)
    fig = px.bar(
        top.sort_values("top_k_freq", ascending=True),
        x="top_k_freq",
        y="economy",
        orientation="h",
        title=f"Top-K Stability Frequency ({selected_year})",
        labels={"top_k_freq": "Share of perturbation draws in top-K", "economy": "Economy"},
    )
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)

    spread = use.sort_values("sd_rank", ascending=False).head(20)
    fig2 = px.bar(
        spread.sort_values("sd_rank", ascending=True),
        x="sd_rank",
        y="economy",
        orientation="h",
        title=f"Rank Uncertainty (SD of Rank, {selected_year})",
        labels={"sd_rank": "Rank SD across perturbations", "economy": "Economy"},
    )
    fig2.update_layout(height=600)
    st.plotly_chart(fig2, use_container_width=True)


def render_method_comparison(method_df: pd.DataFrame) -> None:
    if method_df.empty:
        st.info("Method-comparison output not available.")
        return
    value_cols = [
        "rho_baseline_vs_weighted",
        "rho_baseline_vs_causal",
        "rho_weighted_vs_causal",
    ]
    available = [col for col in value_cols if col in method_df.columns]
    if not available:
        st.info("Method-comparison columns missing.")
        return
    long = method_df.melt(id_vars=["year"], value_vars=available, var_name="metric", value_name="rho")
    fig = px.line(
        long,
        x="year",
        y="rho",
        color="metric",
        markers=True,
        title="Method Agreement Over Time (Spearman rho)",
        labels={"year": "Year", "rho": "Spearman rho", "metric": "Comparison"},
    )
    fig.update_yaxes(range=[-1.0, 1.0])
    st.plotly_chart(fig, use_container_width=True)


def simulate_primary_event(event_type: str) -> None:
    import json
    import random
    import datetime
    import hashlib
    import uuid
    from pathlib import Path

    ROOT = Path(__file__).resolve().parents[1]
    PRIMARY_RESPONSES_PATH = ROOT / "data" / "primary" / "survey_responses.jsonl"
    PRIMARY_EVENTS_PATH = ROOT / "data" / "primary" / "survey_events.jsonl"
    PRIMARY_ELIGIBILITY_PATH = ROOT / "data" / "gold" / "survey_incentive_eligibility.csv"
    PRIMARY_EXPORT_PATH = ROOT / "data" / "gold" / "dpi_dclo_primary_export.csv"

    now_str = datetime.datetime.now(datetime.timezone.utc).isoformat()
    phone = "+91" + "".join(str(random.randint(0, 9)) for _ in range(10))
    phone_hash = hashlib.sha256(phone.encode()).hexdigest()
    sim_id = str(uuid.uuid4())

    if event_type == "whatsapp":
        wa_event = {
            "event_type": "twilio_whatsapp_message_received",
            "timestamp_utc": now_str,
            "campaign_mode": "prototype",
            "message_sid": "SM" + uuid.uuid4().hex[:32],
            "phone_hash": phone_hash,
            "from_number": phone,
            "status": "received",
            "body": random.choice(["नमस्ते DCLO", "प्रणाम", "Yes, start survey", "Option A", "Option B"]),
            "run_id": "20260602T_SIM"
        }
        with open(PRIMARY_EVENTS_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(wa_event) + "\n")

    elif event_type == "call":
        call_sid = "CA" + uuid.uuid4().hex[:32]
        call_event = {
            "event_type": "twilio_call_status_changed",
            "timestamp_utc": now_str,
            "campaign_mode": "prototype",
            "call_sid": call_sid,
            "phone_hash": phone_hash,
            "from_number": phone,
            "status": "completed",
            "steps_answered": random.randint(10, 17),
            "run_id": "20260602T_SIM"
        }
        with open(PRIMARY_EVENTS_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(call_event) + "\n")
        
        response = {
            "response_id": sim_id,
            "timestamp_utc": now_str,
            "campaign_mode": "prototype",
            "survey_mode": "voice",
            "phone_hash": phone_hash,
            "language": random.choice(["Maithili", "Hindi"]),
            "acc_use_cell": 1,
            "skl_internet_skills": random.randint(1, 5),
            "srv_mobile_banking": random.choice([1, 0]),
            "dignity_score": random.randint(4, 10),
            "climate_info_seeking": random.randint(1, 5),
            "climate_self_efficacy": random.randint(1, 5)
        }
        with open(PRIMARY_RESPONSES_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(response) + "\n")

    elif event_type == "payout":
        elig_row = f"\n{sim_id},{now_str},prototype,whatsapp,{phone_hash},complete,true,eligible_full_completion,200"
        with open(PRIMARY_ELIGIBILITY_PATH, "a", encoding="utf-8") as f:
            f.write(elig_row)
        
        dpi_val = round(random.uniform(3.0, 5.0), 2)
        acc = random.randint(3, 5)
        skl = random.randint(3, 5)
        eco = random.randint(3, 5)
        srv = random.randint(3, 5)
        agr = random.randint(3, 5)
        out = random.randint(3, 5)
        dclo_pri = round((acc + skl + eco + srv + agr + out) / 6.0, 2)
        
        export_row = f"\n{sim_id},{now_str},prototype,whatsapp,India,farmer,NGO,{dpi_val},{acc},{skl},{eco},{srv},{agr},{out},{dclo_pri},complete,True,200"
        with open(PRIMARY_EXPORT_PATH, "a", encoding="utf-8") as f:
            f.write(export_row)


def main() -> None:
    st.title("PCS & DCLO Research Dashboard")
    st.caption("Public Capability Stack (PCS) & Digital Capability for Life Outcomes (DCLO) Index (Dual-Track: India State-Year + Country-Year)")

    try:
        state_df = load_data(STATE_DATA_PATH)
    except Exception as exc:
        st.error(f"Failed to load state dataset: {exc}")
        st.stop()
    try:
        country_df = load_data(COUNTRY_DATA_PATH)
    except Exception:
        country_df = pd.DataFrame()
    country_api_df = load_optional_csv(COUNTRY_API_DATA_PATH)
    if not country_api_df.empty and "year" in country_api_df.columns:
        country_api_df["year"] = pd.to_numeric(country_api_df["year"], errors="coerce").astype("Int64")
        for col in ["DCLO_score", "DCLO_score_confidence_weighted"] + COUNTRY_DOMAIN_SCORE_COLUMNS:
            if col in country_api_df.columns:
                country_api_df[col] = pd.to_numeric(country_api_df[col], errors="coerce")

    explainer_text = load_explainer(EXPLAINER_PATH)
    checks = load_standard_checks(STANDARD_CHECKS_PATH)
    causal_coef_df = load_optional_csv(CAUSAL_COEF_PATH)
    causal_fit_df = load_optional_csv(CAUSAL_FIT_PATH)
    rank_stability_df = load_optional_csv(RANK_STABILITY_PATH)
    method_comparison_df = load_optional_csv(METHOD_COMPARISON_PATH)
    state_manifest = load_audit_manifest(STATE_MANIFEST_PATH)
    country_manifest = load_audit_manifest(COUNTRY_MANIFEST_PATH)
    causal_manifest = load_audit_manifest(CAUSAL_MANIFEST_PATH)
    state_verification = load_verification_report(STATE_VERIFICATION_PATH)
    country_verification = load_verification_report(COUNTRY_VERIFICATION_PATH)
    required_columns = {"state_name", "year", "DCLO_score"}
    if not required_columns.issubset(state_df.columns):
        st.error("State dataset missing required columns: state_name, year, DCLO_score")
        st.stop()

    with st.sidebar:
        track = st.selectbox("Track", ["India State-Year", "Country-Year Comparative"])
        st.header("Filters")
        if track == "India State-Year":
            years = sorted([int(year) for year in state_df["year"].dropna().unique().tolist()])
            states = sorted(state_df["state_name"].dropna().astype(str).unique().tolist())
            selected_year = st.selectbox("Year", years, index=len(years) - 1)
            selected_entities = st.multiselect("States for trend chart", states, default=states[:5])
            selected_profile_entity = st.selectbox("State for domain profile", states)
            use_context_adjusted = st.checkbox(
                "Use context-adjusted score (DPI country context)",
                value="DCLO_score_context_adjusted" in state_df.columns,
            )
            score_col = "DCLO_score_context_adjusted" if use_context_adjusted and "DCLO_score_context_adjusted" in state_df.columns else "DCLO_score"
            working_df = state_df.copy()
            entity_col = "state_name"
            domain_cols = STATE_DOMAIN_SCORE_COLUMNS
        else:
            country_source = st.selectbox(
                "Country data source",
                ["API (RestCountries + World Bank)", "DPI Comparative (legacy/high-imputation)"],
            )
            source_df = country_api_df if country_source.startswith("API") else country_df
            if source_df.empty or not {"economy", "year", "DCLO_score"}.issubset(source_df.columns):
                st.error("Country-year dataset not available or missing required columns.")
                st.stop()
            years = sorted([int(year) for year in source_df["year"].dropna().unique().tolist()])
            economies = sorted(source_df["economy"].dropna().astype(str).unique().tolist())
            selected_year = st.selectbox("Year", years, index=len(years) - 1)
            selected_entities = st.multiselect("Countries for trend chart", economies, default=["India"] if "India" in economies else economies[:5])
            selected_profile_entity = st.selectbox("Country for domain profile", economies, index=economies.index("India") if "India" in economies else 0)
            score_mode = st.selectbox("Score mode", ["Baseline", "Confidence-weighted"])
            score_col = (
                "DCLO_score_confidence_weighted"
                if score_mode == "Confidence-weighted" and "DCLO_score_confidence_weighted" in source_df.columns
                else "DCLO_score"
            )
            trust_filter = st.multiselect(
                "Trust tier filter",
                options=["High", "Medium", "Low"],
                default=["High", "Medium", "Low"],
            )
            working_df = source_df.copy()
            if "model_trust_tier" in working_df.columns:
                working_df = working_df[working_df["model_trust_tier"].isin(trust_filter)].copy()
            entity_col = "economy"
            domain_cols = COUNTRY_DOMAIN_SCORE_COLUMNS

        st.divider()
        st.header("Model QA")
        overall_passed = checks.get("overall_passed")
        if overall_passed is True:
            st.success("Standard checks: PASS")
        elif overall_passed is False:
            st.error("Standard checks: FAIL")
        else:
            st.info("Standard checks: not available")

        if track == "India State-Year":
            issues = checks.get("state_track", {}).get("issues", [])
        else:
            issues = checks.get("country_track", {}).get("issues", [])
        if issues:
            st.caption("Active issues")
            for issue in issues[:5]:
                st.write(f"- {issue}")
        if track != "India State-Year":
            c_issues = checks.get("causal_track", {}).get("issues", [])
            if c_issues:
                st.caption("Causal-track issues")
                for issue in c_issues[:5]:
                    st.write(f"- {issue}")
        st.divider()
        st.header("Methodology")
        if track == "India State-Year":
            unit_text = "state-year (India track)"
        elif country_source.startswith("API"):
            unit_text = "country-year (API track: RestCountries + World Bank)"
        else:
            unit_text = "country-year (legacy DPI comparative track)"
        st.markdown(
            "\n".join(
                [
                    "- Construct: **DCLO (Digital Capability for Life Outcomes)**",
                    f"- Unit: **{unit_text}**",
                    "- Measurement: **formative multidimensional composite**",
                    "- Causal layer (country track): **panel model with lagged predictors**",
                    "- Interpretation: prioritize **effect sizes and uncertainty** over raw ranks",
                ]
            )
        )
        with st.expander("Dashboard explainer", expanded=False):
            st.markdown(explainer_text)

    df_year = working_df[working_df["year"] == selected_year].copy().dropna(subset=[score_col])
    if df_year.empty:
        st.warning(f"No DCLO records for {selected_year}.")
        st.stop()
    if checks.get("overall_passed") is False:
        st.warning(
            "Model QA checks are currently failing for at least one track. "
            "Use rankings with caution and review `data/gold/dclo_standard_checks_summary.md`."
        )

    tab_paper1, tab_paper2, tab_paper3, tab_me, tab_qa = st.tabs([
        "Paper 1: Measurement & Formative Construct",
        "Paper 2: Causal Panel & Robustness Diagnostics",
        "Paper 3: Situated Capabilities & Field Survey",
        "PCS Longitudinal M&E (PhD Panel)",
        "Model QA & Reproducibility"
    ])

    with tab_paper1:
        st.markdown(
            """
            ### Paper 1: Measuring the Public Capability Stack (PCS) via the DCLO Index
            Following **Amartya Sen's Capability Approach** (Sen 1985, 1999), digital capability is modeled as a 
            formative composite construct that converts digital resources into valuable life outcomes, mediated by 
            individual, social, and environmental conversion factors. The formative **DCLO Index** operationalizes and measures the layers of the **Public Capability Stack (PCS)**:
            
            * **Access** $\\rightarrow$ *Layer 1: Digital Resources* (DPI platforms, standards)
            * **Skills** $\\rightarrow$ *Layer 3: Conversion Factors* (literacy, agency)
            * **Services** $\\rightarrow$ *Layer 2: Demand Signals* (service engagement)
            * **Agency/Trust** $\\rightarrow$ *Layer 4: Governance & Safeguards* (safeguards, language justice)
            * **Economy/Eco-Sust** $\\rightarrow$ *Layer 5: Public Capabilities* (accountable operational budget)
            * **Outcomes** $\\rightarrow$ *Layer 6: Life Outcomes* (realized capabilities/freedoms)
            
            #### Mathematical Formulation
            Each indicator $X_{i}$ is standard-normalized (winsorised at 5th/95th percentiles to follow OECD 2008 handbook guidelines):
            $$Z_{i} = \\frac{X_{i} - \\mu_{i}}{\\sigma_{i}}$$
            
            The **Domain Score** $S_d$ for each of the 6 pillars is:
            $$S_d = \\frac{1}{|I_d|} \\sum_{i \\in I_d} Z_{i}$$
            
            The **Formative DCLO Composite Index** is computed as the equal-weighted mean of domain scores:
            $$DCLO = \\frac{1}{|D|} \\sum_{d \\in D} S_d$$
            
            When data availability is unequal, the **Confidence-Weighted Composite** DCLO is computed using coverage-aware weights $w_d$:
            $$DCLO_{CW} = \\frac{\\sum_{d \\in D} w_d S_d}{\\sum_{d \\in D} w_d}$$
            """
        )
        render_kpis(df_year, working_df, selected_year, entity_col=entity_col, score_col=score_col)
        col1, col2 = st.columns([1.25, 1])
        with col1:
            rank_title = "State Ranking by DCLO Score" if track == "India State-Year" else "Country Ranking by DCLO Score"
            render_ranking(df_year, entity_col=entity_col, score_col=score_col, title=rank_title)
        with col2:
            render_domain_profile(df_year, selected_profile_entity, entity_col=entity_col, domain_cols=domain_cols)

        if track == "India State-Year":
            render_state_map(df_year)
            render_trends(
                working_df,
                selected_entities,
                entity_col=entity_col,
                score_col=score_col,
                title="DCLO Trend Over Time (Selected States)",
            )
            render_domain_heatmap(df_year, entity_col=entity_col, domain_cols=domain_cols, title="Domain Score Heatmap by State")
            render_downloads(df_year, score_col=score_col, entity_col=entity_col, suffix="state_year")
        else:
            render_country_map(df_year, score_col=score_col)
            if country_source.startswith("API"):
                render_country_coverage_diagnostics(df_year)
            render_trends(
                working_df,
                selected_entities,
                entity_col=entity_col,
                score_col=score_col,
                title="DCLO Trend Over Time (Selected Countries)",
            )
            render_domain_heatmap(df_year, entity_col=entity_col, domain_cols=domain_cols, title="Domain Score Heatmap by Country")
            render_downloads(df_year, score_col=score_col, entity_col=entity_col, suffix="country_year")

    with tab_paper2:
        st.markdown(
            """
            ### Paper 2: Econometric Causal Evidence & Robustness
            This paper evaluates whether digital capabilities drive real-world life outcomes by estimating a 
            **two-way fixed effects (TWFE) lagged panel regression model** (Wooldridge 2010) of the **DCLO Index** (which operationalizes the layers of the **Public Capability Stack**):
            
            $$Y_{it} = \\beta_1 DCLO_{i, t-1} + \\mathbf{X}'_{i, t-1} \\boldsymbol{\\gamma} + \\alpha_i + \\delta_t + \\epsilon_{it}$$
            
            Where:
            * $Y_{it}$ is the real-world **Life Outcomes index** (`OUT_score`), capturing health, agricultural, and economic outcomes (*PCS Layer 6*).
            * $DCLO_{i, t-1}$ is the lagged composite capability score (formative composite of *PCS Layers 1-5*).
            * $\\mathbf{X}'_{i, t-1}$ represents a vector of lagged controls.
            * $\\alpha_i$ absorbs time-invariant country-specific confounders (Entity Fixed Effects).
            * $\\delta_t$ absorbs time-varying macro shocks (Time Fixed Effects).
            """
        )
        if track == "India State-Year":
            st.info("Causal evidence and robustness diagnostics are currently available for the country-year comparative track.")
        else:
            st.caption(
                "Interpret coefficients as conditional associations from the configured panel model. "
                "They are not guaranteed causal unless identification assumptions hold."
            )
            render_causal_forest(causal_coef_df)
            st.subheader("Specification Fit")
            render_model_fit_table(causal_fit_df)
            if not causal_coef_df.empty:
                pvals = pd.to_numeric(causal_coef_df["p_value_norm_approx"], errors="coerce")
                sig = causal_coef_df[pvals < 0.05].copy()
                st.subheader("Statistically Significant Terms (p < 0.05)")
                st.dataframe(sig.sort_values(["spec_kind", "spec_id", "p_value_norm_approx"]), width="stretch")
            
            st.divider()
            st.subheader("Robustness & Rank Stability Diagnostics")
            render_method_comparison(method_comparison_df)
            render_rank_stability(rank_stability_df, selected_year=selected_year)
            st.subheader("Caveats and Identification Notes")
            st.markdown(
                "\n".join(
                    [
                        "- Fixed effects absorb time-invariant country confounders but not all time-varying confounders.",
                        "- Lag structure reduces simultaneity risk but does not fully prove causality.",
                        "- Placebo and robustness diagnostics should be reviewed before policy interpretation.",
                        "- Use this tab jointly with Model QA status from the sidebar.",
                    ]
                )
            )

    with tab_paper3:
        st.markdown(
            """
            ### Paper 3: Situated Capabilities & Field Survey (Twilio Mixed-Mode)
            Paper 3 explores **situated learning and linguistic justice** in low-resource environments by conducting 
            live mixed-mode phone surveys. Surveys are delivered in native local languages (e.g. Hindi, Maithili) 
            via SMS and WhatsApp automated channels, tracking user interaction journeys to assess localized agency.
            """
        )
        
        st.warning(
            "⚠️ **PRE-LAUNCH DEMO DATA PLATFORM**: The metrics shown below are currently loaded from high-fidelity pre-launch "
            "pilot/demo files (`data/primary/*.jsonl`). When the primary phone survey goes live, the Twilio webhook ingestion pipelines "
            "will automatically stream real-time response logs here in production.",
            icon="ℹ️"
        )
        
        with st.expander("🔴 Interactive Real-Life Survey Event Demo Panel", expanded=True):
            st.markdown(
                """
                To see how live primary data collection works, you can simulate an incoming survey call, WhatsApp event, or payout check in real-time. 
                Click any of the buttons below to write a simulated Twilio webhook event to the data files, which will instantly refresh the dashboard:
                """
            )
            demo_cols = st.columns(3)
            with demo_cols[0]:
                if st.button("Simulate WhatsApp Inbound Message", key="demo_wa"):
                    simulate_primary_event("whatsapp")
                    st.success("WhatsApp event appended! Refreshing...")
                    st.rerun()
            with demo_cols[1]:
                if st.button("Simulate Completed Phone Call", key="demo_call"):
                    simulate_primary_event("call")
                    st.success("Completed Voice call & response appended! Refreshing...")
                    st.rerun()
            with demo_cols[2]:
                if st.button("Simulate Payout Eligibility & Export", key="demo_payout"):
                    simulate_primary_event("payout")
                    st.success("Payout row & integrated export appended! Refreshing...")
                    st.rerun()
        # Load and render primary survey metrics
        PRIMARY_RESPONSES_PATH = Path(__file__).resolve().parents[1] / "data" / "primary" / "survey_responses.jsonl"
        PRIMARY_EVENTS_PATH = Path(__file__).resolve().parents[1] / "data" / "primary" / "survey_events.jsonl"
        PRIMARY_ELIGIBILITY_PATH = Path(__file__).resolve().parents[1] / "data" / "gold" / "survey_incentive_eligibility.csv"
        PRIMARY_EXPORT_PATH = Path(__file__).resolve().parents[1] / "data" / "gold" / "dpi_dclo_primary_export.csv"

        def load_jsonl(path: Path) -> pd.DataFrame:
            import json
            import os
            if not path.exists():
                return pd.DataFrame()
            rows = []
            with open(path, "r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rows.append(json.loads(line))
                    except Exception:
                        continue
            return pd.DataFrame(rows)

        responses = load_jsonl(PRIMARY_RESPONSES_PATH)
        events = load_jsonl(PRIMARY_EVENTS_PATH)
        eligibility = load_optional_csv(PRIMARY_ELIGIBILITY_PATH)
        exports = load_optional_csv(PRIMARY_EXPORT_PATH)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Field Responses", int(len(responses)))
        c2.metric("Interaction Events", int(len(events)))
        c3.metric("Validated Eligibility", int(len(eligibility)))
        c4.metric("DPI Primary Exports", int(len(exports)))

        if not eligibility.empty and "completion_status" in eligibility.columns:
            status_counts = eligibility["completion_status"].value_counts().rename_axis("status").reset_index(name="count")
            st.subheader("Field Survey Completion Status")
            st.bar_chart(status_counts.set_index("status"))

        if not eligibility.empty and "incentive_eligible" in eligibility.columns:
            eligible_count = int((eligibility["incentive_eligible"].astype(str).str.lower() == "true").sum())
            st.info(f"Verified Payout-Eligible Respondents (INR 200 Incentive): **{eligible_count}**")

        st.subheader("Latest Live Survey Events (Mixed-Mode Phone Channels)")
        if events.empty:
            st.warning("No live call events recorded in data/primary/survey_events.jsonl.")
        else:
            show_events = events.copy()
            if "timestamp_utc" in show_events.columns:
                show_events = show_events.sort_values("timestamp_utc", ascending=False)
            st.dataframe(show_events.head(15), width="stretch")

        st.subheader("PCS/DCLO Primary Integrated Export Preview")
        if exports.empty:
            st.warning("No export data available in gold/dpi_dclo_primary_export.csv.")
        else:
            st.dataframe(exports.head(15), width="stretch")

    with tab_me:
        st.markdown(
            """
            ### PCS Longitudinal Monitoring & Evaluation (PhD Panel)
            This tab provides real-time access to the panel database managed by the **Longitudinal Spatial Capability Agent**. 
            It tracks spatial accessibility decay and linguistic barriers (*PCS Layer 4: Governance & Safeguards*) and livelihood outcomes (*PCS Layer 6: Life Outcomes*) across 100 villages in Northern Bihar.
            """
        )
        
        if not LONGITUDINAL_DB_PATH.exists():
            st.info("Longitudinal SQLite database not found. Run the ingestion agent to populate panel records.")
        else:
            # 1. Load Data from SQLite
            conn = sqlite3.connect(str(LONGITUDINAL_DB_PATH))
            panel_df = pd.read_sql_query("SELECT * FROM daily_panel", conn)
            villages_df = pd.read_sql_query("SELECT * FROM villages", conn)
            conn.close()
            
            # Merge villages
            full_panel = pd.merge(panel_df, villages_df, on="village_id")
            full_panel['date_parsed'] = pd.to_datetime(full_panel['date'])
            
            # 2. Key KPIs
            num_villages = villages_df['village_id'].nunique()
            avg_dist = panel_df['dist_to_csc_km'].mean()
            avg_exclusion = panel_df['spatial_linguistic_exclusion'].mean()
            panel_corr = panel_df['spatial_linguistic_exclusion'].corr(panel_df['livelihood_outcome'])
            
            kpi1, kpi2, kpi3, kpi4 = st.columns(4)
            kpi1.metric("Villages Monitored", f"{num_villages}")
            kpi2.metric("Mean Distance to CSC", f"{avg_dist:.2f} km")
            kpi3.metric("Mean Spatial-Linguistic Exclusion", f"{avg_exclusion:.2f}")
            kpi4.metric("Exclusion vs. Livelihood Corr", f"{panel_corr:.4f}")
            
            # 3. Spatial Map with Time-Slider (Dignity & Access Evolution)
            st.subheader("I. Spatial-Linguistic Exclusion Zones Over Time")
            st.markdown("Use the slider below to observe how the rollout of Maithili-language interfaces on Day 15 (May 15, 2026) in Madhubani district shrinks the exclusion zones.")
            
            unique_dates = sorted(full_panel['date'].unique().tolist())
            selected_date_str = st.select_slider("Select Date", options=unique_dates, value=unique_dates[0])
            
            day_df = full_panel[full_panel['date'] == selected_date_str].copy()
            
            fig_map = px.scatter(
                day_df,
                x="longitude",
                y="latitude",
                color="spatial_linguistic_exclusion",
                size=day_df["dist_to_csc_km"].apply(lambda d: min(15.0, max(4.0, d))),
                color_continuous_scale="YlOrRd",
                range_color=[0, 10],
                hover_name="village_id",
                hover_data=["district_id", "primary_language", "dist_to_csc_km", "livelihood_outcome"],
                title=f"Village Exclusion Map: {selected_date_str}"
            )
            st.plotly_chart(fig_map, use_container_width=True)
            
            # 4. Interactive Event Study Chart
            st.subheader("II. Causal Event Study: Impact of Language Rollout")
            
            # Compute relative days
            event_date = pd.to_datetime("2026-05-15")
            full_panel['relative_day'] = (full_panel['date_parsed'] - event_date).dt.days
            
            event_df = full_panel[(full_panel['relative_day'] >= -10) & (full_panel['relative_day'] <= 10)]
            
            # Group
            grouped = event_df.groupby(['district_id', 'primary_language', 'relative_day'])['livelihood_outcome'].mean().reset_index()
            grouped['Group'] = grouped.apply(
                lambda r: 'Treated (MDB + Maithili)' if r['district_id'] == 'MDB' and r['primary_language'] == 'Maithili' else 'Control (All Others)',
                axis=1
            )
            grouped_agg = grouped.groupby(['Group', 'relative_day'])['livelihood_outcome'].mean().reset_index()
            
            fig_event = px.line(
                grouped_agg,
                x="relative_day",
                y="livelihood_outcome",
                color="Group",
                markers=True,
                color_discrete_map={'Treated (MDB + Maithili)': 'darkred', 'Control (All Others)': 'gray'},
                title="Event Study: Parallel Pre-Trends and Post-Intervention Divergence"
            )
            fig_event.add_vline(x=0, line_dash="dash", line_color="blue", annotation_text="Policy Rollout (May 15)")
            st.plotly_chart(fig_event, use_container_width=True)
            
            # 5. Econometric Regression Results
            st.subheader("III. Panel Econometrics Estimation Output (Two-Way Fixed Effects)")
            st.markdown("Estimated model: $Y_{it} = \\beta_1 \\text{Exclusion}_{it} + \\mathbf{X}'_{it}\\boldsymbol{\\gamma} + \\alpha_i + \\delta_t + \\epsilon_{it}$")
            
            report_path = Path(__file__).resolve().parents[1] / "data" / "gold" / "panel_regression_report.txt"
            if report_path.exists():
                st.code(report_path.read_text(), language="text")
            else:
                st.info("Run panel diagnostics script to generate econometric report.")
                
            # 6. Database Preview
            st.subheader("IV. Raw Panel Database Preview")
            st.dataframe(full_panel.drop(columns=['date_parsed']).sort_values(['village_id', 'date']).head(100), width="stretch")
            
            # Download button for panel csv
            panel_csv_bytes = full_panel.drop(columns=['date_parsed']).to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Full Panel Data (CSV)",
                data=panel_csv_bytes,
                file_name="dclo_longitudinal_panel.csv",
                mime="text/csv"
            )

    with tab_qa:
        st.markdown(
            """
            ### Model QA, Data Provenance & Transparency Audit
            This tab presents a comprehensive audit trail for reproducibility and transparency in construct aggregation 
            and causal panel estimation, following Christensen & Miguel (2018) principles.
            """
        )
        if track == "India State-Year":
            render_provenance(state_manifest, state_verification, "India State-Year")
        else:
            render_provenance(country_manifest, country_verification, "Country-Year")
            if causal_manifest:
                st.divider()
                st.header("Causal Panel Audit Trail")
                render_provenance(causal_manifest, {}, "Causal Panel")

    with st.expander("About this dashboard", expanded=False):
        st.markdown(explainer_text)

    st.subheader("Filtered Data Preview")
    st.dataframe(df_year.sort_values(score_col, ascending=False), width="stretch")


if __name__ == "__main__":
    main()
