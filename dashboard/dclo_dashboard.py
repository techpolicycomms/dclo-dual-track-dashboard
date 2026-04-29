from pathlib import Path

import json
import pandas as pd
import plotly.express as px
import streamlit as st


st.set_page_config(page_title="DCLO Dashboard", page_icon="📊", layout="wide")

STATE_DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "gold" / "dclo_state_year.csv"
COUNTRY_DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "gold" / "dclo_country_year.csv"
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

REPO_ROOT = Path(__file__).resolve().parents[1]
ACADEMIC_DIR = REPO_ROOT / "academic"
INDICATOR_METADATA_PATH = ACADEMIC_DIR / "indicator_metadata.json"
DAG_PATH = ACADEMIC_DIR / "dag.md"
THEORY_PATH = ACADEMIC_DIR / "02_theoretical_framework.md"
IDENTIFICATION_PATH = ACADEMIC_DIR / "04_identification_strategy_revised.md"
ROBUSTNESS_PATH = ACADEMIC_DIR / "05_robustness_protocol.md"
ETHICS_PATH = ACADEMIC_DIR / "06_ethics_and_responsible_use.md"
GAP_ANALYSIS_PATH = ACADEMIC_DIR / "01_gap_analysis.md"
KNOWN_ISSUES_PATH = ACADEMIC_DIR / "12_known_issues.md"
CHANGELOG_PATH = REPO_ROOT / "CHANGELOG.md"
CITATION_PATH = REPO_ROOT / "CITATION.cff"
EXTERNAL_RANKS_PATH = REPO_ROOT / "data" / "external" / "standard_family_ranks_2023.csv"

DASHBOARD_VERSION = "0.4.0"
DASHBOARD_BUILD_DATE = "2026-04-29"
DASHBOARD_LICENSE = "CC-BY-4.0"

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
        st.dataframe(pd.DataFrame(input_rows), use_container_width=True)

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
        st.dataframe(pd.DataFrame(stage_rows), use_container_width=True)

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
        st.dataframe(pd.DataFrame(out_rows), use_container_width=True)

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


def render_state_map(df_year: pd.DataFrame) -> None:
    # Build a clean, type-pure mini-DataFrame with only the columns plotly needs.
    # Avoids a plotly>=6.6 + narwhals issue where heterogeneous source DataFrames
    # cause narwhals to coerce float columns to object dtype, which the size validator rejects.
    rows = []
    for _, src in df_year.iterrows():
        state = src.get("state_name")
        score = src.get("DCLO_score")
        centroid = STATE_CENTROIDS.get(state)
        if state is None or centroid is None or pd.isna(score):
            continue
        rows.append({"state_name": str(state), "lat": float(centroid[0]),
                     "lon": float(centroid[1]), "DCLO_score": float(score)})
    if not rows:
        st.info("Map not available: no centroid mapping found for current states.")
        return
    map_df = pd.DataFrame(rows)
    min_score = float(map_df["DCLO_score"].min())
    map_df["map_size"] = (map_df["DCLO_score"] - min_score + 0.05).astype(float)

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
    if not map_df.empty:
        map_df = pd.DataFrame({
            "economy": map_df["economy"].astype(str),
            score_col: pd.to_numeric(map_df[score_col], errors="coerce").astype(float),
        }).dropna(subset=[score_col])
    if map_df.empty:
        st.info("No country data available for map.")
        return
    fig = px.choropleth(
        map_df,
        locations="economy",
        locationmode="country names",
        color=score_col,
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
    st.dataframe(fit_df[available].sort_values(["spec_kind", "spec_id"]), use_container_width=True)


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
    st.warning(
        "Read this as a **stability diagnostic**, not 'method agreement'. "
        "Baseline-vs-causal Spearman ρ flips from −0.55 (2016) to +0.76 (2024) in the current build — "
        "a sign that the causal-signal layer identifies something the index does not, "
        "which is the methodological centrepiece of the paper, not a routine plot."
    )
    long = method_df.melt(id_vars=["year"], value_vars=available, var_name="metric", value_name="rho")
    fig = px.line(
        long,
        x="year",
        y="rho",
        color="metric",
        markers=True,
        title="Method Stability Diagnostic Over Time (Spearman rho)",
        labels={"year": "Year", "rho": "Spearman rho", "metric": "Comparison"},
    )
    fig.update_yaxes(range=[-1.0, 1.0])
    st.plotly_chart(fig, use_container_width=True)


CONSTRUCT_VALIDITY_BANNER = (
    "**Known construct-validity gaps in the current build.** "
    "The country track's SRV outcome is built from WTO services-trade exports; "
    "AGR uses Worldwide Governance Indicator percentiles; SKL uses generic education enrolment; "
    "OUT includes population growth and agricultural-land share. "
    "These indicators do not target the capability layer the dashboard's name promises. "
    "The headline TWFE coefficient (DCLO_t-1 → SRV_t, β≈0.62) is largely structural overlap, "
    "not a capability-policy effect. The state track is effectively two cross-sections (NFHS-4 + NFHS-5). "
    "See `academic/01_gap_analysis.md` and `academic/03_indicator_validity_audit.md` for the full audit."
)


# ----------------------------------------------------------------------
# Scholarly-artefact helpers (added in v0.4.0 — see academic/13_dashboard_publication_plan.md).
# These functions implement the "dashboard-as-publication" upgrades: capability-layer overlay,
# DAG render, specification array, disagreement panel, evidence cards, and citation block.
# Each helper degrades gracefully when its data inputs are missing.
# ----------------------------------------------------------------------


@st.cache_data
def load_indicator_metadata(path: Path) -> dict:
    """Load the per-indicator metadata file produced by the academic audit.

    Returns a dict keyed by indicator code with {layer, source_agency, source_url,
    license, mechanism, construct_validity, replacement, name, domain}. Returns
    an empty schema-shaped dict when the file is missing.
    """
    if not path.exists():
        return {"layers": {}, "indicators": {}}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (json.JSONDecodeError, OSError):
        return {"layers": {}, "indicators": {}}


@st.cache_data
def load_text_file(path: Path) -> str:
    """Read a markdown / text file used in the Methods, Releases, and DAG renders."""
    if not path.exists():
        return f"_File not found: `{path.name}`._"
    return path.read_text(encoding="utf-8")


@st.cache_data
def load_external_ranks(path: Path) -> pd.DataFrame:
    """Load the bundled standard-family ranks (ITU IDI, EGDI, GTMI, NRI)."""
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _layer_chip(layer_id: str, metadata: dict) -> str:
    """Render an inline coloured chip for an indicator's capability layer."""
    layers = metadata.get("layers", {})
    info = layers.get(layer_id, {})
    label = info.get("label", layer_id or "—")
    color = info.get("color", "#9E9E9E")
    return (
        f"<span style='background-color:{color};color:white;"
        "padding:2px 8px;border-radius:10px;font-size:0.85em;'>"
        f"{label}</span>"
    )


def render_capability_layer_overlay(metadata: dict) -> None:
    """Render a legend explaining the four-layer scheme (resources / conversion factors /
    capabilities / functionings) used throughout the dashboard.
    """
    if not metadata.get("layers"):
        st.info("Indicator metadata not available — capability-layer overlay disabled.")
        return
    st.markdown("**Capability-layer scheme** — each indicator is tagged by which layer it targets.")
    cols = st.columns(len(metadata["layers"]))
    for col, (layer_id, info) in zip(cols, metadata["layers"].items()):
        with col:
            st.markdown(_layer_chip(layer_id, metadata), unsafe_allow_html=True)
            st.caption(info.get("description", ""))


def render_evidence_card(indicator_code: str, metadata: dict) -> None:
    """Render a per-indicator evidence card with source, layer, mechanism, decision history.

    This is the audit unit-of-account for the artefact: every indicator carries one card.
    """
    indicators = metadata.get("indicators", {})
    info = indicators.get(indicator_code)
    if not info:
        st.info(f"No evidence card available for `{indicator_code}` (not in metadata).")
        return
    name = info.get("name", indicator_code)
    domain = info.get("domain", "—")
    layer_id = info.get("layer", "")
    cv = info.get("construct_validity", "—")
    cv_color = {"ok": "#26A69A", "weak": "#FFB300", "fail": "#EF5350"}.get(cv, "#9E9E9E")

    st.markdown(f"### `{indicator_code}` — {name}")
    st.markdown(
        f"**Domain:** `{domain}` &nbsp; "
        f"**Layer:** {_layer_chip(layer_id, metadata)} &nbsp; "
        f"**Construct validity:** "
        f"<span style='background-color:{cv_color};color:white;padding:2px 8px;border-radius:10px;'>{cv}</span>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"**Source:** {info.get('source_agency', '—')}  "
        f"([source]({info.get('source_url', '#')}))  "
        f"License: `{info.get('license', '—')}`"
    )
    st.markdown(f"**Mechanism.** {info.get('mechanism', '—')}")
    if info.get("replacement"):
        st.markdown(f"**Replacement plan.** {info['replacement']}")


def render_dag(dag_text: str) -> None:
    """Render the identification DAG. Streamlit will display the mermaid block as a code
    fence; environments with mermaid-rendering plug-ins will render it visually."""
    st.markdown(dag_text)
    st.caption(
        "Reading the DAG: solid arrows are paths the estimator closes; dotted arrows are paths "
        "it does not. β is the estimand. See `academic/04_identification_strategy_revised.md` §1–§3."
    )


def render_specification_array(coef_df: pd.DataFrame, fit_df: pd.DataFrame) -> None:
    """Render a forest plot of β across all available specifications, ordered by spec_kind.

    Precursor to the full Simonsohn et al. (2020) specification curve, which is preregistered
    for the next release once the 15-spec battery is implemented (see academic/05_robustness_protocol.md).
    """
    if coef_df.empty:
        st.info("Specification array not available (no causal coefficients).")
        return
    sub = coef_df.copy()
    sub = sub[sub["predictor"].astype(str).str.contains("DCLO_score", na=False)]
    if sub.empty:
        st.info("No DCLO_score-on-outcome rows in coefficient file.")
        return
    sub = sub.sort_values(["spec_kind", "spec_id"])
    sub["label"] = sub["spec_kind"].astype(str) + " | " + sub["spec_id"].astype(str)
    se = pd.to_numeric(sub["std_error"], errors="coerce").fillna(0)
    fig = px.scatter(
        sub,
        x="coef",
        y="label",
        color="spec_kind",
        error_x=1.96 * se,
        title="Specification array (β on lagged DCLO across specs; 95 % CI)",
        labels={"coef": "β (lagged DCLO → outcome)", "label": "Specification"},
    )
    fig.add_vline(x=0.0, line_dash="dash")
    fig.update_layout(height=420)
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Reviewer note: the pooled spec recovers β ≈ 1.0 with R² = 0.91. The within-FE specs collapse to "
        "β ≈ 0.62 with substantially lower R². The gap is the structural-overlap signature. The full "
        "specification curve (S1–S15 from `academic/05_robustness_protocol.md` §1) is preregistered "
        "for the next release."
    )


def render_disagreement_panel(country_df: pd.DataFrame, external_df: pd.DataFrame, score_col: str) -> None:
    """Render the disagreement-with-standard-family panel.

    The headline scholarly claim is that DCLO disagrees with ITU IDI / EGDI / GTMI / NRI in
    informative ways. This view is the empirical instantiation of that claim.
    """
    if country_df.empty:
        st.info("Country data unavailable — disagreement panel disabled.")
        return
    if external_df.empty:
        st.info("External index ranks not bundled. See `data/external/standard_family_ranks_2023.csv`.")
        return

    latest_year = int(pd.to_numeric(country_df["year"], errors="coerce").max())
    target_year = min(latest_year, 2023)
    snap = country_df[country_df["year"] == target_year].copy()
    snap = snap.dropna(subset=[score_col, "economy"])
    if snap.empty:
        st.info(f"No DCLO rows for year {target_year}.")
        return
    snap["dclo_rank"] = snap[score_col].rank(ascending=False, method="min")
    use = snap[["economy", score_col, "dclo_rank"]].merge(external_df, on="economy", how="inner")
    if use.empty:
        st.info("No country-name overlap between DCLO panel and external-index file.")
        return

    st.markdown(
        f"DCLO ranks for **{target_year}** are merged with published ranks from "
        "ITU IDI 2023, UN EGDI 2022, World Bank GTMI 2022, and Portulans NRI 2023 "
        "(`data/external/standard_family_ranks_2023.csv`)."
    )

    rank_cols = [c for c in ["itu_idi_rank_2023", "egdi_rank_2022", "gtmi_rank_2022", "nri_rank_2023"] if c in use.columns]
    if rank_cols:
        rho_rows = []
        for col in rank_cols:
            sub = use[["dclo_rank", col]].apply(pd.to_numeric, errors="coerce").dropna()
            if len(sub) >= 5:
                rho = sub.corr(method="spearman").iloc[0, 1]
                rho_rows.append({"index": col, "spearman_rho_vs_dclo": round(float(rho), 3), "n": len(sub)})
        if rho_rows:
            st.subheader("Spearman ρ between DCLO rank and standard-family ranks")
            rho_df = pd.DataFrame(rho_rows)
            st.dataframe(rho_df, use_container_width=True)
            st.caption(
                "A capability-grounded index that registers what the standard family does not "
                "should produce attenuated ρ values. ρ near 1 would mean DCLO is a relabel of the "
                "existing family; ρ near 0 or negative would mean DCLO surfaces something new."
            )

    st.subheader("Top disagreements (where DCLO and the standard family diverge most)")
    if "itu_idi_rank_2023" in use.columns:
        diff = use.dropna(subset=["dclo_rank", "itu_idi_rank_2023"]).copy()
        diff["dclo_minus_itu"] = diff["dclo_rank"] - diff["itu_idi_rank_2023"]
        diff = diff.reindex(diff["dclo_minus_itu"].abs().sort_values(ascending=False).index).head(15)
        st.markdown("Largest |DCLO rank − ITU IDI rank| differences (n = 15):")
        st.dataframe(
            diff[["economy", "dclo_rank", "itu_idi_rank_2023", "dclo_minus_itu"]].rename(
                columns={"dclo_rank": "DCLO rank", "itu_idi_rank_2023": "ITU IDI rank", "dclo_minus_itu": "Δ rank"}
            ),
            use_container_width=True,
        )
        st.caption(
            "Positive Δ means a country is ranked **lower** by DCLO than by ITU IDI — i.e., the capability "
            "frame penalises the country relative to a resources-and-readiness frame. Negative Δ means the "
            "reverse. Interpret with construct-validity caveats: until the indicator-replacement build, "
            "DCLO is itself contaminated."
        )


def render_methods_tab(theory_text: str, identification_text: str, robustness_text: str) -> None:
    """Render Methods as a first-class tab containing capability framework, identification
    strategy, and robustness protocol, inline. Reviewer must not have to leave the artefact.
    """
    st.subheader("1. Theoretical framework — capability approach + critical data studies")
    with st.expander("Read in full", expanded=False):
        st.markdown(theory_text)
    st.subheader("2. Revised identification strategy")
    with st.expander("Read in full", expanded=False):
        st.markdown(identification_text)
    st.subheader("3. Robustness protocol (specifications, placebos, stability tests)")
    with st.expander("Read in full", expanded=False):
        st.markdown(robustness_text)


def render_reflexivity_tab(country_df: pd.DataFrame, ethics_text: str) -> None:
    """Render the inclusion / exclusion / reflexivity panel.

    A capability-grounded artefact must be explicit about who is rendered legible and who is
    not. This tab makes that visible.
    """
    if not country_df.empty and "economy" in country_df.columns:
        latest_year = int(pd.to_numeric(country_df["year"], errors="coerce").max())
        snap = country_df[country_df["year"] == latest_year]
        included = sorted(snap["economy"].dropna().astype(str).unique().tolist())
        st.subheader(f"Included economies ({len(included)} in {latest_year})")
        st.write(", ".join(included) if included else "None.")
    st.subheader("Notable exclusions and why")
    st.markdown(
        "- **Most low-income African economies** — sparse coverage in WB/ITU/Findex panels.\n"
        "- **Several South Asian peers** (Bhutan, Nepal, Maldives, Sri Lanka, Afghanistan) — partial coverage.\n"
        "- **Small island developing states** — typically below the panel-coverage gate.\n"
        "- **Hong Kong SAR, Macau SAR, Taiwan** — source-data conventions.\n"
        "- **Russia and Belarus from 2022** — data-source disruptions; flagged in robustness.\n\n"
        "These exclusions are not 'data limitations' to be apologised for; they are the political "
        "geography of the standard indicator family. A capability-grounded reading must surface them."
    )
    st.divider()
    st.subheader("Standpoint and reflexivity")
    st.markdown(ethics_text)


def render_releases_tab(changelog_text: str) -> None:
    """Render the version history. Acts as the artefact's version-of-record."""
    st.markdown(changelog_text)
    st.caption(
        "Version 0.4.0 (2026-04-29) is the critical-instrument release. Each release is "
        "audit-logged via the per-pipeline manifests under `data/gold/*_audit_manifest.json`."
    )


def render_citation_block(citation_text: str) -> None:
    """Render the citation apparatus (BibTeX, APA, build hash) — sidebar block."""
    st.markdown("**Cite this dashboard.**")
    bibtex = (
        "@misc{Jha2026DCLODashboard,\n"
        "  author       = {Jha, Rahul},\n"
        "  title        = {{DCLO} -- Digital Capability for Life Outcomes:\n"
        "                  Dashboard and Critical Instrument},\n"
        f"  year         = {{2026}},\n"
        f"  version      = {{{DASHBOARD_VERSION}}},\n"
        "  institution  = {JSGP, O.\\ P.\\ Jindal Global University},\n"
        "  url          = {https://github.com/techpolicycomms/dclo-dual-track-dashboard},\n"
        "  note         = {Companion paper: academic/09\\_paper\\_draft.md}\n"
        "}\n"
    )
    st.code(bibtex, language="bibtex")
    apa = (
        "Jha, R. (2026). DCLO — Digital Capability for Life Outcomes: Dashboard and Critical "
        f"Instrument (Version {DASHBOARD_VERSION}) [Software]. Jindal School of Government and "
        "Public Policy, O. P. Jindal Global University. https://github.com/techpolicycomms/"
        "dclo-dual-track-dashboard"
    )
    st.markdown("APA: " + apa)
    st.caption(
        f"Version `{DASHBOARD_VERSION}` · built `{DASHBOARD_BUILD_DATE}` · license `{DASHBOARD_LICENSE}`. "
        "See `CITATION.cff` for the canonical citation file and `CHANGELOG.md` for version history."
    )


def apply_custom_domain_weights(
    df: pd.DataFrame, weights: dict, domain_score_columns: list[str]
) -> pd.DataFrame:
    """Re-compute the overall DCLO_score under user-supplied domain weights.

    `weights` maps domain code (e.g., "ACC") to a non-negative weight. Missing domains
    in a row are skipped (the weighted mean uses only available domains). Weights sum
    is normalised internally; a zero-sum weight vector falls back to equal weighting.

    The diagnostic value: a capability index that collapses all six domains into a single
    number imposes a trade-off the capability approach itself rejects (Sen 1999;
    Robeyns 2017, Ch. 2). Letting the reader interrogate the trade-off is the artefact's
    central interactive affordance.
    """
    if df.empty:
        return df
    out = df.copy()
    domain_codes = [col.replace("_score", "") for col in domain_score_columns]
    weight_vector = pd.Series(
        {code: max(0.0, float(weights.get(code, 1.0))) for code in domain_codes}
    )
    if weight_vector.sum() <= 0:
        weight_vector = pd.Series({code: 1.0 for code in domain_codes})

    available = [col for col in domain_score_columns if col in out.columns]
    if not available:
        return out

    domain_block = out[available].apply(pd.to_numeric, errors="coerce")
    weights_per_col = pd.Series(
        {col: weight_vector[col.replace("_score", "")] for col in available}
    )

    masked_weights = domain_block.notna().mul(weights_per_col, axis=1)
    weighted_sum = domain_block.fillna(0).mul(weights_per_col, axis=1).sum(axis=1)
    weight_total = masked_weights.sum(axis=1)
    out["DCLO_score"] = weighted_sum / weight_total.replace(0, pd.NA)
    if "DCLO_score_confidence_weighted" in out.columns:
        out["DCLO_score_confidence_weighted"] = out["DCLO_score"]
    return out


def apply_construct_validity_stress_test(
    df: pd.DataFrame, metadata: dict, domain_score_columns: list[str]
) -> pd.DataFrame:
    """Re-compute domain and overall DCLO scores after dropping indicators flagged as failing
    construct validity in `academic/indicator_metadata.json`.

    The diagnostic value: if dropping construct-failing indicators substantially changes the
    rankings, the published scores are not capability-grounded. If they do not change, the
    failing indicators were redundant.
    """
    if df.empty:
        return df
    indicators = metadata.get("indicators", {})
    failing = {
        code for code, info in indicators.items()
        if info.get("construct_validity") == "fail"
    }
    if not failing:
        return df

    out = df.copy()
    domain_to_indicators: dict[str, list[str]] = {}
    for code, info in indicators.items():
        if info.get("construct_validity") == "fail":
            continue
        domain = info.get("domain")
        if not domain:
            continue
        z_col = f"Z_{code}"
        if z_col in out.columns:
            domain_to_indicators.setdefault(domain, []).append(z_col)

    for domain_score_col in domain_score_columns:
        domain = domain_score_col.replace("_score", "")
        valid_z = domain_to_indicators.get(domain, [])
        if not valid_z:
            out[domain_score_col] = pd.NA
            continue
        out[domain_score_col] = out[valid_z].mean(axis=1, skipna=True)

    score_cols_present = [c for c in domain_score_columns if c in out.columns]
    if score_cols_present:
        out["DCLO_score"] = out[score_cols_present].mean(axis=1, skipna=True)
        if "DCLO_score_confidence_weighted" in out.columns:
            out["DCLO_score_confidence_weighted"] = out["DCLO_score"]
    return out


def main() -> None:
    st.title("DCLO Dashboard")
    st.caption("Digital Capability for Life Outcomes (Dual-Track: India State-Year + Country-Year)")
    st.warning(CONSTRUCT_VALIDITY_BANNER, icon="⚠️")

    try:
        state_df = load_data(STATE_DATA_PATH)
    except Exception as exc:
        st.error(f"Failed to load state dataset: {exc}")
        st.stop()
    try:
        country_df = load_data(COUNTRY_DATA_PATH)
    except Exception:
        country_df = pd.DataFrame()

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

    indicator_metadata = load_indicator_metadata(INDICATOR_METADATA_PATH)
    dag_text = load_text_file(DAG_PATH)
    theory_text = load_text_file(THEORY_PATH)
    identification_text = load_text_file(IDENTIFICATION_PATH)
    robustness_text = load_text_file(ROBUSTNESS_PATH)
    ethics_text = load_text_file(ETHICS_PATH)
    changelog_text = load_text_file(CHANGELOG_PATH)
    citation_text = load_text_file(CITATION_PATH)
    external_ranks_df = load_external_ranks(EXTERNAL_RANKS_PATH)

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
                help=(
                    "Known limitation: in the current build, CTX_dpi_composite_v2 is a constant "
                    "across states within each year, so this toggle adds a uniform level shift "
                    "rather than relative context — see academic/01_gap_analysis.md §C3."
                ),
            )
            score_col = "DCLO_score_context_adjusted" if use_context_adjusted and "DCLO_score_context_adjusted" in state_df.columns else "DCLO_score"
            working_df = state_df.copy()
            entity_col = "state_name"
            domain_cols = STATE_DOMAIN_SCORE_COLUMNS
        else:
            if country_df.empty or not {"economy", "year", "DCLO_score"}.issubset(country_df.columns):
                st.error("Country-year dataset not available or missing required columns.")
                st.stop()
            years = sorted([int(year) for year in country_df["year"].dropna().unique().tolist()])
            economies = sorted(country_df["economy"].dropna().astype(str).unique().tolist())
            selected_year = st.selectbox("Year", years, index=len(years) - 1)
            selected_entities = st.multiselect("Countries for trend chart", economies, default=["India"] if "India" in economies else economies[:5])
            selected_profile_entity = st.selectbox("Country for domain profile", economies, index=economies.index("India") if "India" in economies else 0)
            score_mode = st.selectbox("Score mode", ["Baseline", "Confidence-weighted"])
            score_col = (
                "DCLO_score_confidence_weighted"
                if score_mode == "Confidence-weighted" and "DCLO_score_confidence_weighted" in country_df.columns
                else "DCLO_score"
            )
            trust_filter = st.multiselect(
                "Trust tier filter",
                options=["High", "Medium", "Low"],
                default=["High", "Medium", "Low"],
            )
            working_df = country_df.copy()
            if "model_trust_tier" in working_df.columns:
                working_df = working_df[working_df["model_trust_tier"].isin(trust_filter)].copy()
            entity_col = "economy"
            domain_cols = COUNTRY_DOMAIN_SCORE_COLUMNS

        st.divider()
        st.header("Model QA")
        overall_passed = checks.get("overall_passed")
        if overall_passed is True:
            st.success("Data-integrity checks: PASS")
        elif overall_passed is False:
            st.error("Data-integrity checks: FAIL")
        else:
            st.info("Data-integrity checks: not available")
        st.error(
            "Construct-validity checks: FAIL — see academic/03_indicator_validity_audit.md. "
            "Data-integrity passes do not validate the capability claim."
        )

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
        unit_text = "state-year (India track)" if track == "India State-Year" else "country-year (DPI comparative track)"
        st.markdown(
            "\n".join(
                [
                    "- Construct: **DCLO (Digital Capability for Life Outcomes)**",
                    f"- Unit: **{unit_text}**",
                    "- Measurement: **formative multidimensional composite**",
                    "- Causal layer (country track): **panel model with lagged predictors**",
                    "- Interpretation: prioritize **effect sizes and uncertainty** over raw ranks",
                    "- **Critical reading**: see `academic/` for capability-approach grounding,",
                    "  indicator-validity audit, and pre-registered next-release plan.",
                ]
            )
        )
        with st.expander("Dashboard explainer", expanded=False):
            st.markdown(explainer_text)
        with st.expander("Academic track / capability-approach critique", expanded=False):
            st.markdown(
                "The dashboard is a working instrument. The capability-grounded paper that uses it "
                "as a critical instrument lives in `academic/`. Start with "
                "`academic/00_response_to_sarkar.md` and `academic/01_gap_analysis.md`."
            )

        st.divider()
        st.header("Construct-validity stress test")
        stress_test = st.checkbox(
            "Drop construct-failing indicators and re-rank",
            value=False,
            help=(
                "Re-computes DCLO using only indicators that pass the construct-validity audit "
                "in academic/03_indicator_validity_audit.md. The diagnostic value is in the "
                "rank shift: if scores change substantially, the published rankings are not "
                "capability-grounded; if not, the failing indicators were redundant."
            ),
        )
        st.caption(
            "When engaged, all rankings, KPIs, maps, and trends in the Measurement tab "
            "re-render under the construct-validated subset."
        )

        st.divider()
        st.header("Counterfactual domain weights")
        custom_weights_active = st.checkbox(
            "Use custom domain weights",
            value=False,
            help=(
                "A single composite forces a trade-off across plural and arguably "
                "incommensurable domains (Sen 1999; Robeyns 2017, Ch. 2). This panel "
                "lets the reader interrogate the trade-off: shift weight to the domain "
                "they think most central to digital capability and watch the ranking respond."
            ),
        )
        domain_codes_for_weights = [c.replace("_score", "") for c in domain_cols]
        custom_weights = {}
        if custom_weights_active:
            for code in domain_codes_for_weights:
                custom_weights[code] = st.slider(
                    f"{code} weight",
                    min_value=0.0,
                    max_value=2.0,
                    value=1.0,
                    step=0.05,
                    key=f"weight_{code}_{track}",
                )
            st.caption(
                "Equal weight is 1.0 for every domain. Set a domain to 0.0 to drop it; "
                "set above 1.0 to amplify it. Weights are normalised internally; missing "
                "domains in a row are skipped."
            )

        st.divider()
        with st.expander("Cite this dashboard", expanded=False):
            render_citation_block(citation_text)

    if stress_test:
        working_df = apply_construct_validity_stress_test(working_df, indicator_metadata, domain_cols)
    if custom_weights_active and custom_weights:
        working_df = apply_custom_domain_weights(working_df, custom_weights, domain_cols)

    df_year = working_df[working_df["year"] == selected_year].copy().dropna(subset=[score_col])
    if df_year.empty:
        st.warning(f"No DCLO records for {selected_year}.")
        st.stop()

    if checks.get("overall_passed") is False:
        st.warning(
            "Model QA checks are currently failing for at least one track. "
            "Use rankings with caution and review `data/gold/dclo_standard_checks_summary.md`."
        )

    (
        tab_measure,
        tab_causal,
        tab_robust,
        tab_compare,
        tab_methods,
        tab_reflex,
        tab_provenance,
        tab_releases,
    ) = st.tabs(
        [
            "Measurement",
            "Causal Evidence",
            "Robustness",
            "Standard-Family Comparison",
            "Methods",
            "Inclusion & Reflexivity",
            "Data Provenance & Audit",
            "Releases",
        ]
    )

    with tab_measure:
        if stress_test:
            st.success(
                "**Stress-test mode engaged.** Construct-failing indicators dropped per "
                "`academic/03_indicator_validity_audit.md`. Compare the rankings here with "
                "the un-stressed view by toggling the sidebar checkbox."
            )
        if custom_weights_active and custom_weights:
            weights_str = ", ".join(f"{k}={v:.2f}" for k, v in custom_weights.items())
            st.info(
                f"**Counterfactual weights active**: {weights_str}. "
                "DCLO scores below are re-computed under your chosen weights — Sen's plural-"
                "incommensurability point made operational."
            )
        if indicator_metadata.get("layers"):
            with st.expander("Capability-layer scheme", expanded=False):
                render_capability_layer_overlay(indicator_metadata)
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
            render_trends(
                working_df,
                selected_entities,
                entity_col=entity_col,
                score_col=score_col,
                title="DCLO Trend Over Time (Selected Countries)",
            )
            render_domain_heatmap(df_year, entity_col=entity_col, domain_cols=domain_cols, title="Domain Score Heatmap by Country")
            render_downloads(df_year, score_col=score_col, entity_col=entity_col, suffix="country_year")

        if indicator_metadata.get("indicators"):
            st.divider()
            st.subheader("Indicator evidence cards")
            st.caption(
                "Select an indicator to view its capability-layer assignment, source agency, "
                "license, mechanism paragraph, and replacement plan. This is the audit unit "
                "of account for the artefact."
            )
            available_indicators = sorted(
                code for code in indicator_metadata.get("indicators", {})
                if any(code in c for c in working_df.columns) or f"Z_{code}" in working_df.columns
                or code in working_df.columns
            )
            if not available_indicators:
                available_indicators = sorted(indicator_metadata.get("indicators", {}).keys())
            chosen_indicator = st.selectbox(
                "Indicator",
                available_indicators,
                key=f"evidence_card_select_{track}",
            )
            if chosen_indicator:
                render_evidence_card(chosen_indicator, indicator_metadata)

    with tab_causal:
        if track == "India State-Year":
            st.info("Causal evidence tab is currently available for the country-year comparative track.")
        else:
            st.error(
                "**Read before interpreting.** The headline coefficient (DCLO_t-1 → SRV_t, β≈0.62) "
                "is contaminated by structural overlap between predictor and outcome: SRV is built from "
                "WTO services-trade exports, which share macroeconomic content with the predictor "
                "composite. The pooled specification recovers β≈1.00 with R²=0.91 — a giveaway. "
                "Treat these coefficients as conditional associations under disclosed misspecification, "
                "not as a capability-policy effect. See academic/04_identification_strategy_revised.md §5."
            )
            st.caption(
                "Interpret coefficients as conditional associations from the configured panel model. "
                "They are not guaranteed causal unless identification assumptions hold."
            )
            with st.expander("Identification DAG and estimand", expanded=True):
                render_dag(dag_text)
            render_causal_forest(causal_coef_df)
            st.subheader("Specification Fit")
            render_model_fit_table(causal_fit_df)
            if not causal_coef_df.empty:
                pvals = pd.to_numeric(causal_coef_df["p_value_norm_approx"], errors="coerce")
                sig = causal_coef_df[pvals < 0.05].copy()
                st.subheader("Statistically Significant Terms (p < 0.05)")
                st.dataframe(sig.sort_values(["spec_kind", "spec_id", "p_value_norm_approx"]), use_container_width=True)

    with tab_robust:
        if track == "India State-Year":
            st.info("Robustness diagnostics are currently generated for the country-year comparative track.")
        else:
            st.subheader("Specification array")
            render_specification_array(causal_coef_df, causal_fit_df)
            st.divider()
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
                        "- The full 15-spec robustness battery and 6-placebo set are preregistered in `academic/05_robustness_protocol.md` and `academic/11_preregistration.md`.",
                    ]
                )
            )

    with tab_compare:
        st.subheader("Standard-family comparison")
        st.markdown(
            "DCLO is positioned as a **critical instrument**, not a competitor ranking. "
            "The view below compares DCLO ranks against the leading digital-development indices "
            "(ITU IDI 2023; UN EGDI 2022; World Bank GTMI 2022; Portulans NRI 2023). "
            "**Where DCLO disagrees with the family, the disagreement is the diagnostic value.**"
        )
        if track == "India State-Year":
            st.info(
                "Standard-family comparison is currently available for the country-year track only. "
                "An India sub-national equivalent (NeGD State eReadiness; MeitY DPI dashboards) "
                "is on the roadmap."
            )
        else:
            render_disagreement_panel(country_df, external_ranks_df, score_col)

    with tab_methods:
        st.caption(
            "Methods are surfaced inline so a reviewer can evaluate the artefact without "
            "leaving it. The expanders below contain the full text of `academic/02_theoretical_framework.md`, "
            "`academic/04_identification_strategy_revised.md`, and `academic/05_robustness_protocol.md`."
        )
        render_methods_tab(theory_text, identification_text, robustness_text)

    with tab_reflex:
        st.caption(
            "A capability-grounded artefact must be explicit about who it makes legible and who it does not. "
            "Inclusion / exclusion lists, the standpoint statement, and the ethics commitments live here."
        )
        render_reflexivity_tab(country_df, ethics_text)

    with tab_provenance:
        st.caption(
            "Full audit trail for data verification and reproducibility. "
            "Every pipeline run records input checksums (SHA-256), row-level accounting, "
            "analytical parameters, random seeds, and output checksums. "
            "This follows Christensen & Miguel (2018) transparency principles."
        )
        if track == "India State-Year":
            render_provenance(state_manifest, state_verification, "India State-Year")
        else:
            render_provenance(country_manifest, country_verification, "Country-Year")
            if causal_manifest:
                st.divider()
                st.header("Causal Panel Audit Trail")
                render_provenance(causal_manifest, {}, "Causal Panel")

    with tab_releases:
        st.caption(
            "Versioned release history for the dashboard. Each entry is the equivalent of a "
            "journal version-of-record + corrigendum for a paper. Audit manifests for each "
            "release are stored under `data/gold/*_audit_manifest.json`."
        )
        render_releases_tab(changelog_text)

    with st.expander("About this dashboard", expanded=False):
        st.markdown(explainer_text)

    st.subheader("Filtered Data Preview")
    st.dataframe(df_year.sort_values(score_col, ascending=False), use_container_width=True)


if __name__ == "__main__":
    main()
