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


def main() -> None:
    st.title("DCLO Dashboard")
    st.caption("Digital Capability for Life Outcomes (Dual-Track: India State-Year + Country-Year)")

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
        unit_text = "state-year (India track)" if track == "India State-Year" else "country-year (DPI comparative track)"
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

    tab_measure, tab_causal, tab_robust = st.tabs(["Measurement", "Causal Evidence", "Robustness"])

    with tab_measure:
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

    with tab_causal:
        if track == "India State-Year":
            st.info("Causal evidence tab is currently available for the country-year comparative track.")
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
                st.dataframe(sig.sort_values(["spec_kind", "spec_id", "p_value_norm_approx"]), use_container_width=True)

    with tab_robust:
        if track == "India State-Year":
            st.info("Robustness diagnostics are currently generated for the country-year comparative track.")
        else:
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

    with st.expander("About this dashboard", expanded=False):
        st.markdown(explainer_text)

    st.subheader("Filtered Data Preview")
    st.dataframe(df_year.sort_values(score_col, ascending=False), use_container_width=True)


if __name__ == "__main__":
    main()
