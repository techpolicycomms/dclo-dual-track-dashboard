import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml


def read_config(config_path: str) -> Dict[str, object]:
    with open(config_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def zscore(series: pd.Series) -> pd.Series:
    std = series.std(ddof=0)
    if std == 0 or pd.isna(std):
        return pd.Series([0.0] * len(series), index=series.index)
    return (series - series.mean()) / std


def compute_vif(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    rows = []
    cols = [col for col in columns if col in df.columns]
    if len(cols) < 2:
        return pd.DataFrame(columns=["indicator_code", "vif"])
    xdf = df[cols].apply(pd.to_numeric, errors="coerce").dropna()
    # Standardize to avoid numeric overflow in least-squares on mixed-scale indicators.
    xdf = (xdf - xdf.mean()) / xdf.std(ddof=0).replace(0, np.nan)
    xdf = xdf.replace([np.inf, -np.inf], np.nan).dropna()
    if len(xdf) < 10:
        return pd.DataFrame(columns=["indicator_code", "vif"])
    for target in cols:
        y = xdf[target].to_numpy()
        x_cols = [col for col in cols if col != target]
        X = xdf[x_cols].to_numpy()
        X = np.column_stack([np.ones(len(X)), X])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            y_hat = X @ beta
        if not np.isfinite(y_hat).all():
            rows.append({"indicator_code": target, "vif": np.inf})
            continue
        ss_res = float(np.sum((y - y_hat) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        if ss_tot == 0:
            vif = np.inf
        else:
            r2 = 1.0 - (ss_res / ss_tot)
            vif = np.inf if r2 >= 0.999999 else 1.0 / max(1e-8, (1.0 - r2))
        rows.append({"indicator_code": target, "vif": vif})
    return pd.DataFrame(rows)


def infer_dclo_domain(row: pd.Series) -> str:
    pillar = str(row.get("pillar", "")).strip().lower()
    name = str(row.get("indicator_name", "")).strip().lower()
    code = str(row.get("indicator_code", "")).strip().lower()
    text = f"{name} {code}"

    if (
        "skill" in text
        or "literacy" in text
        or "education" in text
        or "school" in text
        or "enroll" in text
        or "enrollment" in text
        or "training" in text
        or "learning" in text
        or code.startswith("wb_se.")
    ):
        return "SKL"
    if "internet" in text or "mobile" in text or "broadband" in text or "connect" in text:
        return "ACC"
    if "service" in text or "government systems" in text or "govtech" in text:
        return "SRV"
    if "trust" in text or "governance" in text or "inclusion" in text or "equity" in text:
        return "AGR"
    if "finance" in text or "payment" in text or "trade" in text or "economic" in text:
        return "ECO"
    if "co2" in text or "emission" in text or "resilience" in text or "sustainability" in text:
        return "OUT"

    if pillar == "access_usage":
        return "ACC"
    if pillar == "trust_governance":
        return "AGR"
    if pillar == "affordability_inclusion":
        return "ECO"
    if pillar == "sustainability_resilience":
        return "OUT"
    return "OUT"


def build_indicator_stats(long_df: pd.DataFrame, coverage_df: pd.DataFrame, panel_coverage: pd.Series) -> pd.DataFrame:
    grp = long_df.groupby("indicator_code", as_index=False).agg(
        indicator_name=("indicator_name", "first"),
        pillar=("pillar", "first"),
        years_covered=("year", "nunique"),
        economies_covered=("economy", "nunique"),
        imputation_share=("was_imputed", lambda s: float(pd.to_numeric(s, errors="coerce").fillna(0).mean())),
    )
    grp["dclo_domain"] = grp.apply(infer_dclo_domain, axis=1)

    coverage_use = coverage_df.rename(columns={"pct_observed": "pct_observed_global"})[
        ["indicator_code", "pct_observed_global"]
    ].copy()
    panel_cov_df = panel_coverage.rename("panel_coverage").reset_index().rename(columns={"index": "indicator_code"})
    stats = grp.merge(coverage_use, on="indicator_code", how="left").merge(panel_cov_df, on="indicator_code", how="left")
    stats["pct_observed_global"] = pd.to_numeric(stats["pct_observed_global"], errors="coerce")
    stats["pct_observed_global"] = stats["pct_observed_global"].fillna(0.0)
    stats["panel_coverage"] = pd.to_numeric(stats["panel_coverage"], errors="coerce").fillna(0.0)

    # Simple balanced ranking: favor observed coverage and penalize imputation.
    stats["ranking_score"] = (
        stats["pct_observed_global"]
        + stats["panel_coverage"] * 100.0
        + stats["years_covered"] * 0.25
        + stats["economies_covered"] * 0.10
        - stats["imputation_share"] * 50.0
    )
    return stats


def prune_by_correlation_and_vif(
    wide_df: pd.DataFrame,
    candidates: List[str],
    ranking_score_map: Dict[str, float],
    correlation_threshold: float,
    max_vif: float,
    min_keep: int,
) -> List[str]:
    selected = [c for c in candidates if c in wide_df.columns]
    if len(selected) <= 1:
        return selected

    # Correlation pruning: keep higher-ranked indicator from highly correlated pairs.
    corr = wide_df[selected].apply(pd.to_numeric, errors="coerce").corr().abs()
    to_drop = set()
    for i, col_a in enumerate(selected):
        for col_b in selected[i + 1 :]:
            if col_a in to_drop or col_b in to_drop:
                continue
            val = corr.loc[col_a, col_b]
            if pd.notna(val) and val >= correlation_threshold:
                score_a = ranking_score_map.get(col_a, 0.0)
                score_b = ranking_score_map.get(col_b, 0.0)
                to_drop.add(col_b if score_a >= score_b else col_a)
    selected = [c for c in selected if c not in to_drop]

    # VIF pruning with floor on min_keep.
    while len(selected) > max(min_keep, 2):
        vif_df = compute_vif(wide_df, selected)
        if vif_df.empty:
            break
        worst = vif_df.sort_values("vif", ascending=False).iloc[0]
        if pd.isna(worst["vif"]) or worst["vif"] <= max_vif:
            break
        selected.remove(str(worst["indicator_code"]))
    return selected


def select_indicators_by_domain(
    stats: pd.DataFrame, gating: Dict[str, object], wide_df: pd.DataFrame
) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    min_pct_observed = float(gating.get("min_pct_observed", 70.0))
    max_imputation_share = float(gating.get("max_imputation_share", 0.35))
    min_overlap_years = int(gating.get("min_overlap_years", 5))
    min_panel_coverage = float(gating.get("min_panel_coverage", 0.40))
    min_per_domain = int(gating.get("min_per_domain", 4))
    max_per_domain = int(gating.get("max_per_domain", 8))
    correlation_threshold = float(gating.get("correlation_prune_threshold", 0.90))
    max_vif = float(gating.get("max_vif", 10.0))

    stats = stats.copy()
    stats["role"] = "exclude"
    stats["gate_pass"] = (
        (stats["pct_observed_global"] >= min_pct_observed)
        & (stats["imputation_share"] <= max_imputation_share)
        & (stats["years_covered"] >= min_overlap_years)
        & (stats["panel_coverage"] >= min_panel_coverage)
    )

    selected: Dict[str, List[str]] = {}
    for domain in sorted(stats["dclo_domain"].dropna().unique().tolist()):
        sub = stats[stats["dclo_domain"] == domain].copy()
        sub_pass = sub[sub["gate_pass"]].sort_values("ranking_score", ascending=False)
        if sub.empty:
            selected[domain] = []
            continue

        n_pick = min(max_per_domain, max(min_per_domain, len(sub_pass)))
        picks = sub_pass.head(n_pick)["indicator_code"].tolist()
        if len(picks) < min_per_domain:
            fallback_pool = sub[~sub["indicator_code"].isin(picks)].sort_values("ranking_score", ascending=False)
            needed = min_per_domain - len(picks)
            picks.extend(fallback_pool.head(needed)["indicator_code"].tolist())

        ranking_map = {row["indicator_code"]: float(row["ranking_score"]) for _, row in sub.iterrows()}
        picks = prune_by_correlation_and_vif(
            wide_df=wide_df,
            candidates=picks,
            ranking_score_map=ranking_map,
            correlation_threshold=correlation_threshold,
            max_vif=max_vif,
            min_keep=1,
        )
        selected[domain] = picks
        stats.loc[stats["indicator_code"].isin(picks), "role"] = "core_formative"
        context_ids = sub_pass[~sub_pass["indicator_code"].isin(picks)]["indicator_code"].tolist()
        if context_ids:
            stats.loc[stats["indicator_code"].isin(context_ids), "role"] = "context_only"

    return stats, selected


def build_country_scores(
    long_df: pd.DataFrame, selected: Dict[str, List[str]], stats_df: pd.DataFrame, scoring_cfg: Dict[str, object]
) -> pd.DataFrame:
    selected_ids = sorted({indicator for indicators in selected.values() for indicator in indicators})
    if not selected_ids:
        raise ValueError("No indicators selected after gating; relax gating thresholds.")

    use = long_df[long_df["indicator_code"].isin(selected_ids)].copy()
    use["value"] = pd.to_numeric(use["norm_score"], errors="coerce")
    use = use.dropna(subset=["economy", "year", "indicator_code", "value"])

    pivot = use.pivot_table(index=["economy", "year"], columns="indicator_code", values="value", aggfunc="mean").reset_index()
    indicator_cols = [col for col in pivot.columns if col not in {"economy", "year"}]
    for col in indicator_cols:
        pivot[f"Z_{col}"] = zscore(pd.to_numeric(pivot[col], errors="coerce"))

    domain_to_score_col = {
        "ACC": "ACC_score",
        "SKL": "SKL_score",
        "SRV": "SRV_score",
        "AGR": "AGR_score",
        "ECO": "ECO_score",
        "OUT": "OUT_score",
    }
    domain_weights: Dict[str, float] = {}
    for domain, indicators in selected.items():
        sub = stats_df[stats_df["indicator_code"].isin(indicators)]
        if sub.empty:
            domain_weights[domain] = 1.0
            continue
        # Weight by coverage quality (0-1 scale), clipped away from zero.
        w = float(sub["panel_coverage"].mean())
        domain_weights[domain] = max(0.05, w)

    for domain, indicators in selected.items():
        z_cols = [f"Z_{code}" for code in indicators if f"Z_{code}" in pivot.columns]
        if not z_cols:
            continue
        score_col = domain_to_score_col.get(domain, f"{domain}_score")
        pivot[score_col] = pivot[z_cols].mean(axis=1, skipna=True)
        pivot[f"{domain}_weight"] = domain_weights.get(domain, 1.0)

    score_cols = [col for col in domain_to_score_col.values() if col in pivot.columns]
    if not score_cols:
        raise ValueError("No domain score columns were computed.")
    pivot["DCLO_score"] = pivot[score_cols].mean(axis=1, skipna=True)
    pivot["n_domains_used"] = pivot[score_cols].notna().sum(axis=1)
    pivot["n_indicators_selected"] = len(selected_ids)

    use_confidence_weighting = bool(scoring_cfg.get("use_confidence_weighting", True))
    if use_confidence_weighting:
        weighted_num = pd.Series(0.0, index=pivot.index)
        weighted_den = pd.Series(0.0, index=pivot.index)
        for domain, score_col in domain_to_score_col.items():
            if score_col not in pivot.columns:
                continue
            w_col = f"{domain}_weight"
            w = pd.to_numeric(pivot.get(w_col, 1.0), errors="coerce").fillna(1.0)
            s = pd.to_numeric(pivot[score_col], errors="coerce")
            mask = s.notna()
            weighted_num += s.fillna(0.0) * w * mask.astype(float)
            weighted_den += w * mask.astype(float)
        pivot["DCLO_score_confidence_weighted"] = weighted_num / weighted_den.replace(0, np.nan)
    else:
        pivot["DCLO_score_confidence_weighted"] = pivot["DCLO_score"]

    min_domains_high = int(scoring_cfg.get("min_domains_high_trust", 4))
    min_domains_med = int(scoring_cfg.get("min_domains_medium_trust", 3))
    pivot["model_trust_tier"] = np.select(
        [
            pivot["n_domains_used"] >= min_domains_high,
            pivot["n_domains_used"] >= min_domains_med,
        ],
        ["High", "Medium"],
        default="Low",
    )
    return pivot.sort_values(["year", "economy"]).reset_index(drop=True)


def write_intake_markdown(intake_df: pd.DataFrame, md_path: Path) -> None:
    lines = [
        "# DPI Indicator Intake (Auto-generated)",
        "",
        "This file classifies indicators for country-year DCLO into `core_formative`, `context_only`, or `exclude` using balanced gates.",
        "",
        "## Gating Columns",
        "",
        "- `pct_observed_global`",
        "- `imputation_share`",
        "- `years_covered`",
        "- `ranking_score`",
        "",
        "## Intake Table (Top 120 rows)",
        "",
    ]
    preview = intake_df.sort_values(["role", "dclo_domain", "ranking_score"], ascending=[True, True, False]).head(120)
    cols = preview.columns.tolist()
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join(["---"] * len(cols)) + "|")
    for _, row in preview.iterrows():
        vals = [str(row.get(col, "")).replace("|", "/") for col in cols]
        lines.append("| " + " | ".join(vals) + " |")
    md_path.write_text("\n".join(lines), encoding="utf-8")


def run(config_path: str) -> None:
    config = read_config(config_path)
    inputs = config.get("inputs", {})
    gating = config.get("gating", {})
    output_cfg = config.get("output", {})
    scoring_cfg = config.get("scoring", {})

    long_df = pd.read_csv(inputs["dpi_long_path"])
    mapping_df = pd.read_csv(inputs["indicator_mapping_path"])
    coverage_df = pd.read_csv(inputs["coverage_by_indicator_path"])

    long_df = long_df.rename(columns={"indicator_name.y": "indicator_name"})
    if "indicator_name" not in long_df.columns and "indicator_name.x" in long_df.columns:
        long_df = long_df.rename(columns={"indicator_name.x": "indicator_name"})
    if "include_v2" in long_df.columns:
        long_df = long_df[long_df["include_v2"].astype(str).str.lower().eq("yes")].copy()

    # fill missing indicator metadata from mapping
    mapping_use = mapping_df[["indicator_code", "indicator_name", "pillar"]].drop_duplicates("indicator_code")
    long_df = long_df.merge(mapping_use, on="indicator_code", how="left", suffixes=("", "_map"))
    long_df["indicator_name"] = long_df["indicator_name"].fillna(long_df["indicator_name_map"])
    long_df["pillar"] = long_df["pillar"].fillna(long_df["pillar_map"])
    long_df = long_df.drop(columns=[col for col in ["indicator_name_map", "pillar_map"] if col in long_df.columns])

    long_df["year"] = pd.to_numeric(long_df["year"], errors="coerce").astype("Int64")
    long_df["was_imputed"] = long_df.get("was_imputed", False).astype(str).str.lower().isin(["true", "1", "yes"])

    year_min = int(gating.get("year_min", 2014))
    year_max = int(gating.get("year_max", 2025))
    long_df = long_df[(long_df["year"] >= year_min) & (long_df["year"] <= year_max)].copy()

    panel_wide = long_df.pivot_table(index=["economy", "year"], columns="indicator_code", values="norm_score", aggfunc="mean")
    panel_coverage = panel_wide.notna().mean(axis=0)
    stats = build_indicator_stats(long_df, coverage_df, panel_coverage=panel_coverage)
    intake_df, selected = select_indicators_by_domain(stats, gating, wide_df=panel_wide.reset_index(drop=True))
    country_scores = build_country_scores(long_df, selected, stats_df=stats, scoring_cfg=scoring_cfg)

    data_dir = Path(output_cfg.get("data_dir", "./data"))
    gold_dir = data_dir / "gold"
    gold_dir.mkdir(parents=True, exist_ok=True)

    country_out = gold_dir / output_cfg.get("country_output_file", "dclo_country_year.csv")
    intake_out_csv = gold_dir / output_cfg.get("intake_csv_file", "dpi_indicator_intake.csv")
    selected_out_json = gold_dir / output_cfg.get("selected_json_file", "dpi_selected_indicators_by_domain.json")
    intake_out_md = Path(__file__).resolve().parents[2] / "docs" / "dpi-indicator-intake.md"

    country_scores.to_csv(country_out, index=False)
    intake_df.sort_values(["dclo_domain", "role", "ranking_score"], ascending=[True, True, False]).to_csv(
        intake_out_csv, index=False
    )
    selected_out_json.write_text(json.dumps(selected, indent=2), encoding="utf-8")
    write_intake_markdown(intake_df, intake_out_md)
    print(f"Wrote country scores: {country_out}")
    print(f"Wrote intake csv: {intake_out_csv}")
    print(f"Wrote selected indicators json: {selected_out_json}")
    print(f"Wrote intake markdown: {intake_out_md}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build country-year DCLO from DPI long panel")
    parser.add_argument("--config", required=True, help="Path to country-year DPI config")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
