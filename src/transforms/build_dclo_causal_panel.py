import argparse
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml


def read_config(config_path: str) -> Dict[str, object]:
    with open(config_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def normal_p_value(t_stat: float) -> float:
    if pd.isna(t_stat):
        return np.nan
    return float(math.erfc(abs(float(t_stat)) / math.sqrt(2.0)))


def two_way_demean(df: pd.DataFrame, columns: List[str], entity_col: str, time_col: str) -> pd.DataFrame:
    out = df[columns].copy()
    grand_mean = out.mean()
    entity_mean = df.groupby(entity_col)[columns].transform("mean")
    time_mean = df.groupby(time_col)[columns].transform("mean")
    return out - entity_mean - time_mean + grand_mean


def cluster_robust_covariance(X: np.ndarray, residuals: np.ndarray, clusters: np.ndarray) -> np.ndarray:
    xtx_inv = np.linalg.pinv(X.T @ X)
    meat = np.zeros((X.shape[1], X.shape[1]))
    unique_clusters = pd.Series(clusters).dropna().unique().tolist()
    for cluster in unique_clusters:
        idx = np.where(clusters == cluster)[0]
        if len(idx) == 0:
            continue
        xg = X[idx, :]
        ug = residuals[idx]
        xu = xg.T @ ug
        meat += np.outer(xu, xu)

    n_obs = X.shape[0]
    n_params = X.shape[1]
    n_clusters = max(len(unique_clusters), 1)
    if n_obs <= n_params:
        correction = 1.0
    else:
        correction = (n_clusters / max(n_clusters - 1, 1)) * ((n_obs - 1) / max(n_obs - n_params, 1))
    return correction * (xtx_inv @ meat @ xtx_inv)


def fit_panel_ols(
    df: pd.DataFrame,
    outcome: str,
    regressors: List[str],
    entity_col: str,
    time_col: str,
    use_entity_fe: bool,
    use_time_fe: bool,
    cluster_col: str,
) -> Tuple[pd.DataFrame, Dict[str, float], pd.DataFrame]:
    needed_cols = list(dict.fromkeys([entity_col, time_col, cluster_col, outcome] + regressors))
    use = df[needed_cols].copy()
    for col in [outcome] + regressors:
        use[col] = pd.to_numeric(use[col], errors="coerce")
    use = use.dropna(subset=[outcome] + regressors + [entity_col, time_col, cluster_col]).copy()
    if use.empty or len(use) < (len(regressors) + 5):
        return pd.DataFrame(), {}, pd.DataFrame()

    model_cols = [outcome] + regressors
    if use_entity_fe and use_time_fe:
        transformed = two_way_demean(use, model_cols, entity_col=entity_col, time_col=time_col)
    elif use_entity_fe:
        transformed = use[model_cols] - use.groupby(entity_col)[model_cols].transform("mean")
    elif use_time_fe:
        transformed = use[model_cols] - use.groupby(time_col)[model_cols].transform("mean")
    else:
        transformed = use[model_cols].copy()

    transformed = transformed.dropna(subset=model_cols).copy()
    use = use.loc[transformed.index].copy()
    if transformed.empty or len(transformed) < (len(regressors) + 5):
        return pd.DataFrame(), {}, pd.DataFrame()

    y = transformed[outcome].to_numpy(dtype=float)
    X = transformed[regressors].to_numpy(dtype=float)
    beta = np.linalg.pinv(X.T @ X) @ (X.T @ y)
    fitted = X @ beta
    residuals = y - fitted

    cov = cluster_robust_covariance(X, residuals, use[cluster_col].to_numpy())
    se = np.sqrt(np.clip(np.diag(cov), a_min=0.0, a_max=None))
    t_stats = beta / np.where(se == 0, np.nan, se)
    p_vals = np.array([normal_p_value(t) for t in t_stats])
    ci_low = beta - 1.96 * se
    ci_high = beta + 1.96 * se

    coef_df = pd.DataFrame(
        {
            "predictor": regressors,
            "coef": beta,
            "std_error": se,
            "t_stat": t_stats,
            "p_value_norm_approx": p_vals,
            "ci_low_95": ci_low,
            "ci_high_95": ci_high,
        }
    )

    y_bar = float(np.mean(y))
    ss_tot = float(np.sum((y - y_bar) ** 2))
    ss_res = float(np.sum(residuals**2))
    r2_within = np.nan if ss_tot == 0 else 1.0 - (ss_res / ss_tot)
    fit = {
        "n_obs": int(len(use)),
        "n_entities": int(use[entity_col].nunique()),
        "n_years": int(use[time_col].nunique()),
        "r2_within": float(r2_within) if not pd.isna(r2_within) else np.nan,
        "residual_std": float(np.std(residuals, ddof=1)) if len(residuals) > 1 else np.nan,
    }

    pred_df = use[[entity_col, time_col]].copy()
    pred_df["fitted"] = fitted
    pred_df["residual"] = residuals
    pred_df["actual"] = y
    return coef_df, fit, pred_df


def add_lagged_columns(df: pd.DataFrame, vars_to_lag: List[str], lag: int, entity_col: str, time_col: str) -> pd.DataFrame:
    out = df.sort_values([entity_col, time_col]).copy()
    for col in vars_to_lag:
        out[f"{col}_lag{lag}"] = out.groupby(entity_col)[col].shift(lag)
    return out


def add_leadged_columns(df: pd.DataFrame, vars_to_lead: List[str], lead: int, entity_col: str, time_col: str) -> pd.DataFrame:
    out = df.sort_values([entity_col, time_col]).copy()
    for col in vars_to_lead:
        out[f"{col}_lead{lead}"] = out.groupby(entity_col)[col].shift(-lead)
    return out


def fit_spec(data: pd.DataFrame, spec: Dict[str, object], panel_cfg: Dict[str, object], spec_kind: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    entity_col = str(panel_cfg.get("entity_col", "economy"))
    time_col = str(panel_cfg.get("time_col", "year"))
    spec_id = str(spec.get("id", "spec"))
    outcome = str(spec["outcome"])
    predictors = [str(c) for c in spec.get("predictors", [])]
    controls = [str(c) for c in spec.get("controls", [])]
    use_entity_fe = bool(spec.get("entity_fixed_effects", True))
    use_time_fe = bool(spec.get("time_fixed_effects", True))
    cluster_col = str(spec.get("cluster_col", entity_col))
    lag = int(spec.get("lag", 0))
    lead = int(spec.get("lead", 0))

    work = data.copy()
    regressors: List[str] = []

    if lag > 0:
        work = add_lagged_columns(work, predictors + controls, lag=lag, entity_col=entity_col, time_col=time_col)
        regressors.extend([f"{c}_lag{lag}" for c in predictors])
        regressors.extend([f"{c}_lag{lag}" for c in controls])
    elif lead > 0:
        work = add_leadged_columns(work, predictors + controls, lead=lead, entity_col=entity_col, time_col=time_col)
        regressors.extend([f"{c}_lead{lead}" for c in predictors])
        regressors.extend([f"{c}_lead{lead}" for c in controls])
    else:
        regressors.extend(predictors + controls)

    coef_df, fit_metrics, pred_df = fit_panel_ols(
        work,
        outcome=outcome,
        regressors=regressors,
        entity_col=entity_col,
        time_col=time_col,
        use_entity_fe=use_entity_fe,
        use_time_fe=use_time_fe,
        cluster_col=cluster_col,
    )
    if coef_df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    coef_df.insert(0, "spec_kind", spec_kind)
    coef_df.insert(1, "spec_id", spec_id)
    coef_df.insert(2, "outcome", outcome)
    coef_df["lag"] = lag
    coef_df["lead"] = lead

    fit_df = pd.DataFrame(
        [
            {
                "spec_kind": spec_kind,
                "spec_id": spec_id,
                "outcome": outcome,
                "lag": lag,
                "lead": lead,
                "entity_fixed_effects": use_entity_fe,
                "time_fixed_effects": use_time_fe,
                **fit_metrics,
            }
        ]
    )

    if not pred_df.empty:
        pred_df = pred_df.rename(columns={entity_col: "entity", time_col: "year"})
        pred_df.insert(0, "spec_id", spec_id)
    return coef_df, fit_df, pred_df


def build_rank_stability(country_df: pd.DataFrame, draws: int, top_k: int, seed: int) -> pd.DataFrame:
    domain_cols = [col for col in ["ACC_score", "SKL_score", "SRV_score", "AGR_score", "ECO_score", "OUT_score"] if col in country_df.columns]
    use = country_df.dropna(subset=["economy", "year"]).copy()
    if not domain_cols or use.empty:
        return pd.DataFrame()

    rng = np.random.default_rng(seed)
    rows: List[Dict[str, object]] = []
    for year, year_df in use.groupby("year"):
        sample = year_df[["economy"] + domain_cols].copy()
        for draw in range(draws):
            weights = rng.dirichlet(alpha=np.ones(len(domain_cols)))
            score = np.zeros(len(sample), dtype=float)
            for row_idx in range(len(sample)):
                row_vals = pd.to_numeric(sample.iloc[row_idx][domain_cols], errors="coerce")
                mask = row_vals.notna().to_numpy()
                if mask.sum() == 0:
                    score[row_idx] = np.nan
                    continue
                w = weights[mask]
                w = w / np.sum(w)
                score[row_idx] = float(np.dot(row_vals[mask].to_numpy(dtype=float), w))
            rank_series = pd.Series(score, index=sample["economy"]).rank(ascending=False, method="average")
            top_entities = set(rank_series[rank_series <= top_k].index.tolist())
            for economy, rank_val in rank_series.items():
                rows.append(
                    {
                        "year": int(year),
                        "economy": str(economy),
                        "draw": draw,
                        "rank": float(rank_val),
                        "in_top_k": int(economy in top_entities),
                    }
                )

    if not rows:
        return pd.DataFrame()
    raw = pd.DataFrame(rows)
    stability = raw.groupby(["year", "economy"], as_index=False).agg(
        mean_rank=("rank", "mean"),
        sd_rank=("rank", "std"),
        top_k_freq=("in_top_k", "mean"),
    )
    return stability


def build_method_comparison(country_df: pd.DataFrame, pred_df: pd.DataFrame, baseline_spec_id: str) -> pd.DataFrame:
    if pred_df.empty:
        return pd.DataFrame()

    base = country_df[["economy", "year", "DCLO_score", "DCLO_score_confidence_weighted"]].copy()
    pred = pred_df[pred_df["spec_id"].eq(baseline_spec_id)][["entity", "year", "fitted"]].rename(
        columns={"entity": "economy", "fitted": "causal_signal_score"}
    )
    merged = base.merge(pred, on=["economy", "year"], how="inner")
    if merged.empty:
        return pd.DataFrame()

    rows: List[Dict[str, object]] = []
    for year, grp in merged.groupby("year"):
        n = len(grp)
        if n < 5:
            continue
        score_cols = ["DCLO_score", "DCLO_score_confidence_weighted", "causal_signal_score"]
        corr = grp[score_cols].corr(method="spearman")
        rows.append(
            {
                "year": int(year),
                "n_entities": int(n),
                "rho_baseline_vs_weighted": float(corr.loc["DCLO_score", "DCLO_score_confidence_weighted"]),
                "rho_baseline_vs_causal": float(corr.loc["DCLO_score", "causal_signal_score"]),
                "rho_weighted_vs_causal": float(corr.loc["DCLO_score_confidence_weighted", "causal_signal_score"]),
            }
        )
    return pd.DataFrame(rows)


def run(config_path: str) -> None:
    cfg = read_config(config_path)
    inputs = cfg.get("inputs", {})
    panel_cfg = cfg.get("panel", {})
    output_cfg = cfg.get("output", {})

    country_path = Path(str(inputs.get("country_scores_path", "./data/gold/dclo_country_year.csv")))
    country_df = pd.read_csv(country_path)
    country_df["year"] = pd.to_numeric(country_df["year"], errors="coerce").astype("Int64")

    entity_col = str(panel_cfg.get("entity_col", "economy"))
    time_col = str(panel_cfg.get("time_col", "year"))
    min_obs_per_entity = int(panel_cfg.get("min_obs_per_entity", 5))
    min_entities = int(panel_cfg.get("min_entities", 25))

    panel = country_df.copy()
    panel = panel.dropna(subset=[entity_col, time_col]).copy()
    if "model_trust_tier" in panel.columns and "model_trust_tier_numeric" not in panel.columns:
        panel["model_trust_tier_numeric"] = (
            panel["model_trust_tier"].map({"High": 2.0, "Medium": 1.0, "Low": 0.0}).fillna(0.0)
        )
    counts = panel.groupby(entity_col)[time_col].nunique()
    keep_entities = counts[counts >= min_obs_per_entity].index.tolist()
    panel = panel[panel[entity_col].isin(keep_entities)].copy()
    if panel[entity_col].nunique() < min_entities:
        raise ValueError(
            f"Insufficient panel coverage after filtering: found {panel[entity_col].nunique()} entities, require {min_entities}."
        )

    all_coef: List[pd.DataFrame] = []
    all_fit: List[pd.DataFrame] = []
    all_pred: List[pd.DataFrame] = []

    baseline_spec = cfg.get("baseline_spec", {})
    coef_df, fit_df, pred_df = fit_spec(panel, baseline_spec, panel_cfg=panel_cfg, spec_kind="baseline")
    all_coef.append(coef_df)
    all_fit.append(fit_df)
    all_pred.append(pred_df)

    for spec in cfg.get("robustness_specs", []):
        coef_df, fit_df, pred_df = fit_spec(panel, spec, panel_cfg=panel_cfg, spec_kind="robustness")
        all_coef.append(coef_df)
        all_fit.append(fit_df)
        all_pred.append(pred_df)

    placebo_spec = cfg.get("placebo_spec", {})
    coef_df, fit_df, pred_df = fit_spec(panel, placebo_spec, panel_cfg=panel_cfg, spec_kind="placebo")
    all_coef.append(coef_df)
    all_fit.append(fit_df)
    all_pred.append(pred_df)

    coef_frames = [df for df in all_coef if not df.empty]
    fit_frames = [df for df in all_fit if not df.empty]
    pred_frames = [df for df in all_pred if not df.empty]
    if not coef_frames or not fit_frames:
        raise ValueError("No model specification could be estimated with the current data coverage and config.")

    coef_out = pd.concat(coef_frames, ignore_index=True)
    fit_out = pd.concat(fit_frames, ignore_index=True)
    pred_out = pd.concat(pred_frames, ignore_index=True) if pred_frames else pd.DataFrame()

    stability_cfg = cfg.get("stability", {})
    if bool(stability_cfg.get("enabled", True)):
        rank_stability = build_rank_stability(
            panel,
            draws=int(stability_cfg.get("draws", 250)),
            top_k=int(stability_cfg.get("top_k", 20)),
            seed=int(stability_cfg.get("random_seed", 42)),
        )
    else:
        rank_stability = pd.DataFrame()

    baseline_spec_id = str(baseline_spec.get("id", "baseline"))
    method_compare = build_method_comparison(panel, pred_out, baseline_spec_id=baseline_spec_id)

    data_dir = Path(str(output_cfg.get("data_dir", "./data")))
    gold_dir = data_dir / "gold"
    gold_dir.mkdir(parents=True, exist_ok=True)

    coef_path = gold_dir / str(output_cfg.get("coefficients_file", "dclo_causal_coefficients.csv"))
    fit_path = gold_dir / str(output_cfg.get("model_fit_file", "dclo_causal_model_fit.csv"))
    stab_path = gold_dir / str(output_cfg.get("rank_stability_file", "dclo_rank_stability.csv"))
    method_path = gold_dir / str(output_cfg.get("method_comparison_file", "dclo_method_comparison.csv"))

    coef_out.to_csv(coef_path, index=False)
    fit_out.to_csv(fit_path, index=False)
    rank_stability.to_csv(stab_path, index=False)
    method_compare.to_csv(method_path, index=False)

    print(f"Wrote causal coefficients: {coef_path}")
    print(f"Wrote causal model fit: {fit_path}")
    print(f"Wrote rank stability: {stab_path}")
    print(f"Wrote method comparison: {method_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build causal panel outputs for DCLO")
    parser.add_argument("--config", required=True, help="Path to causal model config")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
