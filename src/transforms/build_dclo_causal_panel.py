import argparse
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from quality.audit_logger import AuditLogger
from quality.data_verification import (
    VerificationResult,
    verify_minimum_sample_size,
    verify_no_nulls,
    verify_panel_balance,
    verify_required_columns,
    verify_year_coverage,
)


def read_config(config_path: str) -> Dict[str, object]:
    with open(config_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def normal_p_value(t_stat: float) -> float:
    if pd.isna(t_stat):
        return np.nan
    return float(math.erfc(abs(float(t_stat)) / math.sqrt(2.0)))


def two_way_demean(df: pd.DataFrame, columns: List[str], entity_col: str, time_col: str) -> pd.DataFrame:
    """Two-way demeaning for entity and time fixed effects.

    Implements the within-transformation for two-way FE per
    Wooldridge (2010, Ch. 10). Removes entity means and time means,
    then adds back the grand mean to preserve scale.
    """
    out = df[columns].copy()
    grand_mean = out.mean()
    entity_mean = df.groupby(entity_col)[columns].transform("mean")
    time_mean = df.groupby(time_col)[columns].transform("mean")
    return out - entity_mean - time_mean + grand_mean


def cluster_robust_covariance(X: np.ndarray, residuals: np.ndarray, clusters: np.ndarray) -> np.ndarray:
    """Cluster-robust (sandwich) covariance estimator.

    Implements the CRVE per Cameron & Miller (2015) with finite-sample
    correction factor G/(G-1) * (N-1)/(N-K) where G = number of clusters.
    """
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
    """Fit panel OLS with optional TWFE and cluster-robust inference.

    Returns coefficient estimates, fit statistics (including degrees of
    freedom and F-statistic per Wooldridge 2010), and prediction DataFrame.
    """
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

    # Compute fit statistics including degrees of freedom and F-statistic
    n_obs = int(len(use))
    n_entities = int(use[entity_col].nunique())
    n_years = int(use[time_col].nunique())
    n_clusters = int(use[cluster_col].nunique())
    k_regressors = len(regressors)

    # Degrees of freedom accounting for absorbed fixed effects
    df_absorbed = 0
    if use_entity_fe:
        df_absorbed += n_entities - 1
    if use_time_fe:
        df_absorbed += n_years - 1
    df_residual = max(n_obs - k_regressors - df_absorbed, 1)

    y_bar = float(np.mean(y))
    ss_tot = float(np.sum((y - y_bar) ** 2))
    ss_res = float(np.sum(residuals**2))
    r2_within = np.nan if ss_tot == 0 else 1.0 - (ss_res / ss_tot)

    # F-statistic: joint significance of all regressors
    # F = (R2 / k) / ((1 - R2) / (N - k - 1))
    if pd.notna(r2_within) and r2_within < 1.0 and df_residual > 0 and k_regressors > 0:
        f_stat = (r2_within / k_regressors) / ((1.0 - r2_within) / df_residual)
    else:
        f_stat = np.nan

    # Adjusted R-squared
    if pd.notna(r2_within) and (n_obs - k_regressors - 1) > 0:
        r2_adj = 1.0 - (1.0 - r2_within) * (n_obs - 1) / (n_obs - k_regressors - 1)
    else:
        r2_adj = np.nan

    # Durbin-Watson statistic for residual autocorrelation
    if len(residuals) > 1:
        dw_num = float(np.sum(np.diff(residuals) ** 2))
        dw_den = float(np.sum(residuals**2))
        durbin_watson = dw_num / dw_den if dw_den > 0 else np.nan
    else:
        durbin_watson = np.nan

    fit = {
        "n_obs": n_obs,
        "n_entities": n_entities,
        "n_years": n_years,
        "n_clusters": n_clusters,
        "k_regressors": k_regressors,
        "df_residual": df_residual,
        "df_absorbed_fe": df_absorbed,
        "r2_within": float(r2_within) if not pd.isna(r2_within) else np.nan,
        "r2_adjusted": float(r2_adj) if not pd.isna(r2_adj) else np.nan,
        "f_statistic": float(f_stat) if not pd.isna(f_stat) else np.nan,
        "residual_std": float(np.std(residuals, ddof=1)) if len(residuals) > 1 else np.nan,
        "durbin_watson": float(durbin_watson) if not pd.isna(durbin_watson) else np.nan,
        "ss_residual": float(ss_res),
        "ss_total": float(ss_tot),
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


def permute_columns_within_time(df: pd.DataFrame, columns: List[str], time_col: str, seed: int) -> pd.DataFrame:
    out = df.copy()
    rng = np.random.default_rng(seed)
    for year, idx in out.groupby(time_col).groups.items():
        row_idx = np.array(list(idx))
        for col in columns:
            vals = out.loc[row_idx, col].to_numpy()
            if len(vals) <= 1:
                continue
            out.loc[row_idx, col] = vals[rng.permutation(len(vals))]
    return out


def fit_spec(data: pd.DataFrame, spec: Dict[str, object], panel_cfg: Dict[str, object], spec_kind: str,
             audit: AuditLogger) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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
    permute_within_year = bool(spec.get("permute_within_year", False))
    permute_seed = int(spec.get("permute_seed", 0))

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

    if permute_within_year and regressors:
        audit.record_random_seed(f"placebo_{spec_id}", permute_seed)
        work = permute_columns_within_time(work, regressors, time_col=time_col, seed=permute_seed)

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

    audit.record_stage(
        f"fit_{spec_kind}_{spec_id}",
        rows_in=len(work),
        rows_out=fit_metrics.get("n_obs", 0),
        rows_dropped=len(work) - fit_metrics.get("n_obs", 0),
        notes=(
            f"R2_within={fit_metrics.get('r2_within', 'n/a'):.4f}, "
            f"F={fit_metrics.get('f_statistic', 'n/a')}, "
            f"df_resid={fit_metrics.get('df_residual', 'n/a')}, "
            f"DW={fit_metrics.get('durbin_watson', 'n/a')}"
        ) if fit_metrics else "Empty model",
    )

    if not pred_df.empty:
        pred_df = pred_df.rename(columns={entity_col: "entity", time_col: "year"})
        pred_df.insert(0, "spec_id", spec_id)
    return coef_df, fit_df, pred_df


def build_rank_stability(country_df: pd.DataFrame, draws: int, top_k: int, seed: int,
                         audit: AuditLogger) -> pd.DataFrame:
    audit.record_random_seed("rank_stability", seed)
    audit.record_parameter("rank_stability_draws", draws)
    audit.record_parameter("rank_stability_top_k", top_k)

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

    audit.record_stage("build_rank_stability", rows_in=len(use), rows_out=len(stability), rows_dropped=0,
                       notes=f"{draws} Dirichlet draws, top_k={top_k}, seed={seed}")
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

    audit = AuditLogger(pipeline_name="build_dclo_causal_panel")
    audit.record_config(cfg)

    country_path = Path(str(inputs.get("country_scores_path", "./data/gold/dclo_country_year.csv")))
    audit.record_input("country_scores", str(country_path))

    country_df = pd.read_csv(country_path)
    country_df["year"] = pd.to_numeric(country_df["year"], errors="coerce").astype("Int64")

    entity_col = str(panel_cfg.get("entity_col", "economy"))
    time_col = str(panel_cfg.get("time_col", "year"))
    min_obs_per_entity = int(panel_cfg.get("min_obs_per_entity", 5))
    min_entities = int(panel_cfg.get("min_entities", 25))

    # Verify input data
    vr = VerificationResult("causal_input")
    vr = verify_required_columns(country_df, [entity_col, time_col, "DCLO_score"], vr)
    vr = verify_minimum_sample_size(country_df, min_rows=50, result=vr, context="country scores for causal panel")
    vr = verify_no_nulls(country_df, [entity_col, time_col], vr)

    panel = country_df.copy()
    rows_raw = len(panel)
    panel = panel.dropna(subset=[entity_col, time_col]).copy()
    if "model_trust_tier" in panel.columns and "model_trust_tier_numeric" not in panel.columns:
        panel["model_trust_tier_numeric"] = (
            panel["model_trust_tier"].map({"High": 2.0, "Medium": 1.0, "Low": 0.0}).fillna(0.0)
        )
    counts = panel.groupby(entity_col)[time_col].nunique()
    keep_entities = counts[counts >= min_obs_per_entity].index.tolist()
    panel = panel[panel[entity_col].isin(keep_entities)].copy()

    audit.record_stage("panel_filtering", rows_in=rows_raw, rows_out=len(panel),
                       rows_dropped=rows_raw - len(panel),
                       notes=f"Kept entities with >= {min_obs_per_entity} obs, {panel[entity_col].nunique()} entities remain")

    if panel[entity_col].nunique() < min_entities:
        raise ValueError(
            f"Insufficient panel coverage after filtering: found {panel[entity_col].nunique()} entities, require {min_entities}."
        )

    # Panel balance verification
    vr = verify_panel_balance(panel, entity_col, time_col, vr)
    vr = verify_year_coverage(panel, time_col, 2000, 2030, vr)

    audit.record_parameter("panel_entity_col", entity_col)
    audit.record_parameter("panel_time_col", time_col)
    audit.record_parameter("panel_min_obs_per_entity", min_obs_per_entity)
    audit.record_parameter("panel_min_entities", min_entities)
    audit.record_parameter("panel_n_entities", int(panel[entity_col].nunique()))
    audit.record_parameter("panel_n_years", int(panel[time_col].nunique()))
    audit.record_parameter("panel_n_obs", len(panel))

    all_coef: List[pd.DataFrame] = []
    all_fit: List[pd.DataFrame] = []
    all_pred: List[pd.DataFrame] = []

    baseline_spec = cfg.get("baseline_spec", {})
    coef_df, fit_df, pred_df = fit_spec(panel, baseline_spec, panel_cfg=panel_cfg, spec_kind="baseline", audit=audit)
    all_coef.append(coef_df)
    all_fit.append(fit_df)
    all_pred.append(pred_df)

    for spec in cfg.get("robustness_specs", []):
        coef_df, fit_df, pred_df = fit_spec(panel, spec, panel_cfg=panel_cfg, spec_kind="robustness", audit=audit)
        all_coef.append(coef_df)
        all_fit.append(fit_df)
        all_pred.append(pred_df)

    placebo_spec = cfg.get("placebo_spec", {})
    coef_df, fit_df, pred_df = fit_spec(panel, placebo_spec, panel_cfg=panel_cfg, spec_kind="placebo", audit=audit)
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
            audit=audit,
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

    for name, path in [("coefficients", coef_path), ("model_fit", fit_path),
                       ("rank_stability", stab_path), ("method_comparison", method_path)]:
        audit.record_output(name, str(path))

    # Write verification report
    import json
    verification_path = gold_dir / "dclo_causal_panel_verification.json"
    verification_path.write_text(json.dumps(vr.to_dict(), indent=2, default=str), encoding="utf-8")
    audit.record_output("verification_report", str(verification_path))

    # Write audit manifest
    manifest_path = gold_dir / "dclo_causal_panel_audit_manifest.json"
    audit.write_manifest(str(manifest_path))

    print(f"Wrote causal coefficients: {coef_path}")
    print(f"Wrote causal model fit: {fit_path}")
    print(f"Wrote rank stability: {stab_path}")
    print(f"Wrote method comparison: {method_path}")
    print(f"Wrote verification report: {verification_path}")
    print(f"Wrote audit manifest: {manifest_path}")
    for line in audit.get_summary_lines():
        print(f"  {line}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build causal panel outputs for DCLO")
    parser.add_argument("--config", required=True, help="Path to causal model config")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
