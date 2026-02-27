import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


def compute_vif(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    rows = []
    clean_cols = [col for col in columns if col in df.columns]
    if len(clean_cols) < 2:
        return pd.DataFrame(columns=["indicator_code", "vif"])

    xdf = df[clean_cols].copy()
    for col in clean_cols:
        xdf[col] = pd.to_numeric(xdf[col], errors="coerce")
    xdf = xdf.dropna()
    # Standardize to avoid overflow with mixed indicator scales.
    xdf = (xdf - xdf.mean()) / xdf.std(ddof=0).replace(0, np.nan)
    xdf = xdf.replace([np.inf, -np.inf], np.nan).dropna()
    if len(xdf) < 5:
        return pd.DataFrame(columns=["indicator_code", "vif"])

    for target in clean_cols:
        y = xdf[target].to_numpy()
        x_cols = [col for col in clean_cols if col != target]
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


def run(data_dir: str) -> None:
    root = Path(data_dir)
    country_path = root / "dclo_country_year.csv"
    intake_path = root / "dpi_indicator_intake.csv"
    selected_path = root / "dpi_selected_indicators_by_domain.json"

    if not country_path.exists():
        raise FileNotFoundError(f"Missing country-year output: {country_path}")
    if not intake_path.exists():
        raise FileNotFoundError(f"Missing intake output: {intake_path}")
    if not selected_path.exists():
        raise FileNotFoundError(f"Missing selected-indicators file: {selected_path}")

    country = pd.read_csv(country_path)
    intake = pd.read_csv(intake_path)
    selected: Dict[str, List[str]] = json.loads(selected_path.read_text(encoding="utf-8"))

    core = intake[intake["role"] == "core_formative"].copy()
    core = core[["indicator_code", "dclo_domain", "pct_observed_global", "imputation_share", "years_covered", "ranking_score"]]

    missing_rows = []
    for indicator in core["indicator_code"].tolist():
        if indicator not in country.columns:
            missing_rows.append({"indicator_code": indicator, "missingness": 1.0, "n_obs": 0})
            continue
        s = pd.to_numeric(country[indicator], errors="coerce")
        missingness = float(s.isna().mean())
        n_obs = int(s.notna().sum())
        missing_rows.append({"indicator_code": indicator, "missingness": missingness, "n_obs": n_obs})
    missing_df = pd.DataFrame(missing_rows).merge(core[["indicator_code", "dclo_domain"]], on="indicator_code", how="left")
    missing_df = missing_df.sort_values(["dclo_domain", "missingness"], ascending=[True, False])

    core_cols = [col for col in core["indicator_code"].tolist() if col in country.columns]
    corr_df = pd.DataFrame()
    if core_cols:
        corr = country[core_cols].apply(pd.to_numeric, errors="coerce").corr()
        corr_df = (
            corr.stack()
            .reset_index()
            .rename(columns={"level_0": "indicator_a", "level_1": "indicator_b", 0: "correlation"})
        )
        corr_df = corr_df[corr_df["indicator_a"] < corr_df["indicator_b"]].sort_values("correlation", key=lambda s: s.abs(), ascending=False)

    vif_frames = []
    for domain, indicators in selected.items():
        vif_df = compute_vif(country, indicators)
        if vif_df.empty:
            continue
        vif_df["dclo_domain"] = domain
        vif_frames.append(vif_df)
    vif_all = pd.concat(vif_frames, ignore_index=True) if vif_frames else pd.DataFrame(columns=["indicator_code", "vif", "dclo_domain"])
    vif_all = vif_all.sort_values(["dclo_domain", "vif"], ascending=[True, False])

    out_missing = root / "dclo_indicator_missingness_report.csv"
    out_corr = root / "dclo_indicator_correlation_report.csv"
    out_vif = root / "dclo_indicator_vif_report.csv"
    missing_df.to_csv(out_missing, index=False)
    corr_df.to_csv(out_corr, index=False)
    vif_all.to_csv(out_vif, index=False)

    print(f"Wrote missingness report: {out_missing}")
    print(f"Wrote correlation report: {out_corr}")
    print(f"Wrote VIF report: {out_vif}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate selected DCLO country indicators")
    parser.add_argument(
        "--data-dir",
        default="./data/gold",
        help="Directory containing dclo_country_year.csv and indicator selection outputs",
    )
    args = parser.parse_args()
    run(args.data_dir)


if __name__ == "__main__":
    main()
