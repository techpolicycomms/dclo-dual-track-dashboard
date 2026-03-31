"""Data verification module for DCLO pipeline.

Implements pre-transform validation gates aligned with academic best practices:
- Nunnally & Bernstein (1994): measurement reliability via range/type checks
- OECD (2008) Handbook on Composite Indicators: indicator screening
- Wooldridge (2010): panel data diagnostics for econometric validity
- Christensen & Miguel (2018): pre-analysis plan style verification

All verification functions return a VerificationResult that accumulates
issues without halting the pipeline, enabling full audit trail generation.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class VerificationResult:
    """Accumulates verification issues for a single dataset or stage."""

    dataset_name: str
    checks_run: int = 0
    checks_passed: int = 0
    issues: List[Dict[str, Any]] = field(default_factory=list)
    dropped_records: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return len(self.issues) == 0

    @property
    def n_issues(self) -> int:
        return len(self.issues)

    def add_issue(self, check: str, severity: str, message: str, detail: Any = None) -> None:
        self.issues.append(
            {"check": check, "severity": severity, "message": message, "detail": detail}
        )

    def add_dropped(self, reason: str, count: int, examples: Optional[List[Any]] = None) -> None:
        entry: Dict[str, Any] = {"reason": reason, "count": count}
        if examples:
            entry["examples"] = examples[:5]
        self.dropped_records.append(entry)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset": self.dataset_name,
            "checks_run": self.checks_run,
            "checks_passed": self.checks_passed,
            "n_issues": self.n_issues,
            "passed": self.passed,
            "issues": self.issues,
            "dropped_records": self.dropped_records,
        }


# ---------------------------------------------------------------------------
# Range specifications for known indicator types
# ---------------------------------------------------------------------------

# Indicators expressed as percentages (0-100)
PERCENTAGE_INDICATORS = {
    "pop_hh_elec",
    "fem_literacy",
    "pop_hh_sf",
    "fem_15_24_hyg_period",
    "prop_hh_microfin",
    "prop_saving",
    "ACC_pop_hh_elec",
    "SKL_fem_literacy",
    "SRV_pop_hh_sf",
    "AGR_fem_15_24_hyg_period",
    "ECO_prop_hh_microfin",
    "OUT_prop_saving",
    "pct_observed_global",
    "imputation_share",
    "panel_coverage",
}

# Indicators that must be non-negative
NON_NEGATIVE_INDICATORS = {
    "hh_income_monthly",
    "OUT_hh_income_monthly",
    "AGR_shg_member_scale",
    "ECO_shg_credit_scale",
    "total_vol",
    "total_val",
    "p2p_vol",
    "p2m_vol",
    "no_of_transactions",
    "amt_of_transactions",
    "active_users",
    "NAT_upi_total_vol",
    "NAT_upi_total_val",
    "NAT_upi_p2p_vol",
    "NAT_upi_p2m_vol",
    "NAT_ib_no_of_transactions",
    "NAT_ib_amt_of_transactions",
    "NAT_ib_active_users",
    "n_obs",
    "n_entities",
    "n_years",
    "years_covered",
    "economies_covered",
}


def verify_required_columns(
    df: pd.DataFrame, required: List[str], result: VerificationResult
) -> VerificationResult:
    """Check that all required columns exist."""
    result.checks_run += 1
    missing = [col for col in required if col not in df.columns]
    if missing:
        result.add_issue(
            "required_columns",
            "error",
            f"Missing required columns: {missing}",
            detail=missing,
        )
    else:
        result.checks_passed += 1
    return result


def verify_no_duplicates(
    df: pd.DataFrame, key_cols: List[str], result: VerificationResult
) -> Tuple[pd.DataFrame, VerificationResult]:
    """Check for and remove duplicate rows on key columns. Returns cleaned df."""
    result.checks_run += 1
    available_keys = [col for col in key_cols if col in df.columns]
    if not available_keys:
        result.checks_passed += 1
        return df, result

    dupes = df.duplicated(subset=available_keys, keep=False)
    n_dupes = int(dupes.sum())
    if n_dupes > 0:
        dupe_examples = df[dupes][available_keys].head(5).to_dict("records")
        result.add_issue(
            "no_duplicates",
            "warning",
            f"{n_dupes} duplicate rows on {available_keys}",
            detail={"count": n_dupes, "examples": dupe_examples},
        )
        # Keep first occurrence, drop rest
        keep_mask = ~df.duplicated(subset=available_keys, keep="first")
        n_dropped = int((~keep_mask).sum())
        result.add_dropped("duplicate_key", n_dropped)
        df = df[keep_mask].copy()
    else:
        result.checks_passed += 1
    return df, result


def verify_no_nulls(
    df: pd.DataFrame, columns: List[str], result: VerificationResult
) -> VerificationResult:
    """Check for null values in critical columns."""
    for col in columns:
        result.checks_run += 1
        if col not in df.columns:
            continue
        n_null = int(df[col].isna().sum())
        if n_null > 0:
            pct = 100.0 * n_null / max(len(df), 1)
            result.add_issue(
                f"no_nulls_{col}",
                "warning" if pct < 20.0 else "error",
                f"{col}: {n_null} null values ({pct:.1f}%)",
                detail={"column": col, "null_count": n_null, "null_pct": round(pct, 2)},
            )
        else:
            result.checks_passed += 1
    return result


def verify_value_ranges(
    df: pd.DataFrame, result: VerificationResult
) -> Tuple[pd.DataFrame, VerificationResult]:
    """Validate that known indicators fall within plausible ranges.

    Clamps out-of-range values and logs them as issues.
    - Percentage indicators: [0, 100]
    - Non-negative indicators: [0, inf)
    - Z-scored columns: flag but don't clamp values beyond +-5 SD
    """
    for col in df.columns:
        if col in PERCENTAGE_INDICATORS and col in df.columns:
            result.checks_run += 1
            numeric = pd.to_numeric(df[col], errors="coerce")
            below = int((numeric < 0).sum())
            above = int((numeric > 100).sum())
            if below > 0 or above > 0:
                result.add_issue(
                    f"range_{col}",
                    "warning",
                    f"{col}: {below} values < 0, {above} values > 100 (clamped to [0, 100])",
                    detail={"below_zero": below, "above_100": above},
                )
                df[col] = numeric.clip(lower=0, upper=100)
            else:
                result.checks_passed += 1

        elif col in NON_NEGATIVE_INDICATORS and col in df.columns:
            result.checks_run += 1
            numeric = pd.to_numeric(df[col], errors="coerce")
            below = int((numeric < 0).sum())
            if below > 0:
                result.add_issue(
                    f"range_{col}",
                    "warning",
                    f"{col}: {below} negative values (clamped to 0)",
                    detail={"negative_count": below},
                )
                df[col] = numeric.clip(lower=0)
            else:
                result.checks_passed += 1

        elif col.startswith("Z_"):
            result.checks_run += 1
            numeric = pd.to_numeric(df[col], errors="coerce")
            extreme = int(((numeric.abs() > 5) & numeric.notna()).sum())
            if extreme > 0:
                result.add_issue(
                    f"zscore_extreme_{col}",
                    "info",
                    f"{col}: {extreme} values beyond +-5 SD (flagged, not modified)",
                    detail={"extreme_count": extreme},
                )
            else:
                result.checks_passed += 1

    return df, result


def verify_outliers_iqr(
    df: pd.DataFrame,
    numeric_cols: List[str],
    result: VerificationResult,
    multiplier: float = 3.0,
) -> VerificationResult:
    """Flag statistical outliers using the IQR method (Tukey, 1977).

    Does NOT remove outliers (that is a researcher decision), only flags them.
    Uses 3x IQR (far outliers) as the default threshold.
    """
    for col in numeric_cols:
        if col not in df.columns:
            continue
        result.checks_run += 1
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(s) < 10:
            result.checks_passed += 1
            continue
        q1 = float(s.quantile(0.25))
        q3 = float(s.quantile(0.75))
        iqr = q3 - q1
        if iqr == 0:
            result.checks_passed += 1
            continue
        lower = q1 - multiplier * iqr
        upper = q3 + multiplier * iqr
        outliers = int(((s < lower) | (s > upper)).sum())
        if outliers > 0:
            result.add_issue(
                f"outlier_{col}",
                "info",
                f"{col}: {outliers} far outliers (>{multiplier}x IQR, flagged only)",
                detail={
                    "column": col,
                    "n_outliers": outliers,
                    "iqr": round(iqr, 4),
                    "lower_fence": round(lower, 4),
                    "upper_fence": round(upper, 4),
                },
            )
        else:
            result.checks_passed += 1
    return result


def verify_year_coverage(
    df: pd.DataFrame,
    year_col: str,
    expected_min: int,
    expected_max: int,
    result: VerificationResult,
) -> VerificationResult:
    """Check that observed year range is within expected bounds."""
    result.checks_run += 1
    if year_col not in df.columns:
        result.add_issue("year_coverage", "error", f"Missing year column: {year_col}")
        return result
    years = pd.to_numeric(df[year_col], errors="coerce").dropna()
    if years.empty:
        result.add_issue("year_coverage", "error", "No valid year values found")
        return result
    obs_min = int(years.min())
    obs_max = int(years.max())
    if obs_min < expected_min or obs_max > expected_max:
        result.add_issue(
            "year_coverage",
            "warning",
            f"Year range [{obs_min}, {obs_max}] exceeds expected [{expected_min}, {expected_max}]",
            detail={"observed_min": obs_min, "observed_max": obs_max},
        )
    else:
        result.checks_passed += 1
    return result


def verify_panel_balance(
    df: pd.DataFrame,
    entity_col: str,
    time_col: str,
    result: VerificationResult,
    min_balance_ratio: float = 0.3,
) -> VerificationResult:
    """Check panel balance: ratio of observed to potential cells.

    A strongly unbalanced panel (below min_balance_ratio) raises a warning,
    per Wooldridge (2010) Ch. 10 on unbalanced panel diagnostics.
    """
    result.checks_run += 1
    if entity_col not in df.columns or time_col not in df.columns:
        result.checks_passed += 1
        return result
    n_entities = df[entity_col].nunique()
    n_periods = df[time_col].nunique()
    potential = n_entities * n_periods
    actual = len(df.drop_duplicates(subset=[entity_col, time_col]))
    balance = actual / max(potential, 1)
    if balance < min_balance_ratio:
        result.add_issue(
            "panel_balance",
            "warning",
            f"Panel balance ratio {balance:.2f} below threshold {min_balance_ratio}",
            detail={
                "n_entities": n_entities,
                "n_periods": n_periods,
                "potential_cells": potential,
                "observed_cells": actual,
                "balance_ratio": round(balance, 4),
            },
        )
    else:
        result.checks_passed += 1
    return result


def verify_numeric_coercion(
    df: pd.DataFrame, columns: List[str], result: VerificationResult
) -> Tuple[pd.DataFrame, VerificationResult]:
    """Coerce columns to numeric, logging how many values were lost.

    This replaces the silent pd.to_numeric(errors='coerce') pattern with
    an auditable version that records exactly how many values were coerced to NaN.
    """
    for col in columns:
        if col not in df.columns:
            continue
        result.checks_run += 1
        original_non_null = int(df[col].notna().sum())
        df[col] = pd.to_numeric(df[col], errors="coerce")
        new_non_null = int(df[col].notna().sum())
        coerced = original_non_null - new_non_null
        if coerced > 0:
            result.add_issue(
                f"numeric_coercion_{col}",
                "warning",
                f"{col}: {coerced} non-null values coerced to NaN during numeric conversion",
                detail={"column": col, "coerced_count": coerced},
            )
            result.add_dropped(f"numeric_coercion_{col}", coerced)
        else:
            result.checks_passed += 1
    return df, result


def verify_minimum_sample_size(
    df: pd.DataFrame,
    min_rows: int,
    result: VerificationResult,
    context: str = "",
) -> VerificationResult:
    """Ensure minimum sample size for statistical validity.

    Per standard econometric practice, small samples (< 30) limit
    the reliability of z-score normalization and correlation estimates.
    """
    result.checks_run += 1
    if len(df) < min_rows:
        result.add_issue(
            "minimum_sample_size",
            "error",
            f"Sample size {len(df)} below minimum {min_rows}{' (' + context + ')' if context else ''}",
            detail={"n_rows": len(df), "min_required": min_rows, "context": context},
        )
    else:
        result.checks_passed += 1
    return result


def verify_score_completeness(
    df: pd.DataFrame,
    score_col: str,
    entity_col: str,
    result: VerificationResult,
    max_missing_pct: float = 25.0,
) -> VerificationResult:
    """Verify that the composite score is computed for a sufficient share of entities."""
    result.checks_run += 1
    if score_col not in df.columns:
        result.add_issue("score_completeness", "error", f"Missing score column: {score_col}")
        return result
    total = len(df)
    missing = int(df[score_col].isna().sum())
    pct_missing = 100.0 * missing / max(total, 1)
    if pct_missing > max_missing_pct:
        result.add_issue(
            "score_completeness",
            "warning",
            f"{score_col}: {pct_missing:.1f}% missing (threshold: {max_missing_pct}%)",
            detail={
                "total": total,
                "missing": missing,
                "pct_missing": round(pct_missing, 2),
            },
        )
    else:
        result.checks_passed += 1
    return result


def verify_domain_coverage(
    df: pd.DataFrame,
    domain_cols: List[str],
    result: VerificationResult,
    min_domains_available: int = 3,
) -> VerificationResult:
    """Check that at least min_domains_available domain scores are non-null per row.

    Per OECD (2008) composite indicator guidelines, a composite index
    should not be computed from too few constituent domains.
    """
    result.checks_run += 1
    available = [col for col in domain_cols if col in df.columns]
    if not available:
        result.add_issue(
            "domain_coverage",
            "error",
            "No domain score columns found",
        )
        return result
    domains_per_row = df[available].notna().sum(axis=1)
    insufficient = int((domains_per_row < min_domains_available).sum())
    if insufficient > 0:
        result.add_issue(
            "domain_coverage",
            "warning",
            f"{insufficient} rows have fewer than {min_domains_available} domains available",
            detail={
                "insufficient_rows": insufficient,
                "total_rows": len(df),
                "min_required": min_domains_available,
            },
        )
    else:
        result.checks_passed += 1
    return result
