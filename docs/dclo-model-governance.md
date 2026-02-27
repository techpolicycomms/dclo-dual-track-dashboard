# DCLO Model Governance

## Purpose

This document defines governance for the dual-track DCLO model:

- India state-year track
- Country-year comparative track (DPI-backed)

## Indicator Acceptance Policy (Balanced)

Core formative indicators must satisfy these gates unless explicitly approved as exceptions:

1. `pct_observed_global >= 70`
2. `imputation_share <= 0.35`
3. `years_covered >= 5`
4. Domain-level target of `4-8` indicators where feasible

If a domain has fewer than 4 gate-passing indicators, controlled fallback can be used with explicit annotation.

## Indicator Role Definitions

- `core_formative`: contributes directly to domain score and DCLO.
- `context_only`: available for diagnostics or contextual overlays, excluded from core scoring.
- `exclude`: fails quality criteria or duplicates existing signal.

## Weighting Policy

- Baseline within-domain: equal weights across selected indicators.
- Baseline across domains: equal weights across available domain scores.
- State-year contextual adjustment (optional): additive context term using India DPI composite z-score.

## Validation Pack (Required per update)

Run:

```bash
python src/quality/validate_dclo_indicators.py --data-dir data/gold
```

Artifacts:

- `data/gold/dclo_indicator_missingness_report.csv`
- `data/gold/dclo_indicator_correlation_report.csv`
- `data/gold/dclo_indicator_vif_report.csv`

## Change Management

For every indicator-set revision:

1. Regenerate intake and selected-indicator files.
2. Recompute country and state outputs.
3. Run validation pack.
4. Update dashboard explainer if interpretation changes.
5. Record rationale in this section.

## Current Revision Notes

- Added DPI country-year track from `dpi_ready_long_with_imputation.csv`.
- Added intake gating + role classification.
- Added state-year context-adjusted DCLO option using India DPI composite.
- Added dual-track dashboard mode and comparative visuals.
