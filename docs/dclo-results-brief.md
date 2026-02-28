# DCLO Results Brief (Current Release)

## Executive Summary

The latest DCLO release upgrades the framework from a score-only dashboard to a rigor-oriented measurement plus causal-evidence system.
All quality gates now pass across state, country, and causal tracks.

## What Changed in This Release

1. Strengthened indicator-domain mapping for the Skills (`SKL`) domain to reduce under-representation.
2. Reworked placebo diagnostics to a stricter falsification setup (within-year permutation placebo).
3. Regenerated country scoring, causal outputs, rank stability outputs, and standard checks.
4. Updated dashboard views and QA outputs to align with inference-first interpretation.

## Quality Gate Status

From `data/gold/dclo_standard_checks_summary.json`:

- `state_track.passed = true`
- `country_track.passed = true`
- `causal_track.passed = true`
- `overall_passed = true`

## Main Empirical Finding

Baseline causal specification (`twfe_l1_srv`) shows a positive lagged association between DCLO and service outcomes:

- `DCLO_score_lag1` coefficient: approximately `0.624`
- 95% CI: approximately `[0.428, 0.820]`
- `p < 0.001`

Trust-tier control is also positive and statistically significant in baseline:

- `model_trust_tier_numeric_lag1` coefficient: approximately `0.323`
- `p ≈ 0.031`

## Falsification and Robustness

The placebo specification (`placebo_permuted_l1`) does not show a significant effect:

- `DCLO_score_lag1` placebo coefficient: approximately `-0.014`
- `p ≈ 0.119`

This is expected for a valid falsification test and supports inference credibility.

## How to Communicate Results

- Use the **Measurement** tab for descriptive benchmarking (scores, maps, profiles).
- Use the **Causal Evidence** tab for effect sizes and uncertainty intervals.
- Use the **Robustness** tab to confirm stability before policy conclusions.

## Suggested Public Narrative

“DCLO now combines multidimensional capability measurement with a panel-based causal-evidence layer. The current release passes all standard checks, shows robust positive lagged associations with service outcomes, and includes falsification diagnostics that reduce risk of spurious inference.”
