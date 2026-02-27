# DCLO Dashboard Explainer

## What this dashboard shows

This dashboard presents the **Digital Capability for Life Outcomes (DCLO)** index in two tracks:

1. **India state-year track**
2. **Country-year comparative track**

DCLO is a composite score designed to capture how effectively people can participate in the digital economy and convert that participation into improved life outcomes.

## How to read the score

- Higher `DCLO_score` indicates stronger digital capability and outcome potential.
- Scores are **relative** to the comparison set in the dataset (z-score-based normalization).
- A score near `0` is around the sample average; positive values are above average; negative values are below average.

## Domain structure used

The dashboard aggregates indicators into six domains:

1. `ACC` - Access and Connectivity
2. `SKL` - Skills and Literacy
3. `SRV` - Service Enablement
4. `AGR` - Agency, Safety, and Rights
5. `ECO` - Economic Participation
6. `OUT` - Outcome Realization

## Data sources in this build

- UPI transactions
- Internet banking statistics
- NAFIS
- NFHS
- SHG financial and member profile data (chunk-processed large file)
- DPI long panel and indicator mapping outputs for country-year comparative mode

## Tab-by-tab guide

- **Measurement**:
  - KPI cards, ranking, map, trend, domain heatmap, and profile.
  - Use this tab for descriptive benchmarking.
- **Causal Evidence** (country track):
  - coefficient forest plot with 95% confidence intervals,
  - specification fit table,
  - significant-term view (`p < 0.05`, normal approximation).
- **Robustness** (country track):
  - method-agreement trend over time,
  - rank stability frequencies and rank uncertainty,
  - identification caveats summary.

## Method summary

- Indicators are standardized using z-scores.
- Domain scores are means of available standardized indicators in each domain.
- Overall DCLO is the mean of available domain scores.
- Missing values are handled by using available indicators/domains rather than forcing imputation in this dashboard layer.
- Country-year track uses a quality-gated indicator intake (`core_formative/context/exclude`) before scoring.
- Country-year causal layer estimates lagged panel associations with entity/year fixed effects and clustered uncertainty.

## Important caveats

- Measurement tab is a **comparative composite**, not by itself a causal impact estimate.
- Causal Evidence tab is assumption-dependent; coefficients should be interpreted with the robustness tab and QA checks together.
- Coverage and comparability differ across sources and years.
- Some indicators are proxy measures due to administrative data availability.
- SHG integration includes state-level fallback for sparse year matches; interpret year-specific changes with caution.
- Country-year comparisons depend on cross-country indicator harmonization quality and coverage.

## Recommended use

- Use Measurement for benchmarking and diagnostics.
- Use Causal Evidence for effect-size interpretation, not rank-only interpretation.
- Use Robustness before final policy claims.
- Cross-check important findings with raw indicator tables, QA summaries, and methodology notes.
