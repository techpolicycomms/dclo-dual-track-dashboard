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

## Panel-by-panel guide

- **KPI cards**: mean score, coverage, top and bottom entity for selected year.
- **Ranking chart**: compares states or countries (depending on selected track).
- **Map**:
  - state map for India track
  - country choropleth for comparative track
- **Trend chart**: trajectories for selected states/countries over time.
- **Domain heatmap**: reveals strengths/weaknesses by domain.
- **Domain profile**: domain detail for one selected entity.

## Method summary

- Indicators are standardized using z-scores.
- Domain scores are means of available standardized indicators in each domain.
- Overall DCLO is the mean of available domain scores.
- Missing values are handled by using available indicators/domains rather than forcing imputation in this dashboard layer.
- Country-year track uses a quality-gated indicator intake (`core_formative/context/exclude`) before scoring.

## Important caveats

- This is a **comparative composite**, not a causal impact estimate.
- Coverage and comparability differ across sources and years.
- Some indicators are proxy measures due to administrative data availability.
- SHG integration includes state-level fallback for sparse year matches; interpret year-specific changes with caution.
- Country-year comparisons depend on cross-country indicator harmonization quality and coverage.

## Recommended use

- Use for benchmarking and diagnostic targeting.
- Pair with domain view before making policy conclusions.
- Cross-check important findings with raw indicator tables and metadata notes.
