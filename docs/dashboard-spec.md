# Power BI Dashboard Spec (India Open Data)

## KPI-to-Measure Mapping

- Dataset freshness -> `Latest Refresh Timestamp`
- Coverage -> `Records Loaded`
- State comparison -> `Records by State`
- Category trend -> `Records by Category Over Time`

## Semantic Model Design

- Fact table(s): one per gold output (`*_gold.csv`) with `record_count`
- Dimensions:
  - `dim_state` (state names/codes)
  - `dim_time` (date/month/year)
  - `dim_category` (sector/category where available)
- Relationship preference: one-directional from dimensions to fact

## Page Layout Plan

1. Executive Summary
   - KPI cards: records, freshness
   - National trend line
2. State Insights
   - Map/bar ranking by state
   - Drill-through to dataset detail
3. Category Trends
   - Sector/category split over time
   - Top movers

## DAX Design Notes

- Build base measures first:
  - `[Records Loaded] = SUM(Fact[record_count])`
- Add presentation measures:
  - `[Records Loaded (M)]`
  - `% contribution by state/category`
- Use a canonical calendar table for time intelligence.

## Performance and Governance Plan

- Start with Import mode
- Enable incremental refresh for larger facts
- Keep model slim (drop unused columns)
- Set RLS if audience requires state/domain segregation
- Publish through dev -> test -> prod workspaces
