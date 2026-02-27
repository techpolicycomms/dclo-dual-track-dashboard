# DCLO Data Catalog (Local Downloads - Phase 1)

This catalog tracks local datasets available for first-pass DCLO scoring.

## Included Sources

| source_id | file_name | domain(s) | geography_level | time_level | key fields (expected) | status | notes |
|---|---|---|---|---|---|---|---|
| upi_transactions | upi-transactions-p2p-and-p2m.csv | ECO (context) | national | month/year | month, total_vol, total_val, p2p_vol, p2m_vol | ready | national covariate; no state split |
| internet_banking | internet-banking-statistics.csv | ECO (context) | bank/national | month/year | month, bank_name, no_of_transactions, amt_of_transactions, active_users | ready | aggregated to national yearly context |
| nafis | nafis.csv | ECO, OUT | state | year | year, state_name, prop_saving, prop_hh_microfin, hh_income_monthly | ready | strong state-level economic inclusion signal |
| nfhs | national-family-health-survey.csv | ACC, SKL, SRV, AGR | district/state | survey year | year, state_name, district_name, pop_hh_elec, fem_literacy, pop_hh_sf, fem_15_24_hyg_period | ready | district-level, aggregated to state-year |
| shg_profile | shg-financial-and-member-profile-details.csv | AGR, ECO | state/district (detected from file) | year (parsed where available) | auto-detected via chunking | ready (chunked) | integrated with memory-safe chunk pipeline |

## Unit of Analysis (Phase 1)

- **state-year**
- District-level files are aggregated to state-year means.
- National-only files are merged as yearly context variables.

## Indicator Mapping in Use (Phase 1)

- ECO: `prop_hh_microfin`
- OUT: `hh_income_monthly`, `prop_saving`
- ACC: `pop_hh_elec`
- SKL: `fem_literacy`
- SRV: `pop_hh_sf`
- AGR: `fem_15_24_hyg_period`

## Exclusions (Current)

- ATM-level data is not yet included (not downloaded).
- SHG large file is not required for first-pass computation; add after schema profiling.

## Next Catalog Upgrade

After SHG profiling and ATM inclusion:

- expand AGR and ECO coverage
- add geographic harmonization notes
- add source quality flags and missingness rates per indicator

## Dual-Track Extension (DPI Comparative)

Additional comparative-country inputs now integrated:

| source_id | file_name | domain(s) | geography_level | time_level | key fields | status |
|---|---|---|---|---|---|---|
| dpi_long | dpi_ready_long_with_imputation.csv | ACC, SKL, SRV, AGR, ECO, OUT | country | year | economy, year, indicator_code, norm_score, was_imputed, pillar | ready |
| dpi_mapping | indicator_mapping.csv | all (metadata) | indicator-level | n/a | indicator_code, indicator_name, pillar, direction | ready |
| dpi_coverage | coverage_by_indicator.csv | quality gating | indicator-level | n/a | indicator_code, pct_observed | ready |

Generated comparative outputs:

- `data/gold/dclo_country_year.csv`
- `data/gold/dpi_indicator_intake.csv`
- `data/gold/dpi_selected_indicators_by_domain.json`
- `docs/dpi-indicator-intake.md`
