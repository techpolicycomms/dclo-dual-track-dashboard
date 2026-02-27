# DCLO Dataset Intake Template (for IDP browsing)

Use this sheet while browsing India Data Portal collections.  
Fill one row per candidate dataset and keep only rows that pass the minimum filters.

## Minimum filters (must be YES)

- Has geography field (prefer LGD code)
- Has time field (year/month)
- Has numeric value field
- Has reasonable coverage window
- Has clear source/metadata

## Copy-Paste Table (Markdown)

| keep? | dclo_domain | indicator_code | dataset_title | idp_collection | dataset_url | api_or_download_url | source_agency | geography_level | lgd_available | geo_field_name | time_field_name | value_field_name | unit | coverage_start | coverage_end | update_frequency | missingness_estimate | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| YES/NO | ACC/SKL/ECO/SRV/AGR/OUT | e.g., ACC_1 |  |  |  |  |  | state/district/block/village | YES/NO |  |  |  |  |  |  | annual/monthly/etc | low/med/high |  |

## CSV Version (easy to import)

```csv
keep,dclo_domain,indicator_code,dataset_title,idp_collection,dataset_url,api_or_download_url,source_agency,geography_level,lgd_available,geo_field_name,time_field_name,value_field_name,unit,coverage_start,coverage_end,update_frequency,missingness_estimate,notes
YES,ACC,ACC_1,,,,,,,,,,,,,,,,,
YES,SKL,SKL_1,,,,,,,,,,,,,,,,,
YES,ECO,ECO_1,,,,,,,,,,,,,,,,,
```

## Domain code guide

- `ACC` = Access and Connectivity
- `SKL` = Skills and Literacy
- `ECO` = Economic Participation
- `SRV` = Service Enablement
- `AGR` = Agency, Safety, Rights
- `OUT` = Outcome Realization

## Selection priorities (rank each candidate 1-5)

1. Granularity (district/block better than state)
2. Time depth (more years better)
3. Completeness (low missingness better)
4. Policy relevance to DCLO theory
5. Stability of definitions over time

## What to send back

When done, send me either:

1. Completed markdown table, or
2. CSV rows, or
3. Dataset URLs only (I will complete the rest)

Then I will generate:

- `docs/dclo-data-catalog.md`
- `config/sources.yml`
- field mapping and transform rules for scoring
