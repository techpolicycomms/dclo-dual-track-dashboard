# DCLO Standard Checks Summary

- overall_passed: `True`

## State Track
- passed: `True`
- rows: `94`
- states: `35`
- years: `2014` to `2019`
- issues:

## Country Track
- passed: `True`
- rows: `577`
- economies: `49`
- years: `2014` to `2025`
- panel_balance: `0.9813`
- issues:

## Causal Track
- passed: `True`
- coefficient_rows: `9`
- n_specs: `5`
- issues:

## Audit Manifests
- passed: `True`
- manifests_found: `3/3`
- issues:

## Verification Reports
- passed: `True`
- issues:

## Weekly Secondary Loops
- passed: `False`
- loops_available: `3/6`
  - acc: passed=`False` rows=`0`
    - missing_gold_output:dclo_acc_connectivity_weekly.csv
  - skl: passed=`True` rows=`6`
  - srv: passed=`True` rows=`7`
  - agr: passed=`True` rows=`4`
  - eco: passed=`False` rows=`0`
    - missing_gold_output:dclo_eco_upi_monthly.csv
  - out: passed=`False` rows=`0`
    - missing_gold_output:dclo_out_air_quality_weekly.csv