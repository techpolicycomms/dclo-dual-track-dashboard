# DCLO Causal Methodology

## Purpose

This note documents the causal-evidence layer used alongside the DCLO measurement system.
The measurement layer and causal layer are deliberately separated:

- **Measurement layer**: computes comparative DCLO scores from domain composites.
- **Causal layer**: estimates whether prior-period digital capability signals predict subsequent service-outcome performance in panel data.

## Data Inputs

- `data/gold/dclo_country_year.csv`
- `config/dclo_causal_model.yml`

Primary panel keys:

- entity: `economy`
- time: `year`

## Baseline Specification

The baseline model is configured as a two-way fixed-effects panel with clustered errors:

- outcome: `SRV_score`
- predictor: lagged `DCLO_score` at `t-1`
- control: lagged `model_trust_tier_numeric` at `t-1`
- fixed effects: entity and year
- standard errors: cluster-robust by entity

## Estimation Steps

1. Filter entities with at least the minimum required time observations.
2. Construct lagged (and for placebo, lead) regressors by entity.
3. Apply within transformation (entity + year demeaning) for FE models.
4. Estimate coefficients using OLS on transformed variables.
5. Compute cluster-robust covariance matrix at entity level.
6. Report coefficient estimates, standard errors, 95% confidence intervals, and normal-approximation p-values.

## Robustness Battery

Configured robustness variants include:

- `t-2` lag version of baseline
- FE model without the trust-tier control
- pooled (no FE) benchmark
- placebo lead specification

Outputs:

- `data/gold/dclo_causal_coefficients.csv`
- `data/gold/dclo_causal_model_fit.csv`
- `data/gold/dclo_rank_stability.csv`
- `data/gold/dclo_method_comparison.csv`

## Uncertainty and Stability

The robustness layer reports:

- confidence intervals for each coefficient,
- model-fit variation across specifications,
- rank stability under domain-weight perturbations (`dclo_rank_stability.csv`),
- agreement/divergence between baseline, confidence-weighted, and causal-signal rankings (`dclo_method_comparison.csv`).

## Interpretation Guidance

- Coefficients are conditional associations under model assumptions.
- Results are **not** automatically causal in the strong policy-evaluation sense.
- Strong interpretation requires passing the standard checks and reviewing placebo behavior.

## Reproducibility

Run:

```bash
python src/transforms/build_dclo_causal_panel.py --config config/dclo_causal_model.yml
python src/quality/run_standard_checks.py --data-dir data/gold
```
