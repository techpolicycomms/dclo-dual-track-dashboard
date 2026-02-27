# DCLO Identification Strategy

## Identification Goal

Estimate whether prior-period digital capability is associated with subsequent service-related performance while reducing confounding from:

- time-invariant entity differences,
- global/common shocks by year.

## Core Strategy

The baseline causal layer uses:

- entity fixed effects,
- year fixed effects,
- lagged predictors (`t-1`),
- entity-clustered standard errors.

This approach targets within-entity temporal variation and removes static between-entity bias.

## Threats to Validity

1. **Time-varying confounding**  
   Policy reforms or shocks that vary across countries and correlate with both DCLO and outcomes can bias estimates.

2. **Reverse causality**  
   Better service outcomes may improve observed digital capability metrics in later periods.

3. **Measurement error**  
   Sparse domains and indicator harmonization limitations can attenuate or distort coefficients.

4. **Model dependence**  
   Effect size can vary by lag length, controls, and FE structure.

## Mitigations Implemented

- Lagged predictors to reduce contemporaneous simultaneity.
- Multiple robustness specifications in `config/dclo_causal_model.yml`.
- Placebo lead specification to detect timing inconsistencies.
- Model QA checks that flag:
  - weak panel coverage,
  - placebo patterns that are not weaker than baseline,
  - degenerate stability outputs.

## Interpretation Rule Set

- Treat estimates as **causal-evidence oriented**, not definitive causal proof.
- Prioritize effects that are:
  - sign-consistent across baseline and robustness specs,
  - statistically distinguishable from zero with reasonable uncertainty bounds,
  - not contradicted by placebo diagnostics.

## Reporting Standard

Every release should publish:

- baseline + robustness coefficient table,
- fit metrics (`n_obs`, entities, years, within-R2),
- placebo outcomes,
- rank stability diagnostics,
- explicit caveats from standard checks summary.
