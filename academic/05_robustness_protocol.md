# Robustness Protocol

This document specifies the robustness battery the next release will run. It is intended to be **lockable** as a pre-analysis plan (see `11_preregistration.md`).

## 1. Specifications

For each estimand:

| Spec ID | Estimator | Lag | Controls | FE | Cluster | Notes |
|---|---|---|---|---|---|---|
| **S1** | Two-way FE OLS | 1 | X1 (GDP, urban, pop, electricity) | entity, year | entity | Baseline. |
| **S2** | Two-way FE OLS | 2 | X1 | entity, year | entity | Lag length sensitivity. |
| **S3** | Two-way FE OLS | 1 | X1 + region × year FE | entity, year, region×year | entity | Sub-global shocks. |
| **S4** | Two-way FE OLS | 1 | none | entity, year | entity | Control sensitivity. |
| **S5** | Pooled OLS | 1 | X1 | none | entity | Stripped benchmark. |
| **S6** | First-difference | 1 | ΔX1 | none | entity | FD vs FE check. |
| **S7** | Arellano-Bond GMM | 1 | X1 | dynamic | entity | Dynamic panel; lagged-DV concerns. |
| **S8** | Errors-in-variables IV | 1 | X1 | entity, year | entity | Instrument D with disjoint-source D'. |
| **S9** | Long-difference | t−5 → t+5 | X1 (level changes) | none | entity | Structural change. |
| **S10** | Synthetic-control case | India around UPI 2016 | — | — | — | One-country case. |
| **S11** | Synthetic-control case | Estonia around e-residency 2014 | — | — | — | One-country case. |
| **S12** | Synthetic-control case | Brazil around PIX 2020 | — | — | — | One-country case. |
| **S13** | Event-study (DiD) | dynamic ±4y | X1 | entity, event-time | entity | DPI events as treatment. |
| **S14** | Spatial-lag | 1 | X1 + WY | entity, year | entity | SUTVA test. |
| **S15** | Population-weighted | 1 | X1 | entity, year | entity | Selection-on-coverage check. |

## 2. Placebos

| Placebo ID | Description | Expected result |
|---|---|---|
| **P1** | Permute D within year. | Coefficient near zero. |
| **P2** | Permute D within country. | Coefficient near zero. |
| **P3** | Placebo outcome 1: agricultural land share `WB_AG.LND.AGRI.ZS`. | Coefficient indistinguishable from zero; significantly different from S1 β. |
| **P4** | Placebo outcome 2: average annual rainfall (CRU). | Coefficient indistinguishable from zero. |
| **P5** | Leave-future-out: regress D_{t+1} on Y_t (reverse). | Coefficient should be smaller than S1; if larger, identification fails. |
| **P6** | Pre-treatment trend: estimate S1 on data restricted to t < event year for each DPI rollout. | Coefficient indistinguishable from zero (parallel-trends test). |

## 3. Stability

| Test | Description |
|---|---|
| **St1** | Dirichlet-perturbed domain weights (current build does this; keep). |
| **St2** | Leave-one-domain-out aggregation. |
| **St3** | Leave-one-country-out re-estimation; report β distribution. |
| **St4** | Leave-one-year-out re-estimation. |
| **St5** | Bootstrap on entity (cluster bootstrap), 1000 reps. |
| **St6** | Manski bounds with worst-case unobserved-indicator value. |
| **St7** | Cinelli–Hazlett sensitivity: minimum unobserved-confounder strength to overturn S1. |

## 4. Reporting

For each estimand the paper reports:

1. A coefficient table with S1–S15 columns and {β, SE, CI95, n_obs, n_entities, n_years, R²_within} rows.
2. A placebo table with P1–P6.
3. A stability panel with St1–St7.
4. A specification curve in the spirit of Simonsohn et al. (2020): all admissible model permutations as a single distribution.
5. The Cinelli–Hazlett sensitivity threshold (RV) for the headline coefficient.

## 5. Decision rules

A finding is reported as **causal-evidence-supported** only if:

1. β has the same sign across S1, S2, S3, S6, S8, and S15; **and**
2. P1, P2, P3, P5, P6 all return coefficients smaller in magnitude than S1; **and**
3. Cinelli–Hazlett RV ≥ 0.5 of any one observed control's strength; **and**
4. The leave-one-out distribution from St3 contains zero only in fewer than 5 % of draws.

If any of (1)–(4) fail, the finding is reported as a *conditional association*, not a causal effect, with the failure foregrounded.

## 6. Implementation status

- S1, S2, S4, S5 implemented in current build (`src/transforms/build_dclo_causal_panel.py`).
- P1 implemented as `placebo_permuted_l1`.
- St1 implemented as Dirichlet weight perturbation.
- **All other specifications not yet implemented.** This is the priority work for the next sprint.

## 7. References

- Simonsohn, U., Simmons, J. P., & Nelson, L. D. (2020). Specification curve analysis. *Nature Human Behaviour*, 4(11), 1208–1214.
- Cinelli, C., & Hazlett, C. (2020). Making sense of sensitivity. *JRSS B*, 82(1), 39–67.
- Manski, C. F. (1995). *Identification Problems in the Social Sciences*. Harvard UP.
- Roodman, D. (2009). How to do xtabond2: An introduction to difference and system GMM in Stata. *Stata Journal*, 9(1), 86–136.
- Abadie, A., & L'Hour, J. (2021). A penalized synthetic control estimator. *JASA*, 116(536), 1817–1834.
