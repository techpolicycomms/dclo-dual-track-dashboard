# Revised Identification Strategy

This document supersedes `docs/dclo-identification-strategy.md` for the academic track. It states the estimand, presents a DAG, names the threats, and lists the empirical commitments for credible identification.

## 1. Estimand

Let `i` index countries (or Indian states), `t` index years, `Y_{i,t}` denote a *capability-targeted* outcome (a functioning, e.g., share of households that completed a financial transaction through a digital channel in the past year), and `D_{i,t}` denote a *capability-targeted* digital-capability composite that explicitly does **not** share constituent indicators with `Y`.

The estimand is

> **β = E [ Y_{i,t} | D_{i,t-1} = d+1, X_{i,t}, ν_i, λ_t ] − E [ Y_{i,t} | D_{i,t-1} = d, X_{i,t}, ν_i, λ_t ]**

i.e., the change in the expected functioning level associated with a one-unit change in lagged capability, conditional on:

- X_{i,t}: time-varying controls (GDP per capita, urbanisation share, demographic structure, electricity access, public-spending share);
- ν_i: country fixed effect (time-invariant heterogeneity);
- λ_t: year fixed effect (global shocks).

This is a within-entity, within-period **conditional association**. Under the unconfoundedness assumption that, conditional on (X, ν, λ), there are no remaining confounders that vary at i × t and affect both D_{t-1} and Y_t, β recovers an average treatment effect on the treated. Under any weaker assumption, β is reported as an association.

The current build's estimand is mis-specified because `Y` and `D` share constituent indicators (see §2). The proposed estimand is correct only after the indicator-validity audit (`03_indicator_validity_audit.md`) is implemented.

## 2. DAG

```
                ┌────────────┐
                │  GDP_{i,t} │  (time-varying confounder)
                └─────┬──────┘
                      │
                      v
           ┌────────────────────┐
           │  D_{i, t-1}        │ ── digital capability (capabilities + conv. factors)
           └────────┬───────────┘
                    │ (1)
                    v
           ┌────────────────────┐
           │  Y_{i, t}          │ ── functioning (achievement)
           └────────┬───────────┘
                    │
                    v
                  policy / measurement
                    artefact

   ν_i  ──► both D and Y (entity-level structure)
   λ_t  ──► both D and Y (global shocks; e.g., COVID-19, smartphone-cost decline)

   policy reform R_{i,t} ──► both D_{i,t-1} (a year prior) AND Y_{i,t}
       └── this is the time-varying confounder TWFE cannot close.
```

Backdoor paths from `D_{t-1}` to `Y_{t}` through:

- `ν_i`: closed by entity fixed effects.
- `λ_t`: closed by year fixed effects.
- `GDP_{i,t}`, urbanisation, demographics: closed by `X_{i,t}` controls.
- `R_{i,t}` (a country-year-specific reform that drives both digital build-out and service uptake): **not closed**. This is the binding identification threat.

## 3. Threats and mitigations

### 3.1 Time-varying country-specific shocks (R_{i,t})

**Examples.** UPI rollout 2016; India Stack 2017–2019; Jio data-price collapse 2016; pandemic stay-home 2020. Each affects D and Y near-simultaneously.

**Mitigations.**

1. **Event-study around DPI events.** Treat known events as quasi-experiments and estimate dynamic effects pre- and post-event for the country and a synthetic control.
2. **Long-difference specification.** Y_{i, t+5} − Y_{i, t-5} on D_{i, t} − D_{i, t-10}, capturing structural change rather than annual fluctuation.
3. **Synthetic-control case studies** for India, Estonia (e-residency 2014), Brazil (PIX 2020), Kenya (M-Pesa 2007 expansion).

### 3.2 Reverse causality

**Mechanism.** A higher achieved functioning (e.g., higher digital-payment usage) generates more service-supplier investment in connectivity, raising the resource-layer indicators that flow into D.

**Mitigations.**

1. **Lag structure.** Lag D by 1 and 2 years (current build does this).
2. **Lag controls in addition to lag predictor.** Currently controls are also lagged; consider strict exogeneity tests (regressing D on future Y and checking for non-zero coefficients).
3. **Instrumental variable.** Where defensible, use a country's *international-bandwidth-cable-landing-share* (geographic luck) as an IV for D-resources (Hjort & Poulsen 2019 use submarine cable arrival; the same instrument is admissible here).

### 3.3 Measurement error in D

**Mechanism.** D is a composite of imperfectly-observed indicators with substantial imputation. Classical measurement error attenuates β; differential measurement error (worse-measured countries getting systematically lower D) biases β arbitrarily.

**Mitigations.**

1. **Errors-in-variables IV** using a second composite of the same construct from disjoint sources (e.g., construct D-A from ITU + Findex, D-B from GSMA + WB GovTech, instrument D-A with D-B).
2. **Coverage-weighted re-estimation.** Re-estimate β using only entity-years with `model_trust_tier == "High"`; report difference.
3. **Bounds.** Compute Manski-style bounds under worst-case unobserved-indicator values.

### 3.4 Construct contamination

**Mechanism.** As shown in `03_indicator_validity_audit.md`, the current SRV outcome shares structural macroeconomic content with predictors. β is therefore mechanically positive.

**Mitigations.**

1. **Disjoint indicator partition.** Outcome built only from functioning-layer indicators (Findex, EGDI, WHO digital-health, UNESCO digital-education uptake). Predictor built only from resource + conversion + capability layers.
2. **Cross-source partition.** Predictor and outcome cannot share a source (e.g., not both from World Bank).

### 3.5 Spillovers (SUTVA violation)

**Mechanism.** A country's DPI capability may depend on neighbour-country investment (regional roaming agreements, undersea-cable consortium membership, regulatory harmonisation). Cross-country spillovers violate SUTVA.

**Mitigations.**

1. Spatial-lag controls (Y of contiguous-region average; D of contiguous-region average).
2. Region × year fixed effects (subsuming sub-global shocks).
3. Sensitivity to drop bordering pairs.

### 3.6 Selection on coverage

**Mechanism.** The 47-country panel is biased toward larger, more-data-rich economies. Inference does not generalise to under-covered states.

**Mitigations.**

1. Report the panel composition explicitly (table of included/excluded countries).
2. Re-weight by population size as a sensitivity.
3. State that the inference is local to the included panel.

## 4. Empirical commitments for the next release

The next release of the academic track will include:

1. **Disjoint partition** (§3.4): a redefined SRV outcome built only from Findex digital-payment-use, EGDI service-delivery score, and WHO digital-health-uptake; the predictor will not include WTO services trade or WGI percentiles.
2. **Event-study** for India around UPI 2016, with permutation-inferred placebos.
3. **Errors-in-variables IV** using ITU IDI as instrument for the construct-validated DCLO; report 2SLS β.
4. **Country-level robustness panel** with at least eight specifications (current build has three).
5. **Manski bounds** under a worst-case unobserved-indicator value.
6. **Population-weighted re-estimation.**
7. **Spatial-lag and region-FE robustness.**
8. **Reporting of all DAG-implied controls** as in Cinelli & Hazlett (2020).
9. **Pre-registration** of the analysis plan with all eight specifications and the placebo set, deposited at `11_preregistration.md` and timestamped.

## 5. What the current build's β really says

The current TWFE β = 0.62 (CI [0.43, 0.82]) on `DCLO_score(t-1) → SRV_score(t)`, given §3.4, should be read as: **the within-country, within-year association between a macro-structural composite measured in t−1 and a closely-related macro-structural composite measured in t, conditional on entity and year fixed effects.** It is consistent with structural overlap; it is not a capability-policy estimate. The paper will say so, and the dashboard caveat banner now also says so.

## 6. References

- Cinelli, C., & Hazlett, C. (2020). Making sense of sensitivity. *Journal of the Royal Statistical Society B*, 82(1), 39–67.
- Hjort, J., & Poulsen, J. (2019). The arrival of fast internet and employment in Africa. *American Economic Review*, 109(3), 1032–1079.
- Manski, C. F. (1995). *Identification Problems in the Social Sciences*. Harvard UP.
- Wooldridge, J. M. (2010). *Econometric Analysis of Cross Section and Panel Data*. MIT Press.
- Cameron, A. C., & Miller, D. L. (2015). A practitioner's guide to cluster-robust inference. *Journal of Human Resources*, 50(2), 317–372.
