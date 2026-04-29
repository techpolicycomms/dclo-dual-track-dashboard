# Pre-Analysis Plan (Pre-Registration)

**Status.** DRAFT — to be locked, signed by author and supervisor, and timestamped to OSF or AsPredicted before the next data refresh.

## 1. Study design

**Object.** A capability-grounded reading of cross-national digital-development indicators, instantiated through a constructed comparator index (DCLO).

**Population.** All countries with non-missing capability-targeted indicators in the included sources (ITU IDI, Findex, GovTech Maturity Index, EGDI, GSMA, Freedom House FOTN, ILOSTAT digital-occupations) for 2014–2024.

**Outcome variable.** A capability-targeted SRV (Service Enablement) score built only from functioning-layer indicators: Findex digital-payment-use; EGDI service-delivery sub-score; WHO digital-health-uptake.

**Predictor.** A capability-targeted DCLO score built only from resource + conversion-factor + capability-layer indicators (ITU connectivity; ITU/UNESCO SDG 4.4.1 ICT skills; Freedom House FOTN; GSMA Mobile Gender Gap; Findex female-controlled account; SDG 16.10.2 access-to-information).

The disjoint-partition condition (predictor and outcome built from disjoint indicator and source pools) is the binding pre-registration commitment.

## 2. Hypotheses

**H1.** Conditional on entity and year fixed effects, the capability-targeted DCLO at *t-1* is positively associated with the capability-targeted SRV at *t*. We pre-specify a one-sided test at α = 0.05.

**H2.** The placebo outcome — agricultural land share `WB_AG.LND.AGRI.ZS` at *t* — is **not** associated with the capability-targeted DCLO at *t-1*. We pre-specify a two-sided test at α = 0.05.

**H3.** The current-build DCLO and the capability-targeted DCLO disagree on country rank by Spearman ρ < 0.6 in at least 60 % of years 2015–2024. We pre-specify Spearman ρ as the test statistic.

## 3. Specifications

The full battery is `academic/05_robustness_protocol.md`, S1–S15 + P1–P6 + St1–St7. Decision rules are in §5 of that file.

The headline coefficient is the S1 baseline, conditional on the decision rule passing. If any of the four decision-rule conditions fail, the headline reports the coefficient as a *conditional association* and foregrounds the failure.

## 4. Sample size and power

Target panel: ≥ 60 entities × 11 years = 660 observations. Power calculation (α = 0.05, two-sided, with one focal coefficient and 5 controls, intra-cluster correlation = 0.4): minimum detectable effect ≈ 0.15 standard deviations. Pre-registered.

## 5. Inclusion / exclusion rules

**Include** any country-year with:
- Outcome SRV available for at least 2 of {Findex, EGDI, WHO}.
- Predictor DCLO available for at least 4 of 6 domains.
- Country has at least 5 panel-years of observation.

**Exclude** any country with:
- Average annual indicator imputation share > 0.35.
- Joint panel coverage < 0.40.

**Sensitivity.** Re-estimate including borderline-coverage countries; report difference.

## 6. Multiple-comparison policy

Bonferroni correction across the three primary hypotheses (H1, H2, H3). Secondary results (case studies, decompositions) are exploratory and reported with raw p-values; readers are warned not to interpret them as primary tests.

## 7. Deviations protocol

Any deviation from this plan after lock is reported in the paper with:
- The deviation in plain language.
- The reason.
- The result with and without the deviation.

## 8. Lock

To be locked at OSF or AsPredicted. SHA-256 checksum of this file at lock time recorded here:

```
[checksum to be inserted at lock time]
```

Author signature: **\_\_\_\_\_\_\_\_\_\_**  Date: **\_\_\_\_**
Supervisor signature: **\_\_\_\_\_\_\_\_\_\_**  Date: **\_\_\_\_**
