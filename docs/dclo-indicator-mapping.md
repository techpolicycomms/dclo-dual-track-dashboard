# DCLO Indicator Mapping (India Open Data Starter)

This dictionary operationalizes **Digital Capability for Life Outcomes (DCLO)** as a higher-order formative construct.  
Use this as the working mapping before finalizing exact `resource_id` selections.

## Unit of Analysis

- Preferred: individual-level (if microdata available)
- Practical default with open administrative data: state-year (or district-year)

## Normalization and Sign Rules

- Normalize continuous indicators to z-scores (default) or min-max for dashboard display.
- Ensure all indicators are coded so **higher = better capability/outcome**.
- For adverse indicators (for example dropout, fraud, mortality), reverse-code before aggregation.

## Domain-Indicator Dictionary (Starter)


| Domain                 | Indicator code | Candidate indicator (proxy)                                     | Expected sign | Suggested transform | Level      |
| ---------------------- | -------------- | --------------------------------------------------------------- | ------------- | ------------------- | ---------- |
| Access & Connectivity  | ACC_1          | Internet subscribers per 100 population                         | +             | log or z-score      | state-year |
| Access & Connectivity  | ACC_2          | Mobile broadband penetration                                    | +             | z-score             | state-year |
| Access & Connectivity  | ACC_3          | Rural connectivity coverage share                               | +             | proportion/z-score  | state-year |
| Skills & Literacy      | SKL_1          | Digital literacy rate (or proxy via ICT training participation) | +             | proportion/z-score  | state-year |
| Skills & Literacy      | SKL_2          | Secondary completion with ICT exposure proxy                    | +             | z-score             | state-year |
| Skills & Literacy      | SKL_3          | Female digital skills participation ratio                       | +             | ratio/z-score       | state-year |
| Economic Participation | ECO_1          | Digital payments adoption rate                                  | +             | log(1+x) or z-score | state-year |
| Economic Participation | ECO_2          | Share of MSME digital registrations/transactions                | +             | proportion/z-score  | state-year |
| Economic Participation | ECO_3          | Platform-enabled employment proxy                               | +             | z-score             | state-year |
| Service Enablement     | SRV_1          | e-Health service utilization rate                               | +             | proportion/z-score  | state-year |
| Service Enablement     | SRV_2          | e-Learning participation/enrollment proxy                       | +             | proportion/z-score  | state-year |
| Service Enablement     | SRV_3          | Digital financial inclusion usage rate                          | +             | proportion/z-score  | state-year |
| Agency, Safety, Rights | AGR_1          | Women-controlled digital account usage                          | +             | proportion/z-score  | state-year |
| Agency, Safety, Rights | AGR_2          | Cyber safety awareness/complaint resolution proxy               | +             | z-score             | state-year |
| Agency, Safety, Rights | AGR_3          | Trusted digital identity usage success rate                     | +             | proportion/z-score  | state-year |
| Outcome Realization    | OUT_1          | Income resilience proxy linked to digital adoption              | +             | z-score             | state-year |
| Outcome Realization    | OUT_2          | Service access improvement differential                         | +             | difference/z-score  | state-year |
| Outcome Realization    | OUT_3          | Education/health continuity proxy with digital channel use      | +             | z-score             | state-year |


## Measurement Design Decisions

1. **Formative by construction**: indicators define domain scores; domains define DCLO.
2. **Domain balance**: start with 2-4 indicators per domain; avoid overloading one domain.
3. **Collinearity guardrail**: remove/merge indicators with high redundancy after VIF checks.
4. **Policy relevance filter**: keep indicators interpretable to non-technical stakeholders.

## Missing Data Rules

Apply in order:

1. Prefer official revised values over provisional values.
2. If one year missing between two valid years: linear interpolation.
3. If leading/trailing missing for short spans: carry nearest value with flag.
4. If missingness > 30% for an indicator in analysis window: drop indicator or replace with validated proxy.
5. Track all imputations with binary flags for robustness tests.

## Outlier and Quality Rules

- Winsorize extreme tails (for example 1st/99th percentile) for unstable rate indicators.
- Keep an untreated copy for sensitivity checks.
- Validate monotonic direction after every transform.

## Weighting and Aggregation (Initial)

- Within-domain: equal weights as baseline.
- Cross-domain to DCLO: equal domain weights as baseline.
- Sensitivity scenarios:
  - expert-prior domain weights
  - data-driven weights from bootstrap-stable formative estimates

## SEM/PLS-SEM Readiness Checklist

- Domain indicators finalized with theoretical justification
- Indicator signs harmonized
- Missing-data and outlier handling logged
- VIF and weight significance evaluated
- Nomological links to external outcomes specified
- Sensitivity scenarios pre-registered

## Data Engineering Link to Pipeline

Map each selected indicator to:

- `resource_id`
- extraction field names
- transformation logic
- final analytical column name

Store this mapping in `config/sources.yml` and transformation metadata so Power BI model refresh stays reproducible.

## Next File to Build

Create `docs/dclo-data-catalog.md` with one row per chosen dataset:

- source agency
- resource_id
- indicator code(s)
- update frequency
- known quality caveats

## Dual-Track Note (Country Comparative)

For the country-year comparative track:

- Intake and role assignment is generated in `docs/dpi-indicator-intake.md`.
- Indicators are classified into:
  - `core_formative`
  - `context_only`
  - `exclude`
- Balanced inclusion target is 4-8 indicators per domain where feasible.
- If a domain has limited candidates, fallback selection is allowed with validation flags.

