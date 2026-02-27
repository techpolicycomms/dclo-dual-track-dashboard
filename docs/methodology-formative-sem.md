# Methodology: Formative Multidimensional Construct for Digital Inclusion

## Construct Name and Definition

**Primary construct name:** Digital Capability for Life Outcomes (DCLO)

**Definition:**  
DCLO measures the degree to which an individual can meaningfully participate in the digital economy and convert digital participation into agency, service access, resilience, and positive life outcomes across major life domains.

## Why This is a Formative Construct

DCLO should be modeled as **formative** (causal indicators), not reflective, because:

1. Indicators represent distinct components (access, skills, economic participation, agency, outcomes) that are not interchangeable.
2. Removing one indicator changes the conceptual meaning of the construct.
3. High inter-item correlation is not required for validity in causal-indicator settings.

This follows foundational measurement arguments distinguishing causal/formative and effect/reflective indicators (Bollen and Lennox, 1991; Jarvis et al., 2003; Diamantopoulos and Winklhofer, 2001).

## SDG-Aligned Domain Architecture

Model DCLO as a higher-order formative composite with first-order domains:

1. Digital Access and Connectivity (SDG 9, SDG 10)
2. Digital Skills and Literacy (SDG 4)
3. Economic Participation and Opportunity (SDG 8)
4. Essential Service Enablement: health, education, finance (SDG 3, SDG 4)
5. Agency, Safety, and Rights in Digital Use (SDG 5, SDG 16)
6. Outcome Realization: welfare and resilience proxies (SDG 1, SDG 3, SDG 8)

SDG alignment is used as a policy mapping scaffold, while empirical indicator selection remains theory-first and data-feasible.

## Decision Process for Method Choice

### Decision 1: Reflective vs Formative Measurement

- **Choice:** Formative.
- **Reasoning:** DCLO is a composite capability construct where dimensions jointly define the latent concept.
- **Evidence base:** Misspecifying formative constructs as reflective can bias conclusions and invalidate interpretation (Jarvis et al., 2003; Petter et al., 2007).

### Decision 2: First-Order and Higher-Order Structure

- **Choice:** Hierarchical component model (domain-level composites -> DCLO).
- **Reasoning:** This preserves interpretability, allows domain diagnostics, and supports policy use where domain improvements matter independently.

### Decision 3: Estimation Framework (PLS-SEM vs CB-SEM)

- **Choice:** PLS-SEM as the default for estimation and scoring.
- **Reasoning:** PLS-SEM is well-suited for formative composites, prediction-oriented settings, and models with mixed indicator properties.
- **Guardrail:** Use CB-SEM for confirmatory robustness only when identification and distribution assumptions are comfortably met.
- **Evidence base:** Established methodological guidance for PLS-SEM use and reporting in applied research (Hair et al., 2012).

### Decision 4: Validation Strategy

- **Choice:** Multi-layer validation, not a single fit statistic.
- **Reasoning:** Formative quality is assessed through collinearity, weight relevance, and external validity checks, not internal consistency alone.
- **Required checks:**
  - Indicator/domain VIF
  - Bootstrap significance of formative weights
  - Redundancy analysis with a global criterion
  - Nomological validity with external outcomes

### Decision 5: Scoring and Interpretation

- **Choice:** Report both overall DCLO and domain scores.
- **Reasoning:** Policymaking needs both headline comparability and domain-specific actionability.
- **Sensitivity analysis:** Re-estimate under alternate normalization and weighting assumptions.

## Estimation and Diagnostics Protocol

1. Harmonize indicator direction and scaling.
2. Screen missingness and outliers; document treatment.
3. Estimate first-order formative domains.
4. Estimate second-order DCLO composite.
5. Bootstrap weights and paths for inference.
6. Test nomological relations with outcome variables (for example income resilience, service uptake, educational continuity).
7. Run sensitivity checks for alternate weighting and domain sets.

## Why This Method Supports High-Quality Inference

- Respects the construct's causal-indicator nature.
- Avoids reflective misspecification risks.
- Produces both explanatory and policy-usable scores.
- Keeps validity anchored in theory, diagnostics, and external outcome relationships.

## Key References (High-Quality Journals)

1. Bollen, K. A., and Lennox, R. (1991). Conventional wisdom on measurement: A structural equation perspective. *Psychological Bulletin*, 110(2), 305-314.
2. Diamantopoulos, A., and Winklhofer, H. M. (2001). Index construction with formative indicators: An alternative to scale development. *Journal of Marketing Research*, 38(2), 269-277.
3. Jarvis, C. B., MacKenzie, S. B., and Podsakoff, P. M. (2003). A critical review of construct indicators and measurement model misspecification in marketing and consumer research. *Journal of Consumer Research*, 30(2), 199-218.
4. Petter, S., Straub, D., and Rai, A. (2007). Specifying formative constructs in information systems research. *MIS Quarterly*, 31(4), 623-656.
5. Hair, J. F., Sarstedt, M., Ringle, C. M., and Mena, J. A. (2012). An assessment of the use of partial least squares structural equation modeling in marketing research. *Journal of the Academy of Marketing Science*, 40, 414-433.
6. Alkire, S., and Foster, J. (2011). Counting and multidimensional poverty measurement. *Journal of Public Economics*, 95(7-8), 476-487.
7. Pradhan, P., Costa, L., Rybski, D., Lucht, W., and Kropp, J. P. (2017). A systematic study of Sustainable Development Goal interactions. *Earth's Future*, 5(11), 1169-1179.

## Naming Recommendation

Use:

- **Formal research label:** Digital Capability for Life Outcomes (DCLO)
- **Dashboard label:** Digital Empowerment Index (DEI)

This keeps scientific precision in research documents and communication clarity in stakeholder dashboards.
