# Gap Analysis — DCLO Dashboard as a Publishable Artefact

This document is a structured diagnostic. Each gap is named, classified by severity, evidenced from the current build, and matched to a fix locus.

Severity scale:

- **P0 — blocks any peer-reviewed publication.** The current build's claim is invalid as stated; reviewers would reject.
- **P1 — substantially weakens publication.** The current build's claim is qualified, not invalid.
- **P2 — would be raised in review and should be addressed.**
- **P3 — known limitations to declare honestly.**

## Section A — Theoretical / Construct-level gaps

### A1. The capability framing is named but not operationalised. **(P0)**

**Evidence.**
- `docs/methodology-formative-sem.md` defines DCLO as "the degree to which an individual can meaningfully participate in the digital economy and convert digital participation into agency, service access, resilience, and positive life outcomes." Yet the cited literature is measurement theory (Bollen, Diamantopoulos, Jarvis, Hair, Petter), Alkire-Foster, and Pradhan et al. on SDG interactions. **Sen, Nussbaum, Robeyns are absent.**
- "Capability" in Sen's sense requires distinguishing **resources / conversion factors / capabilities / functionings**. The build conflates them: ACC ("Access and Connectivity") is a resource indicator; SKL is a resource indicator; SRV is institutional service-delivery infrastructure; OUT in the country track is macroeconomic stocks. None of these are capabilities (substantive freedoms) and none are functionings (achieved doings/beings).

**Why it matters.** A paper named *Digital Capability for Life Outcomes* must defend the capability framing against (a) the resourcist reading (HDI-style aggregate of access measures) and (b) the achievementist reading (functionings indices). It currently does neither.

**Fix locus.** `02_theoretical_framework.md`; new sections in `09_paper_draft.md`; rename or re-justify domain labels.

### A2. "Outcome Realisation" (OUT) is operationally a macroeconomic-stock domain, not an outcome domain. **(P0)**

**Evidence.** From `data/gold/dpi_selected_indicators_by_domain.json`:

```json
"OUT": [
  "WB_SP.POP.GROW",
  "WB_AG.LND.AGRI.ZS",
  "WB_EG.CFT.ACCS.ZS"
]
```

Population growth, agricultural land share, and access to clean cooking fuels. Two of these (POP.GROW, AG.LND.AGRI.ZS) are **structural macroeconomic indicators with no plausible causal pathway from digital capability**.

**Why it matters.** The dependent variable in the causal panel is constructed largely from outcomes that should not respond to digital capability. The "positive lagged association" with SRV is mechanically an association between aggregate-economy stocks and aggregate-economy stocks.

**Fix locus.** `03_indicator_validity_audit.md`; replace OUT indicators (Findex active-account use, ITU mobile-internet penetration of bottom 40%, ILOSTAT digitally-mediated employment share, SDG 4.4.1 ICT skills indicator).

### A3. "Agency, Safety, Rights" (AGR) is operationalised by Worldwide Governance Indicators. **(P0)**

**Evidence.** From `dpi_selected_indicators_by_domain.json`:

```json
"AGR": ["WB_CC.PER.RNK", "WB_PV.PER.RNK"]
```

Control of Corruption percentile and Political Stability percentile.

**Why it matters.** WGI percentiles are **expert-coded perception composites** (Kaufmann et al.), contested in the political-science literature (Apaza 2009; Thomas 2010; Langbein & Knack 2010). They measure macro-political institutional quality, not digital agency. Reducing **digital agency** (consent, voice, complaint redress, gendered safe use, identity self-determination) to a percentile rank produced by a different consortium for a different purpose is exactly the indicator-laundering Merry (2016) and Bhuta et al. (2018) flag.

**Fix locus.** `03_indicator_validity_audit.md`; replace with: GSMA Mobile Gender Gap (digital agency proxy); Findex female-controlled account; SDG 16.10.2 access-to-information laws status; ITU consumer-protection sub-index; APC Internet Rights Charter signatories at national level.

### A4. "Skills and Literacy" (SKL) is generic education enrolment, not digital skills. **(P0)**

**Evidence.**

```json
"SKL": ["WB_SE.TER.ENRR", "WB_SE.SEC.ENRR"]
```

Tertiary and secondary gross enrolment ratios. The original `docs/dclo-indicator-mapping.md` specifies SKL_1 should be "Digital literacy rate (or proxy via ICT training participation)" — but the country-track build does not implement this.

**Why it matters.** SDG 4.4.1 is the canonical ICT-skills indicator (proportion of youth/adults with at least basic ICT skills). It exists. ITU has comparable cross-country data. The build does not use it.

**Fix locus.** Replace SKL indicators with SDG 4.4.1 (ITU); add ITU/UNESCO ICT proficiency where available. State-track: NSS-PLFS ICT module; ASER digital-skills supplement when published.

### A5. "Service Enablement" (SRV) is operationalised by WTO services-export proxies. **(P0)**

**Evidence.**

```json
"SRV": [
  "WTO_SERVICES_POSTAL_COURIER_EXPORTS",
  "WTO_SERVICES_TOTAL_EXPORTS",
  "WTO_SERVICES_TRANSPORT_EXPORTS",
  "WTO_SERVICES_TRAVEL_EXPORTS"
]
```

These are macroeconomic services-trade exports.

**Why it matters.** "Service enablement" should measure whether households/individuals can access health, education, finance, and government through digital channels. WTO services-trade exports measure the country's macro service-economy size and external orientation. Construct error.

**Fix locus.** Replace with: GovTech Maturity Index (World Bank, country-year); UN E-Government Development Index (EGDI); SDG 16.6.2 satisfaction-with-public-services where available; Findex digital-payment use; WHO digital-health country profile.

### A6. The composite is means-based; no distributional / inequality measure. **(P1)**

**Evidence.** DCLO is the mean of available domain z-scores, optionally confidence-weighted. No within-country distributional component.

**Why it matters.** A capability index that ignores inequality fails the Sen/Nussbaum criterion that capabilities are individual entitlements (Robeyns 2017, Ch. 2). Mansell & Wehn (1998), Helsper (2021), and Hilbert (2011) show digital divides operate at multiple levels (access, use, outcomes, attitudes) and across gender, urban-rural, class. None of these decompositions exist in the current build.

**Fix locus.** Add IHDI-style penalty with Atkinson inequality; add gender / urban-rural decomposition tab in dashboard for India track; add Hilbert four-level digital-divide diagnostic.

### A7. Power and political economy are absent from the construct. **(P1)**

**Evidence.** No discussion of:
- platform power asymmetry (gig labour platforms, data extraction);
- surveillance capacity in DPI architecture (Aadhaar critique; UPI transaction visibility);
- data sovereignty and cross-border flows;
- algorithmic discrimination in service delivery.

**Why it matters.** A JSGP paper that names a "capability" framing without engaging power is incomplete. Sarkar's research community would expect this engagement.

**Fix locus.** `02_theoretical_framework.md` integrates Khera (2022), Masiero (2023), Krishna (2024), Zuboff (2019), Couldry & Mejias (2019), Mejias & Couldry (2024).

## Section B — Identification / Causal-claim gaps

### B1. The headline result is mechanically over-determined. **(P0)**

**Evidence.** `data/gold/dclo_causal_model_fit.csv`:

| spec | n_obs | r2_within | residual_std |
|---|---|---|---|
| baseline (TWFE l1) | 470 | 0.381 | 0.117 |
| pooled l1 | 470 | **0.910** | 0.265 |

Pooled β on `DCLO_score_lag1` ≈ **0.998** (95 % CI [0.949, 1.048]). This is essentially a regression of a composite on a one-period lag of the same composite, mediated by the structural overlap in indicators. The TWFE specification reduces this by within-entity demeaning, but the residual β = 0.62 still substantially reflects the indicator overlap, not a behavioural channel.

**Why it matters.** The paper cannot present this as a "DCLO causes service outcomes" finding without the reader concluding the framework is circular. The `04_identification_strategy_revised.md` document treats this honestly.

### B2. The placebo is too weak. **(P1)**

**Evidence.** Within-year permutation placebo (`placebo_permuted_l1`): β ≈ −0.014, p ≈ 0.119. The build calls this "expected for a valid falsification test."

**Why it matters.** Within-year permutation breaks every signal — including spurious ones. A reviewer would expect at minimum:
- a **placebo outcome** (a variable that should not respond, e.g. agricultural land share);
- a **leave-future-out** placebo regressing future-DCLO on present-outcome;
- a **policy-event placebo** around DPI rollouts (UPI 2016, Aadhaar saturation 2014–2018, India Stack 2017–2019).

None implemented.

**Fix locus.** `05_robustness_protocol.md`.

### B3. Robustness battery is small and lacks GMM / synthetic control / IV. **(P1)**

**Evidence.** Three robustness specs (t-2 lag, no-control, pooled). No Arellano-Bond GMM (warranted given lagged-DV concerns). No instrument. No synthetic-control on India around the DPI rollout. No event study.

**Fix locus.** `05_robustness_protocol.md`.

### B4. Method-comparison instability is reported as "agreement". **(P0)**

**Evidence.** `data/gold/dclo_method_comparison.csv`:

| year | ρ baseline vs causal |
|---|---|
| 2015 | −0.19 |
| 2016 | **−0.55** |
| 2017 | −0.42 |
| 2018 | −0.17 |
| 2019 | +0.29 |
| 2024 | **+0.76** |

The dashboard plots this as "method agreement over time" with no narrative. The pattern shows the causal-signal score and the index disagree more than they agree across years — a sign that the causal layer is identifying something other than the index does.

**Why it matters.** This is the single most diagnostic finding in the build, and it is currently presented as a routine plot. A paper would foreground it.

**Fix locus.** `09_paper_draft.md` Section 6 makes this the centerpiece of the methodological argument. Dashboard caveat banner now flags it.

### B5. No DAG, no estimand, no SUTVA discussion. **(P1)**

**Evidence.** `docs/dclo-identification-strategy.md` lists "threats" verbally but provides no DAG. No formal estimand. No discussion of SUTVA / cross-country spillovers.

**Fix locus.** `04_identification_strategy_revised.md` includes a DAG.

## Section C — Data and panel-coverage gaps

### C1. India state-year panel is effectively two cross-sections. **(P0 for the state track.)**

**Evidence.** `data/gold/dclo_state_year.csv` has 95 rows. Years present are 2014, 2016, 2019. NFHS-4 (2015–16) and NFHS-5 (2019–21) are the binding constraint. The dashboard markets this as a "panel."

**Fix locus.** State track must either (i) be reframed as repeated cross-section, (ii) be expanded to include yearly indicators (RBI DBIE, MoSPI, NPCI/UPI, NSS-PLFS), or (iii) be deferred. `12_known_issues.md` records this; `09_paper_draft.md` proposes (i) for the paper and a roadmap to (ii).

### C2. Country panel uses indicators with 89–91 % missingness. **(P1)**

**Evidence.** `data/gold/dclo_indicator_missingness_report.csv`:

| indicator | missingness | n_obs | domain |
|---|---|---|---|
| WB_IT.NET.USER.ZS | 0.915 | 45 | ACC |
| WB_IT.CEL.SETS.P2 | 0.915 | 45 | ACC |
| WB_IT.NET.SECR.P6 | 0.906 | 50 | ACC |
| WB_CC.PER.RNK | 0.915 | 45 | AGR |
| WB_PV.PER.RNK | 0.915 | 45 | AGR |
| WB_SE.ADT.LITR.ZS | **0.987** | 7 | SKL |

Yet `docs/dclo-indicator-mapping.md` states the rule: "If missingness > 30% for an indicator in analysis window: drop indicator or replace with validated proxy." The intake gating uses `pct_observed_global` (which is computed per-indicator on populated rows only), not joint coverage in the panel. Effective panel coverage is ~8 %. Indicators are retained.

**Fix locus.** Tighten gate in `config/dpi_country_sources.yml`: replace `pct_observed_global` gate with `panel_coverage` gate. `12_known_issues.md` records.

### C3. State-track context-adjustment is a constant. **(P1)**

**Evidence.** Inspection of `dclo_state_year.csv` shows `CTX_dpi_composite_v2 = 3.857...`, `CTX_dpi_confidence_score = 72.0`, `Z_CTX_dpi_composite_v2 = -0.469` for **every row** (verified by sampling). The "context-adjusted DCLO" is therefore a rigid additive shift identical for all states.

**Why it matters.** The dashboard offers a `DCLO_score_context_adjusted` toggle as if it adjusts state scores by India's DPI context. It does, but only as a global level shift — adding zero comparative information.

**Fix locus.** Either (a) drop the toggle, or (b) make CTX time-varying using the year-specific India DPI score and document the addition as a level-shift only.

### C4. State centroids hard-coded; no boundary file. **(P3)**

**Evidence.** `dashboard/dclo_dashboard.py` lines 33–69 hard-code state centroid lat/lon. No GeoJSON boundary file shipped. Choropleth is not possible.

**Fix locus.** Optional. Ship Survey-of-India-aligned GeoJSON if dashboard is the published artefact.

### C5. No microdata. **(P2)**

**Evidence.** Build uses macro composites only. ITU IDI, ITU ICT-skills, IFC Findex micro file, OECD PIAAC, UNDP-OPHI MPI, NSS-PLFS, NFHS — none are ingested at micro level.

**Fix locus.** Phase-2 plan in `12_known_issues.md`.

## Section D — Reporting / standard-checks gaps

### D1. "All standard checks passed" is misleading. **(P0)**

**Evidence.** `data/gold/dclo_standard_checks_summary.json` reports `overall_passed: true`. The checks test data integrity (no duplicates, no nulls, range plausibility). They do **not** test construct validity. A reviewer reading the dashboard headline "Standard checks: PASS" alongside the rankings would conclude the rankings are validated. They are not.

**Fix locus.** Dashboard sidebar now distinguishes **Data-integrity checks (PASS)** from **Construct-validity checks (FAIL — see academic/03_indicator_validity_audit.md)**. The latter is now an explicit failing-check.

### D2. No comparison with existing indices. **(P1)**

**Evidence.** No table comparing DCLO ranks to ITU IDI ranks, WEF NRI, World Bank GovTech Maturity Index, UN EGDI, or BCG Digital Inclusion Index.

**Fix locus.** Required figure in `09_paper_draft.md` Section 5.

## Section E — Reproducibility / open-science gaps

### E1. No preregistration. **(P2)**

**Fix locus.** `11_preregistration.md` (drafted; to be locked before next refresh).

### E2. No DOI; no archived data version per release. **(P2)**

**Fix locus.** Zenodo deposit per release; checksums already produced by audit logger.

### E3. No data-management plan, no ethics statement. **(P2)**

**Fix locus.** `06_ethics_and_responsible_use.md`.

## Section F — Ethical / political-economy gaps

### F1. No reflexivity on who is rendered legible. **(P0 for a JSGP paper.)**

**Evidence.** Country panel covers 47 (mostly larger) economies; small island states, low-income African states, and several South Asian peers do not appear. The dashboard claims a "comparative track" without naming this exclusion.

**Fix locus.** `06_ethics_and_responsible_use.md`; explicit caveat in dashboard.

### F2. No engagement with DPI critique literature. **(P0 for a JSGP paper.)**

**Evidence.** Khera (2022), Masiero (2023), Rao (2019), Krishna (2024), Bhuta et al. (2018), Merry (2016), Couldry & Mejias (2019) — all uncited.

**Fix locus.** `02_theoretical_framework.md`; `09_paper_draft.md` Section 2.

### F3. No statement on harms / misuse of rankings. **(P1)**

**Fix locus.** `06_ethics_and_responsible_use.md`.

## Section G — Composition of the paper

### G1. No paper exists. **(by design until now.)**

**Fix locus.** `08_paper_outline.md` and `09_paper_draft.md`.

## Summary table

| ID | Severity | Fix locus | Status |
|---|---|---|---|
| A1 | P0 | `02_theoretical_framework.md` | drafted |
| A2 | P0 | `03_indicator_validity_audit.md` | drafted |
| A3 | P0 | `03_indicator_validity_audit.md` | drafted |
| A4 | P0 | `03_indicator_validity_audit.md` | drafted |
| A5 | P0 | `03_indicator_validity_audit.md` | drafted |
| A6 | P1 | dashboard + `02_theoretical_framework.md` | proposed |
| A7 | P1 | `02_theoretical_framework.md` | drafted |
| B1 | P0 | `04_identification_strategy_revised.md` | drafted |
| B2 | P1 | `05_robustness_protocol.md` | drafted |
| B3 | P1 | `05_robustness_protocol.md` | drafted |
| B4 | P0 | dashboard caveat + `09_paper_draft.md` §6 | live + drafted |
| B5 | P1 | `04_identification_strategy_revised.md` | drafted |
| C1 | P0 | `12_known_issues.md`; reframed in paper | recorded |
| C2 | P1 | tighten gate; `12_known_issues.md` | recorded |
| C3 | P1 | dashboard caveat | live |
| C4 | P3 | optional GeoJSON | not started |
| C5 | P2 | phase-2 plan | recorded |
| D1 | P0 | dashboard sidebar split | live |
| D2 | P1 | `09_paper_draft.md` §5 | drafted (placeholder for figures) |
| E1 | P2 | `11_preregistration.md` | drafted |
| E2 | P2 | Zenodo per release | recorded |
| E3 | P2 | `06_ethics_and_responsible_use.md` | drafted |
| F1 | P0 | `06_ethics_and_responsible_use.md` + caveat | drafted + live |
| F2 | P0 | `02_theoretical_framework.md`; `09_paper_draft.md` §2 | drafted |
| F3 | P1 | `06_ethics_and_responsible_use.md` | drafted |
| G1 | — | `08_paper_outline.md`; `09_paper_draft.md` | drafted |
