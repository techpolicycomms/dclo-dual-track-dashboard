# DCLO — Academic Track

This folder is the **publication track** of the Digital Capability for Life Outcomes (DCLO) project. It exists because the dashboard, as built, falls short of what would survive a review by a critical-political-economy supervisor (Prof. Swagato Sarkar, JSGP, OPJGU). It documents what is missing, what has been fixed, what is still aspirational, and what a paper-length defensible version of this work would look like.

The dashboard repo proper (`dashboard/`, `data/`, `src/`) is the **engineering and visualisation artefact**. This folder is the **scholarly artefact** — designed so that a reader, a journal editor, or a thesis committee can:

1. understand the conceptual claim DCLO makes,
2. trace the evidence and method back to that claim,
3. see honestly where the current build does not yet support the claim,
4. follow a roadmap to a publishable paper.

## Contents

| File | Purpose |
|---|---|
| `00_response_to_sarkar.md` | Point-by-point response to Prof. Sarkar's suggestion to treat the dashboard as a paper. What that requires, and where the current build complies / does not. |
| `01_gap_analysis.md` | Structured gap diagnostic across theory, identification, measurement, data, robustness, ethics, normative framing. |
| `02_theoretical_framework.md` | Sen capability approach + Nussbaum + Robeyns + critical-data-studies grounding. Why DCLO is a capability index and not a digital-development index. |
| `03_indicator_validity_audit.md` | Indicator-by-indicator audit of the country-track and state-track gold tables. Identifies category errors (e.g. WTO services exports as "outcome realisation"). Lists the right indicator family and where to find it. |
| `04_identification_strategy_revised.md` | DAG, threats to identification, why the headline TWFE result is fragile, what would need to change for a credible causal claim. |
| `05_robustness_protocol.md` | Pre-specified robustness battery (currently the build has 3 specs; the protocol calls for ~15). |
| `06_ethics_and_responsible_use.md` | Who benefits from being measured, who is harmed, how a ranked dashboard interacts with state datafication and platform power. |
| `07_reproducibility_checklist.md` | Mapped to TOP guidelines, ACM Artifact Review, and Christensen & Miguel transparency principles. |
| `08_paper_outline.md` | Full section outline with target word counts. |
| `09_paper_draft.md` | First-draft paper text — long-form, ready to refine with Sarkar. |
| `10_target_venues.md` | Ranked target journals with fit notes, scope, and lead times. |
| `11_preregistration.md` | Pre-analysis plan to be locked before the next data refresh. |
| `12_known_issues.md` | Live issue log. Pulls forward the things this folder identifies as broken in the current build. |
| `bibliography.bib` | Working bibliography. |

## Status (2026-04-29)

- Drafted: all of the above (first-pass).
- Not done yet: indicator replacement, NSS-PLFS / ITU-IDI / ILOSTAT / Findex ingestion for India and comparators, 2026 data refresh, full preregistration sign-off.
- The **dashboard caveat banner**, **indicator-validity warning**, and **KNOWN_ISSUES** surfaced through the dashboard sidebar are now live (see `dashboard/dclo_dashboard.py` and `12_known_issues.md`).

## Reading order for a reviewer

1. `00_response_to_sarkar.md` — frames the project.
2. `01_gap_analysis.md` — what is wrong.
3. `02_theoretical_framework.md` and `03_indicator_validity_audit.md` — what would be right.
4. `09_paper_draft.md` — proposed publishable form.

## Reading order for an internal contributor

1. `12_known_issues.md` — current debt.
2. `04_identification_strategy_revised.md` and `05_robustness_protocol.md` — the technical work.
3. `11_preregistration.md` — what we lock before next results.
