# Changelog

All notable changes to the DCLO dashboard. Versioning is semantic where the schema is concerned; release cadence is irregular.

## 0.4.0 — 2026-04-29 — Critical-instrument release

The dashboard is reframed as a **critical instrument** rather than a ranking. Companion academic folder is added; dashboard surfaces the construct-validity audit it produces.

### Added
- `academic/` folder with 14 documents: response-to-Sarkar (`00`), gap analysis (`01`, 29 numbered gaps with severity P0–P3), capability-approach theoretical framework (`02`), indicator-validity audit (`03`, 16/20 country-track indicators flagged), revised identification strategy with DAG (`04`), 15-spec robustness protocol with placebo battery (`05`), ethics and responsible-use commitments (`06`), reproducibility checklist (`07`), paper outline with two structural options (`08`), 10,086-word first-pass paper draft targeting *Information, Communication & Society* (`09`), 9 ranked target venues (`10`), pre-analysis plan to lock at 2026-07-01 (`11`), live known-issues log (`12`), dashboard-as-publication plan (`13`), bibliography of ~50 entries.
- `academic/indicator_metadata.json` with per-indicator layer, source, license, mechanism, construct-validity verdict.
- `academic/dag.md` with rendered identification DAG.
- `data/external/standard_family_ranks_2023.csv` with public 2023 ranks for ITU IDI, EGDI, GTMI, NRI for the panel-overlap subset (with citations and source URLs).
- `CITATION.cff` for citable use of the dashboard.
- Dashboard banners that surface the audit:
  - top-of-page **construct-validity warning**;
  - sidebar **Model QA split**: data-integrity (PASS) vs construct-validity (FAIL);
  - **CTX-toggle help text** documenting that the context-adjusted score is a uniform level shift in the current build;
  - **Causal Evidence tab disclosure** reading the headline β as a structural-overlap signature;
  - **Method-comparison plot** retitled "Stability Diagnostic" with a banner explaining the ρ flip from −0.55 (2016) to +0.76 (2024).
- New tabs: **Methods**, **Standard-Family Comparison**, **Inclusion & Reflexivity**, **Releases**.
- **Capability-layer overlay** on the Measurement tab indicator filter (resources / conversion factors / capabilities / functionings / macro context).
- **DAG render** in the Causal Evidence tab.
- **Specification-array forest plot** in the Robustness tab (precursor to the full specification curve preregistered for the next release).
- **Per-indicator evidence cards** with layer chip, source, license, mechanism, decision history.
- **Construct-validity stress-test toggle** in the sidebar (drops construct-failing indicators and re-renders rankings live).
- **Citation block** in the sidebar with BibTeX, APA, build hash, build date, license.
- Top-of-`README.md` pointer to the academic track.

### Changed
- Dashboard reframed throughout: language shifts from "DCLO scores countries on digital capability" to "DCLO is a critical instrument that surfaces the construct-validity gap in the standard digital-development indicator family."
- Method-comparison view renamed from "Method agreement over time" to "Method stability diagnostic over time."
- Sidebar Methodology block now points readers at the academic track.

### Known issues at this release
See `academic/12_known_issues.md`. Of the 17 known issues at this release: 7 are P0, 10 are P1. The P0 set is dominated by the indicator-replacement work (KI-001 through KI-005), which is the priority of the next sprint.

### Reviewer notes
- Construct-validity gates: 16 of 20 country-track indicators fail layer validity in this release.
- Causal coefficient: β = 0.624, SE = 0.100 (n = 470, 47 entities × 10 years). Reviewers must read this as a structural-overlap signature rather than a capability-policy effect — see `academic/04_identification_strategy_revised.md` §5.
- Method stability: Spearman ρ between baseline and causal ranks ranges from −0.547 (2016) to +0.759 (2024) across the panel window.

## 0.3.0 — 2026-02-27 — Dual-track release
- Country-year DPI track added.
- Indicator gating + role classification.
- State-track context-adjusted DCLO option using India DPI composite.
- Dual-track dashboard mode and comparative visuals.

## 0.2.0 — earlier — State-year release
- India state-year DCLO from NFHS, NAFIS, RBI DBIE.
- Streamlit dashboard with ranking, map, trend, heatmap, profile views.

## 0.1.0 — earlier — Pipeline release
- `data.gov.in` ingestion client.
- RBI DBIE ingestion mode.
- Bronze / curated / gold pipeline scaffold.
