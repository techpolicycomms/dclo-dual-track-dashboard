# Plan: Dashboard as a Publishable Scholarly Artefact

The dashboard can become *the* publication, not merely a companion to one, on the model of *Distill*, *Journal of Statistical Software*, *MIT Communications in Humanities*, *Frictionless Data*-style executable papers, and the ACM Artifact Reusable badge tradition. To do that the artefact has to satisfy what a reviewer of an *interactive paper* expects, not what a Streamlit demo provides.

This document specifies the top ten improvements that take DCLO from "good engineering documentation" to "publishable scholarly artefact." Each is paired with a concrete implementation locus in the code or data. Items 1–10 are then implemented in the same commit.

## Why this matters

A dashboard-as-paper has to do four things a static paper does *automatically* and a dashboard usually does not:

1. **Make its claims auditable** — every number must be traceable to a source, a method, and a version.
2. **Make its assumptions stress-testable** — the reader must be able to interrogate the result, not only consume it.
3. **Make its position defensible** — the framing, the inclusions, the exclusions, the authorial standpoint are all named.
4. **Make itself citable and versioned** — DOI, BibTeX, changelog, archived snapshot.

The ten improvements below are organised against these four tests.

## Top 10 improvements

### 1. Capability-layer overlay on every indicator

**Why.** Reviewers cannot see whether an indicator targets resources, conversion factors, capabilities, or functionings. The four-layer scheme is the central theoretical commitment of the paper (`02_theoretical_framework.md`). It must be in the dashboard, not just the docs.

**How.** A new `academic/indicator_metadata.json` file records `{layer, source, license, mechanism, status}` per indicator. The Measurement tab gains a "Layer filter" multiselect: "show only capability-layer indicators", "show only functioning-layer indicators", etc. Each indicator carries a coloured layer chip in evidence cards (improvement 9).

### 2. Disagreement-with-standard-family panel

**Why.** The headline scholarly contribution is that DCLO disagrees with the standard family (ITU IDI, EGDI, GTMI, NRI) in informative ways. That is the diagnostic value of the index. Currently the dashboard does not render this comparison at all.

**How.** A new tab "Standard-Family Comparison" renders rank tables for the panel-overlap subset, plus a Spearman matrix and a slope-chart of rank shifts. Public 2022/2023 ranks for ITU IDI, EGDI, GovTech Maturity, WEF NRI for the top-N panel overlap are bundled as a small CSV `data/external/standard_family_ranks_2023.csv` with citations.

### 3. DAG and estimand panel

**Why.** No causal-evidence dashboard is publishable without an explicit DAG and an explicit estimand. Reviewers will refuse the paper otherwise.

**How.** A static SVG (or mermaid block rendered server-side) embedded in the Causal Evidence tab, sourced from `academic/dag.md`. The estimand statement (one paragraph, from `04_identification_strategy_revised.md` §1) renders alongside.

### 4. Specification array → curve

**Why.** Five ad-hoc specifications are not a robustness battery. A specification curve (Simonsohn et al. 2020) is the modern reviewer expectation. Until the full 15-spec battery is implemented (`05_robustness_protocol.md`), the dashboard renders the five existing specifications as a *specification array* with a forest plot of β across specs and a clearly marked "extension to full curve pending pre-registration."

**How.** New helper `render_specification_array()` in the Robustness tab; uses `data/gold/dclo_causal_coefficients.csv` and `data/gold/dclo_causal_model_fit.csv`. Reads the placebo and pooled rows separately so the reader sees the contamination signal at a glance.

### 5. Citable / archival apparatus

**Why.** Any artefact-as-paper must be cited the same way a paper is. CITATION.cff and a recommended BibTeX block are the *minimum* expected by JOSS, JSS, Distill, and most Q1 venues today.

**How.** New `CITATION.cff` at repo root in the standard format (Schubotz et al. 2017 spec); a `Cite this dashboard` expander in the sidebar shows BibTeX and APA. The current build hash, build date, and license are surfaced in the same expander.

### 6. Methods tab as a first-class object

**Why.** The methodology, the identification strategy, and the robustness protocol live in `academic/` markdown files. The dashboard has a small "About" expander. That asymmetry is wrong if the dashboard is the publication. The methods need to be *in* the artefact.

**How.** New top-level tab "Methods" renders the contents of `02_theoretical_framework.md`, `04_identification_strategy_revised.md`, and `05_robustness_protocol.md` inline (with a table of contents and anchored sections). Helps the reviewer evaluate without leaving the dashboard.

### 7. Reflexivity & inclusion-exclusion panel

**Why.** Sarkar-shaped review demands explicit accounting for who is rendered legible and who is not. The dashboard currently shows "Country comparative track: 47 economies" with no reasoning.

**How.** New tab "Inclusion & Reflexivity" renders the panel inclusion table (countries actually in the sample for the selected year), the exclusion table with reasons, the panel-coverage statistic, and the author's standpoint statement from `06_ethics_and_responsible_use.md` §1.6.

### 8. Construct-validity stress-test toggle

**Why.** The strongest argument the artefact can make is to show *what happens if we drop the indicators that fail construct validity*. Currently that requires re-running the pipeline. A live toggle would let any reviewer see the diagnostic in seconds.

**How.** Pre-compute two scoring variants at build time:
  - `DCLO_score` (current — all selected indicators)
  - `DCLO_score_construct_validated` (drops the 16/20 indicators flagged in `03_indicator_validity_audit.md`)

The dashboard sidebar gains a toggle "Stress-test mode (drop construct-failing indicators)". When engaged, all rankings, maps, and trend lines re-render with the construct-validated subset. The score-attenuation is the diagnostic.

For this commit (without rebuilding the pipeline), the dashboard pre-computes the construct-validated score *in-app* by dropping the relevant Z-scaled indicator columns from the gold table at load time and recomputing the domain means.

### 9. Per-indicator evidence cards

**Why.** Reviewers can't see why a particular indicator is in the index. They have to read three documents to find out. That is publication-fatal for an interactive artefact.

**How.** A new helper `render_evidence_card(indicator_code)` shows:
  - layer chip,
  - source agency + URL + license,
  - mechanism paragraph (one paragraph, from `indicator_metadata.json`),
  - missingness, VIF, panel coverage, role,
  - decision history (when it was added; whether it has ever been excluded; replacement plan).

Indicator codes are made clickable in the heatmap and the evidence preview tables.

### 10. Releases / annotated changelog tab

**Why.** A paper has versions; a dashboard usually does not. For a paper-equivalent artefact, the version history with annotations is the equivalent of a journal's version-of-record + corrigenda. A reviewer must be able to read what changed between v1 and v2 and why.

**How.** A new tab "Releases" reads `CHANGELOG.md` and the audit-manifest history (each release writes a manifest under `data/gold/*_audit_manifest.json` with input/output SHA-256 and parameters). Renders a timeline with build hashes, dataset versions, parameter changes, and prose annotations.

## Out-of-scope for this commit (parked for next sprint)

- **Counterfactual sliders** (let the user re-weight domains or substitute indicators live and watch the rank shift). Higher value than the current toggle, but requires re-implementing the scoring pipeline as a pure function. Recommended next sprint.
- **Annotation / peer-review mode** (let reviewers leave inline comments on rank/coefficient rows). Useful for community review pre-submission.
- **Bootstrap CI for ranks** (replace Dirichlet stability with a bootstrap-based rank confidence band). Requires deeper rebuild.
- **Container deposit / Binder-style replication** — Dockerfile + Binder badge.
- **Microdata layer** — once NSS-PLFS / NFHS / Findex micro is ingested.

## Acceptance criteria for the commit

1. Dashboard parses (`python -m py_compile dashboard/dclo_dashboard.py` succeeds).
2. New tabs render even if optional inputs are missing (graceful degradation).
3. Each new helper has a one-paragraph docstring stating what scholarly function it performs.
4. `academic/indicator_metadata.json` contains every indicator currently in `data/gold/dclo_country_year.csv` and `data/gold/dclo_state_year.csv`.
5. `CITATION.cff` validates against [Citation File Format 1.2.0](https://citation-file-format.github.io/).
6. `CHANGELOG.md` records this release with version `0.4.0` and the rationale.
7. New academic doc `13_dashboard_publication_plan.md` (this file) cross-links to all of the above.
