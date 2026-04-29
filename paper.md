---
title: "DCLO: a capability-grounded interactive critical instrument for reading the standard digital-development indicator family"
tags:
  - capability approach
  - digital public infrastructure
  - composite indicators
  - critical data studies
  - reproducibility
  - Python
  - Streamlit
  - panel data
authors:
  - name: Rahul Jha
    orcid: 0000-0000-0000-0000
    affiliation: 1
    corresponding: true
affiliations:
  - name: Jindal School of Government and Public Policy (JSGP), O. P. Jindal Global University, Sonipat, India
    index: 1
date: 2026-04-29
bibliography: academic/bibliography.bib
---

# Summary

`DCLO` (*Digital Capability for Life Outcomes*) is an interactive dashboard and a research artefact for reading the standard family of digital-development indices (ITU IDI, WEF NRI, UN EGDI, World Bank GovTech Maturity Index, Portulans NRI) through the capability approach of Sen [@Sen1985Commodities; @Sen1999Development], Nussbaum [@Nussbaum2000Women; @Nussbaum2011Creating], and Robeyns [@Robeyns2017Wellbeing]. It bundles a Python data pipeline, a Streamlit interface with eight tabs, an audit logger, and a companion paper draft into a single citable artefact.

The artefact's scholarly contribution is **diagnostic**, not normative-rank: it surfaces the construct-validity gap between what currently-available cross-national indicators measure and what a capability-grounded reading would require them to measure. Sixteen of twenty country-track indicators in the current build fail layer validity under a four-layer scheme (resources / conversion factors / capabilities / functionings); the artefact makes this gap interactive and inspectable, including a per-indicator evidence card, a construct-validity stress-test toggle that drops failing indicators and re-ranks live, and a counterfactual domain-weight slider panel that lets the user interrogate the index's plural-incommensurability problem.

# Statement of need

Digital Public Infrastructure (DPI) has become a flagship development-policy frame, and the indicator family that legitimates it has proliferated. Existing indices are well-instrumented [@OECDJRC2008Handbook] but, on their own measurement-theoretic terms, are *resource-and-readiness* indices: they track whether infrastructure, institutions, and skills exist, not whether people are substantively free to use them. The capability approach was specifically constructed to resist this conflation [@Sen1999Development]. There is no published cross-national index that operationalises Sen's distinction between resources, conversion factors, capabilities, and functionings for digital systems; nor is there a publicly-released interactive instrument that lets a researcher or reviewer interrogate the standard family from a capability standpoint.

`DCLO` fills that gap as an *artefact-as-publication* in the tradition of `Distill`, the *Journal of Statistical Software*, and the JOSS itself. It targets three audiences:

1. **Researchers** working on digital divides, digital inequalities, DPI political economy, and capability measurement — who need a reproducible benchmark against which to test claims.
2. **Reviewers and editors** who need to see methods, identification strategy, robustness battery, and ethical commitments inline rather than scattered across files.
3. **Policy readers** who need a critically-framed instrument rather than a uncritical ranking.

The dashboard is also designed to support the JSGP doctoral programme on digital infrastructure and labour, where it functions as an empirical instrument for the political-economy critique of DPI [@Khera2022Dissent; @Masiero2023ICT4D; @Krishna2024DPI; @CouldryMejias2019Costs].

# Functionality

`DCLO` provides four user-visible affordances that make it more than a static rank table.

1. **Capability-layer overlay.** Every indicator in the current build is tagged with its target layer (resources, conversion factors, capabilities, functionings, or macro-context-only) and rendered with a coloured chip (`academic/indicator_metadata.json`).
2. **Construct-validity stress-test.** A sidebar toggle drops the 24 indicators flagged as failing the four-layer audit (`academic/03_indicator_validity_audit.md`) and re-computes domain and overall scores live; the rank shift is the diagnostic.
3. **Counterfactual domain-weight sliders.** A second sidebar panel exposes the six domain weights (ACC, SKL, SRV, AGR, ECO, OUT) as live sliders. Sen's plural-incommensurability point is made operational: a reader who thinks AGR is the central capability can shift weight there and see the rank distribution respond.
4. **Standard-family comparison.** The dashboard merges its own ranks with published 2022/2023 ranks from ITU IDI, UN EGDI, World Bank GTMI, and Portulans NRI (bundled as `data/external/standard_family_ranks_2023.csv`), reports Spearman ρ for each pair, and lists the largest rank shifts. Where DCLO disagrees with the family, the disagreement is the diagnostic value.

In addition, `DCLO` ships:

- a two-way fixed-effects panel layer (`src/transforms/build_dclo_causal_panel.py`) with cluster-robust standard errors, a within-year permutation placebo, and Dirichlet-perturbed rank-stability draws, exposing the headline coefficient (β = 0.624, SE = 0.100, n = 470) as a structural-overlap signature [@Wooldridge2010; @CameronMiller2015];
- a per-pipeline audit logger that writes input/output SHA-256 checksums, package versions, environment metadata, and random seeds [@ChristensenMiguel2018];
- an inline DAG and estimand statement (`academic/dag.md`);
- a 10,086-word companion paper draft targeting *Information, Communication & Society*;
- a pre-analysis plan ready to lock at OSF before the next data refresh (`academic/11_preregistration.md`);
- 17 known issues with severity P0/P1 explicitly enumerated (`academic/12_known_issues.md`) — the artefact is honest about what it does not yet do.

# Quality control and reproducibility

The artefact ships three reproducibility paths: a Streamlit-on-bare-Python path (`requirements.txt`, `runtime.txt`), a Docker path (`Dockerfile`), and a Binder path (`binder/`). Each gold-table file is accompanied by a JSON verification report and an audit manifest with input/output checksums. The dashboard's `Data Provenance & Audit` tab surfaces these inline. A `CITATION.cff` file in CFF 1.2.0 [@SchubotzWagner2017CFF] gives the canonical citation; a `CHANGELOG.md` records release-by-release version-of-record edits.

A live smoke test launches the dashboard in headless Streamlit and exercises it with a Playwright browser session; the most recent run reports zero console errors across the eight rendered tabs.

The artefact is published under CC-BY-4.0; bundled external indicators (ITU, UN, World Bank, Portulans) are used under their respective licences as documented in `data/external/SOURCES.md`.

# Limitations

The current build is **not** a capability-validated ranking. It is an instrument for surfacing where the standard indicator family fails to register capability, and a scaffold for the next-release indicator-replacement build (ITU IDI, Findex, GovTech Maturity Index, EGDI, Freedom House FOTN, ILOSTAT, NSS-PLFS, NPCI). Sixteen of twenty country-track indicators currently fail layer validity, and the India state-track is effectively two cross-sections (NFHS-4 and NFHS-5). These limitations are foregrounded throughout the dashboard rather than disguised, in the spirit of [@ChristensenMiguel2018].

# Acknowledgements

This work grew out of supervisory conversations with Prof. Swagato Sarkar at JSGP, whose work on infrastructure and political economy informs the framing throughout. The author is solely responsible for remaining errors.

# References
