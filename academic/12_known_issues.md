# Known Issues — Live Issue Log

Cross-referenced from `01_gap_analysis.md`. Update whenever a new release goes out.

## Open

| # | Severity | Title | Locus | Status |
|---|---|---|---|---|
| KI-001 | P0 | OUT domain (country track) uses macroeconomic stocks (POP.GROW, AG.LND.AGRI.ZS, EG.CFT.ACCS.ZS), not functionings. | `data/gold/dpi_selected_indicators_by_domain.json`; `config/dpi_country_sources.yml` | Replacement set in `03_indicator_validity_audit.md` §1.6. Awaiting Findex / EGDI / WHO ingestion. |
| KI-002 | P0 | AGR domain (country track) uses Worldwide Governance Indicators, not digital agency. | same | Replacement set in `03_indicator_validity_audit.md` §1.4. |
| KI-003 | P0 | SRV domain (country track) uses WTO services-export proxies, not digital service uptake. | same | Replacement set in `03_indicator_validity_audit.md` §1.3. |
| KI-004 | P0 | SKL domain (country track) uses general-education enrolment, not digital skills. | same | Replacement set in `03_indicator_validity_audit.md` §1.2. SDG 4.4.1 from ITU. |
| KI-005 | P0 | ECO domain (country track) uses macroeconomic stocks, not digital-economic participation. | same | Replacement set in `03_indicator_validity_audit.md` §1.5. |
| KI-006 | P0 | Indicator-gating uses pct_observed_global, not joint panel coverage. Indicators with 91 % missingness retained. | `src/transforms/build_dclo_country.py` `select_indicators_by_domain`; `config/dpi_country_sources.yml` `gating.min_panel_coverage` | Tighten: switch primary gate to `panel_coverage`, set min_panel_coverage = 0.40, fail loudly. |
| KI-007 | P0 | "All standard checks passed" reported in dashboard despite construct-validity failures. | `dashboard/dclo_dashboard.py`; `data/gold/dclo_standard_checks_summary.json`; `src/quality/run_standard_checks.py` | **Live fix in dashboard:** sidebar now distinguishes data-integrity (PASS) from construct-validity (FAIL). |
| KI-008 | P0 | India state-year panel is effectively three cross-sections (2014, 2016, 2019). Marketed as a "panel". | `data/gold/dclo_state_year.csv` | Reframe in paper as repeated cross-section. Roadmap to expand: TRAI quarterly internet, NPCI state UPI, NSS-PLFS ICT, eSanjeevani, DigiLocker, PFMS, eNAM, GST, NCRB. |
| KI-009 | P1 | State-track NAT_* indicators are national constants applied to all states; contribute nothing to within-year state variation. | `data/gold/dclo_state_year.csv` (NAT_upi_*, NAT_ib_*) | Move to year-context table; do not include in state-z-score computation. |
| KI-010 | P1 | State-track CTX_dpi_composite_v2 is a single constant 3.857 applied to all rows; "context-adjusted" DCLO is therefore a level shift. | `data/gold/dclo_state_year.csv` | Either drop the toggle or make CTX time-varying using year-specific India DPI score. **Live fix:** dashboard sidebar now warns when toggle is engaged. |
| KI-011 | P1 | Method-comparison ρ between baseline and causal flips from −0.55 (2016) to +0.76 (2024). Currently presented as routine. | `data/gold/dclo_method_comparison.csv`; `dashboard/dclo_dashboard.py` `render_method_comparison` | **Live fix:** dashboard surfaces this as a stability flag. Paper §6 makes it the headline diagnostic. |
| KI-012 | P1 | Placebo battery has only within-year permutation. Missing: leave-future-out, placebo outcome, event-study, parallel-trends. | `src/transforms/build_dclo_causal_panel.py`; `config/dclo_causal_model.yml` | Implement P3, P5, P6 from `05_robustness_protocol.md`. |
| KI-013 | P1 | Robustness battery has 3 specs. Plan calls for 15. | `config/dclo_causal_model.yml` | Implement S6–S15 per `05_robustness_protocol.md`. |
| KI-014 | P1 | No DAG, no estimand statement, no sensitivity (Cinelli–Hazlett RV). | `docs/dclo-identification-strategy.md` | Superseded by `04_identification_strategy_revised.md`. Implement RV computation. |
| KI-015 | P1 | No within-country distributional decomposition (gender, urban-rural, inequality penalty). | dashboard; gold tables | Phase-2 plan. NFHS supports for India track. |
| KI-016 | P1 | No comparison with ITU IDI / EGDI / GTMI / NRI. | gold tables; paper | Required figure for paper. |
| KI-017 | P1 | No engagement with DPI-critique literature in any docs/. | docs/ | Drafted in `02_theoretical_framework.md`. |
| KI-018 | P2 | No preregistration. | folder | `11_preregistration.md` drafted; pending lock. |
| KI-019 | P2 | No DOI / Zenodo deposit per release. | release process | Add at next release. |
| KI-020 | P2 | No data-management plan, no ethics statement file in repo. | repo | `06_ethics_and_responsible_use.md` covers; surface in repo README. |
| KI-021 | P2 | No microdata (Findex micro, ITU IDI raw, ILOSTAT, NSS-PLFS). | data/raw | Phase-2 plan. |
| KI-022 | P2 | No Dockerfile / Nix flake; environment lock is requirements.txt only. | repo root | Phase-2. |
| KI-023 | P3 | State-track choropleth not possible (no GeoJSON shipped). | dashboard | Optional. |
| KI-024 | P3 | Two duplicate repo dirs (`dclo-dual-track-dashboard` and `dclo-dual-track-dashboard-20260227-153409`). | repos/ | Decide on canonical, deprecate other. **Canonical = `dclo-dual-track-dashboard`** (this directory). |

## Closed

(none yet)

## Triage rules

- **P0** must clear before the next academic submission.
- **P1** must clear before the *Information, Communication & Society* submission window.
- **P2** must be acknowledged in the paper limitations section.
- **P3** can be carried.
