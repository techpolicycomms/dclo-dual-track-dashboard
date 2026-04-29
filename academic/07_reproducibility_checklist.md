# Reproducibility Checklist

Mapped to TOP (Nosek et al. 2015) and ACM Artifact Review levels.

| Item | Status | Evidence |
|---|---|---|
| Data accessible | partial | Source data are public; gold tables in `data/gold/`. No DOI yet. |
| Data citation in paper | pending | `09_paper_draft.md` Section "Data". |
| Code accessible | yes | Repo `dclo-dual-track-dashboard`. |
| Code citation in paper | pending | `09_paper_draft.md` Section "Code". |
| Materials accessible | yes | Configs in `config/`. |
| Design transparency | partial | This folder. |
| Analysis-plan transparency | pending | `11_preregistration.md` to be locked. |
| Preregistration | pending | `11_preregistration.md`. |
| Replication | pending | One independent run by a co-author once preregistration is locked. |
| Provenance per output | yes | Audit-logger SHA-256 inputs/outputs (`src/quality/audit_logger.py`). |
| Random-seed disclosure | yes | Recorded in audit manifests. |
| Environment lockfile | partial | `requirements.txt`, `runtime.txt`. No Docker yet. |
| Computational environment archived | pending | Add Dockerfile or Nix flake. |
| Datasets versioned | partial | Gold tables versioned by repo commit; not by Zenodo DOI. |
| Verification reports per pipeline | yes | `*_verification.json` next to each gold table. |
| Standard-checks summary | yes | `data/gold/dclo_standard_checks_summary.json`. |
| Construct-validity summary | new | `academic/03_indicator_validity_audit.md`. |
| Sensitivity / robustness battery | partial | 3 specs implemented; full protocol in `academic/05_robustness_protocol.md`. |
| Placebo battery | partial | 1 placebo implemented; full set in `academic/05_robustness_protocol.md`. |

## Acceptance criteria for ACM Artifact Available + Functional

1. Repo public.
2. README with build, run, test instructions.
3. Sample input → expected output reproducible from a clean clone.
4. Outputs match published figures within numerical tolerance.

## Acceptance criteria for ACM Artifact Reusable

5. Configurable for new datasets without code changes.
6. Documented extension points.
7. Tests covering critical paths.
