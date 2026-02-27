# DCLO Data Plan from India Data Portal (Granular India Coverage)

## What the Documentation Confirms

- India Data Portal organizes datasets by thematic collections and supports cross-sector integration.
- LGD coding is central for spatial interoperability across districts/blocks/villages where available.
- The portal supports search, dataset pages, download, and visualization workflows.

## DCLO-Relevant Collections to Prioritize

Use these collections first to source domain indicators:

1. Financial Inclusion
2. Finance
3. Economy
4. Education
5. Health
6. Socio Economic
7. Social Welfare
8. Government Schemes
9. Infrastructure
10. Science and Technology
11. Rural Development

Treat Climate and Weather as contextual covariates, not core DCLO dimensions.

## Domain-to-Collection Mapping (DCLO)

### 1) Digital Access and Connectivity

- Infrastructure
- Science and Technology
- Rural Development

Target proxies:
- broadband/mobile penetration
- telecom coverage
- rural connectivity indicators

### 2) Digital Skills and Literacy

- Education
- Socio Economic

Target proxies:
- ICT-enabled learning participation
- digital training exposure
- literacy and schooling support indicators

### 3) Economic Participation and Opportunity

- Financial Inclusion
- Finance
- Economy
- Government Schemes

Target proxies:
- digital payment adoption and transaction intensity
- formal account usage
- MSME digital participation proxies

### 4) Service Enablement

- Health
- Education
- Social Welfare

Target proxies:
- digital public service utilization
- e-health/e-learning uptake
- benefit transfer or digital welfare access proxies

### 5) Agency, Safety, and Rights

- Social Welfare
- Crime
- Government Schemes

Target proxies:
- gendered digital participation markers
- cyber grievance/safety proxies
- identity/authentication-enabled access proxies

### 6) Outcome Realization

- Economy
- Health
- Education
- Socio Economic

Target proxies:
- income/resilience proxies
- service continuity gains
- education/health outcome improvements linked to digital enablement

## Granularity Rules (Critical)

- Prefer district-year or block-year where available.
- If datasets mix levels, harmonize to district-year baseline.
- Use LGD codes as primary key for spatial merges.
- Maintain a crosswalk for changed boundaries and code revisions.

## Data Selection Filters

Keep only datasets that pass all filters:

1. Has geography key (preferably LGD-based).
2. Has time field (year/month).
3. Has measurable value field.
4. Has >= 5 years of coverage (or strongest feasible span).
5. Has documented update frequency and source agency.

## Ingestion Metadata to Capture per Dataset

For each chosen dataset record:

- dataset title
- collection/theme
- source agency
- download/API endpoint
- geography level
- LGD availability (yes/no)
- time span
- update frequency
- indicator code mapping (ACC/SKL/ECO/SRV/AGR/OUT)
- data caveats

## Quality Controls Before Modeling

- directionality harmonization (higher = better)
- missingness profile and imputation flags
- outlier handling log
- comparability notes for cross-state/cross-time interpretation

## Deliverables to Create Next

1. `docs/dclo-data-catalog.md` (dataset inventory)
2. `config/sources.yml` (ingestion config)
3. `docs/dclo-variable-codebook.md` (field-level transforms and signs)

## Execution Sequence

1. Build catalog from chosen IDP datasets by collection.
2. Freeze indicator mapping and transformations.
3. Ingest and curate to district-year/state-year tables.
4. Compute domain scores and DCLO.
5. Run formative diagnostics and sensitivity checks.
