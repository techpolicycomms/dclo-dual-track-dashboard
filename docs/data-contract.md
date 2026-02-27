# Data Contract (India Open Data Portal)

## Contract Metadata

- Contract ID: `india-open-data-v1`
- Producer: `data.gov.in`
- Consumer: `india-open-data-powerbi`
- Version: `1.0.0`

## Scope

- One or more `resource_id` datasets configured in `config/sources.yml`
- API extract uses JSON format and controlled field projection

## Guarantees

- Ingestion schedule: daily (initial target)
- Freshness objective: <= 24h from source update
- Idempotency rule: each run writes timestamped raw snapshots and replaces curated latest views

## Breaking Change Policy

Breaking changes include:
- field removal or renaming in configured projections
- source endpoint unavailability beyond SLA window

Response:
1. Alert in runbook channel.
2. Freeze downstream refresh.
3. Update transform mappings and semantic model.
