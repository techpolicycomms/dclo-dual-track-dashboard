# Runbook (Ingestion + Transformation)

> **Summary**: 1. Ingestion: `src/ingestion/run_ingestion.py`
2. Quality checks: `src/quality/validate.py`
3. Gold build: `src/transforms/build_gold.py`
4. Power BI refresh

- **Source**: project:docs/runbook.md
- **Ingested**: 20260403T102018Z
- **Tags**: operations, pipeline
- **Category**: operations

---

# Runbook (Ingestion + Transformation)

## Pipeline Flow

1. Ingestion: `src/ingestion/run_ingestion.py`
2. Quality checks: `src/quality/validate.py`
3. Gold build: `src/transforms/build_gold.py`
4. Power BI refresh

## Failure Handling

- API 4xx/5xx:
  - Verify API key and `resource_id`
  - Retry once manually
- Empty dataset:
  - Check source portal status
  - Confirm dataset still publishes records
- Schema drift:
  - Update `config/sources.yml` fields
  - Reconcile downstream transforms

## Backfill

- Re-run ingestion for updated config
- Keep raw timestamped files as replay history
- Rebuild gold outputs and refresh dashboard

## Operational Checklist

- [ ] Last successful ingestion timestamp < 24h
- [ ] Validation passed
- [ ] Gold outputs regenerated
- [ ] Power BI refresh succeeded


---

## Key Concepts

- [[Backfill]]
- [[Failure Handling]]
- [[Operational Checklist]]
- [[Pipeline Flow]]
