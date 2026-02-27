# India Open Data Pipeline + Power BI

This project ingests datasets from India's National Open Data Portal (`data.gov.in`), builds analytics-ready tables, and prepares outputs for Power BI dashboards.
It also supports RBI DBIE dataset ingestion from direct download URLs or local DBIE exports.

## Architecture Outline

- **Ingest**: Pull JSON records from `api.data.gov.in/resource/{resource_id}`
- **Store raw**: Save each extract as timestamped JSONL in `data/raw/`
- **Curate**: Normalize records into tabular CSV in `data/curated/`
- **Serve (gold)**: Build KPI-friendly aggregates in `data/gold/`
- **Visualize**: Connect Power BI to gold CSVs (or warehouse tables later)

## Project Structure

```text
india-open-data-powerbi/
  config/
    sources.example.yml
    rbi_dbie_sources.example.yml
  data/
    raw/
    curated/
    gold/
  docs/
    data-contract.md
    dashboard-spec.md
    runbook.md
  src/
    ingestion/
      data_gov_in_client.py
      run_ingestion.py
      run_rbi_dbie_ingestion.py
    transforms/
      build_gold.py
    quality/
      validate.py
  .env.example
  requirements.txt
```

## Quick Start

1. Create a virtual environment and install dependencies.
2. Copy `.env.example` to `.env` and add your API key.
3. Copy `config/sources.example.yml` to `config/sources.yml`.
4. Run ingestion, quality checks, and transformation.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
cp config/sources.example.yml config/sources.yml

python src/ingestion/run_ingestion.py --config config/sources.yml
python src/quality/validate.py
python src/transforms/build_gold.py
```

## RBI DBIE Pipeline

Use this when sourcing indicators from RBI DBIE tables at [DBIE Home](https://data.rbi.org.in/DBIE/#/dbie/home).

Why this mode exists:

- Some DBIE tables can be downloaded directly by URL.
- Some tables are easier to export manually from the DBIE UI and process as local files.

Setup and run:

```bash
cp config/rbi_dbie_sources.example.yml config/rbi_dbie_sources.yml
# Edit config/rbi_dbie_sources.yml with DBIE download URLs or local exported file paths
python src/ingestion/run_rbi_dbie_ingestion.py --config config/rbi_dbie_sources.yml
```

Output:

- raw snapshots: `data/raw/<dataset>_<timestamp>.csv`
- latest curated files: `data/curated/<dataset>_latest.csv`

## DCLO Dashboard (Local App)

An interactive dashboard is available at:

- `dashboard/dclo_dashboard.py`

Run locally:

```bash
pip install -r requirements.txt
streamlit run dashboard/dclo_dashboard.py
```

The dashboard includes:

- state ranking by DCLO score
- year filter and state trend lines
- domain score heatmap
- state domain profile bars
- built-in dashboard explainer
- map view and CSV download actions
- dual-track mode:
  - India state-year
  - country-year comparative (DPI)

## Hosting

See:

- `docs/hosting-guide.md`

Fastest path is Streamlit Community Cloud with app entrypoint:

- `dashboard/dclo_dashboard.py`

## DPI Country-Year Pipeline

Build comparative country-year DCLO from DPI panel data:

```bash
python src/transforms/build_dclo_country.py --config config/dpi_country_sources.yml
```

Outputs:

- `data/gold/dclo_country_year.csv`
- `data/gold/dpi_indicator_intake.csv`
- `data/gold/dpi_selected_indicators_by_domain.json`
- `docs/dpi-indicator-intake.md`

## Validation Pack

Run diagnostics for selected country indicators:

```bash
python src/quality/validate_dclo_indicators.py --data-dir data/gold
```

Governance reference:

- `docs/dclo-model-governance.md`

## Data Contract Notes

- Producer: `data.gov.in` portal datasets
- Consumer: Power BI model for national trend and state-level comparisons
- Contract boundary: each configured `resource_id` and selected fields in `sources.yml`
- Change handling: treat added fields as non-breaking; renamed/removed fields as breaking

## Initial Power BI Focus

- National trends over time
- State-level ranking and comparisons
- Sector/category drilldowns
- Freshness status tile (latest successful ingestion timestamp)
