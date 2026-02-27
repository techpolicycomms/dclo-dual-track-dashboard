# Hosting Guide (DCLO Dashboard)

## Option A: Streamlit Community Cloud (Recommended)

Best for quick public/private sharing with minimal ops.

### Steps

1. Push this project to a GitHub repository.
2. Go to Streamlit Community Cloud and create a new app.
3. Select:
   - repository: your repo
   - branch: your chosen branch
   - app file path: `dashboard/dclo_dashboard.py`
4. Deploy.
5. Re-deploy whenever `data/gold/dclo_state_year.csv` or dashboard code changes.

### Notes

- Keep `requirements.txt` in repo root (already done).
- Ensure `data/gold/dclo_state_year.csv` is committed or fetched during startup.
- For controlled sharing, use app visibility settings in Streamlit Cloud.

## Option B: Container Host (Render/Railway/Fly)

Use this if you need more control or larger workloads.

### Minimal start command

```bash
streamlit run dashboard/dclo_dashboard.py --server.port $PORT --server.address 0.0.0.0
```

### Environment

- Python 3.9+
- Install from `requirements.txt`

## Current Local Access

If the app is already running locally:
- local URL: `http://localhost:8501`

This local URL is useful for validation before cloud deploy.
