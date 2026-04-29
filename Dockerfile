# Dockerfile for the DCLO Dashboard — reproducible container image.
#
# Build:    docker build -t dclo:0.4.0 .
# Run:      docker run --rm -p 8501:8501 dclo:0.4.0
# Then open http://localhost:8501 in a browser.
#
# This Dockerfile is intentionally minimal so it doubles as a replication recipe.
# Pin the base image and the requirements file; the audit logger writes
# environment metadata at every pipeline run for finer-grained provenance.

FROM python:3.11.9-slim

LABEL org.opencontainers.image.title="DCLO Dashboard"
LABEL org.opencontainers.image.description="Digital Capability for Life Outcomes — interactive critical instrument and companion paper artefact."
LABEL org.opencontainers.image.version="0.4.0"
LABEL org.opencontainers.image.licenses="CC-BY-4.0"
LABEL org.opencontainers.image.source="https://github.com/techpolicycomms/dclo-dual-track-dashboard"
LABEL org.opencontainers.image.authors="Rahul Jha <rjha11@jgu.edu.in>"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHERUSAGESTATS=false

WORKDIR /app

# System packages required for some scientific Python wheels.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies (cache-friendly — copy lockfile first).
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# Copy the rest of the project.
COPY . /app

EXPOSE 8501

# Default command launches the Streamlit dashboard.
CMD ["streamlit", "run", "dashboard/dclo_dashboard.py", \
     "--server.address=0.0.0.0", \
     "--server.port=8501", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
