FROM python:3.11-slim

# System basics
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install lightweight build deps (kept minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl ca-certificates git \
    && rm -rf /var/lib/apt/lists/*

# Working directory
WORKDIR /app

# Install Python deps first (layer caching)
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r /app/requirements.txt

# Copy the repo
COPY . /app

# Default env (override in .env / deploy)
ENV APP_HOST=0.0.0.0 \
    APP_PORT=8000 \
    EMBED_BACKEND=openai \
    EMBED_MODEL=pranav-s/MaterialsBERT \
    OPENAI_EMB=text-embedding-3-small \
    SPACY_MODEL=/app/harvester/miner/ner_model/model-best \
    ADMIN_TOKEN=changeme

# Healthcheck (optional)
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=5 \
  CMD python - <<'PY'\
import os, sys, json, urllib.request
try:
    req = urllib.request.Request(f"http://127.0.0.1:{os.getenv('APP_PORT','8000')}/healthz")
    with urllib.request.urlopen(req, timeout=4) as r:
        sys.exit(0 if r.status==200 else 1)
except Exception:
    sys.exit(1)
PY

EXPOSE 8000

# Gunicorn with sensible resource limits
CMD ["gunicorn", "-w", "2", "-k", "gthread", "--threads", "2", "--timeout", "90", \
     "--worker-tmp-dir", "/dev/shm", "-b", "0.0.0.0:8000", "app:app"]
