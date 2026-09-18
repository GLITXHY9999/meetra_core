FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /srv/app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY run.py .

RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p /srv/app/storage/models \
    && chown -R appuser:appuser /srv/app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s \
    CMD python -c "import urllib.request as u; u.urlopen('http://localhost:8000/api/health', timeout=3)" || exit 1

# Single worker: the ModelService uses an in-process threading.RLock singleton.
# Multiple OS-level workers each get their own isolated memory space, causing
# divergent model state (worker A trains, worker B still returns "untrained").
# If horizontal scaling is required, externalise model state (e.g. Redis /
# shared volume) and remove this comment before adding --workers N.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
