FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /srv/app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY storage ./storage
COPY run.py .
COPY eda_insights ./eda_insights
COPY classification_insights ./classification_insights

RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p /srv/app/storage/models \
    && chown -R appuser:appuser /srv/app
USER appuser

EXPOSE 8000 7860 10000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s \
    CMD python -c "import urllib.request as u; u.urlopen('http://localhost:8000/api/health', timeout=3)" || exit 1

# Single worker: the ModelService uses an in-process threading.RLock singleton.
# Binds to dynamic environment variable $PORT provided by Render/HuggingFace, defaulting to 8000.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
