# syntax=docker/dockerfile:1

ARG PYTHON_VERSION=3.12

# ---------- base : Python + dépendances ----------
FROM python:${PYTHON_VERSION}-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    SENSAI_DATA_DIR=/data

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

# ---------- dev : hot reload, code monté en volume ----------
FROM base AS dev

COPY requirements-dev.txt .
RUN pip install -r requirements-dev.txt

# Le code est monté par docker-compose.dev.yml (pas de COPY ici)
EXPOSE 8000 5678
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--reload", "--reload-dir", "/app"]

# ---------- prod : image minimale, utilisateur non-root ----------
FROM base AS prod

RUN useradd --create-home --uid 1000 sensai \
    && mkdir -p /data && chown sensai:sensai /data

COPY --chown=sensai:sensai . .

USER sensai
VOLUME ["/data"]
EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request as u; u.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
