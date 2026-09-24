# syntax=docker/dockerfile:1

ARG PYTHON_VERSION=3.12

# ---------- base : Python (+ Node pour les serveurs MCP lancés via npx) + dépendances ----------
FROM python:${PYTHON_VERSION}-slim AS base

# Le code vit dans /app/sensai : `python -m sensai.mcp_server.server` (agent.yaml) trouve le paquet `sensai`,
# et les imports à plat (`from core...`) marchent aussi.
ARG INSTALL_NODE=true
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app:/app/sensai

RUN if [ "$INSTALL_NODE" = "true" ]; then \
        apt-get update && apt-get install -y --no-install-recommends nodejs npm \
        && rm -rf /var/lib/apt/lists/*; \
    fi

# Les chemins de agent.yaml (data/, sessions/, logs/, workspace/) sont relatifs à ce dossier
WORKDIR /app/sensai

COPY requirements.txt .
RUN pip install -r requirements.txt

# ---------- dev : hot reload, code monté en volume ----------
FROM base AS dev

COPY requirements-dev.txt .
RUN pip install -r requirements-dev.txt

# Le code est monté par docker-compose.dev.yml (pas de COPY ici)
EXPOSE 8000 5678
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000", \
     "--reload", "--reload-dir", "/app/sensai", "--reload-include", "*.yaml"]

# ---------- prod : utilisateur non-root ----------
FROM base AS prod

RUN useradd --create-home --uid 1000 sensai

COPY --chown=sensai:sensai . .
# Dossiers écrits à l'exécution : montés en volumes par docker-compose.yml
RUN mkdir -p data sessions logs workspace && chown sensai:sensai data sessions logs workspace

USER sensai
EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=3s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request as u; u.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
