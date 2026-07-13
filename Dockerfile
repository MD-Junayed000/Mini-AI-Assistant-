# Multi-stage build for Mini AI Assistant.
#
# Stage 1 — frontend: install Node deps, build the React/Vite SPA into
#           /web/dist which the FastAPI app serves as static files.
# Stage 2 — backend:  copy /web/dist + the Python sources into a slim image
#           and run uvicorn. One image, one container, one port.

# ---------- Stage 1: frontend build --------------------------------------
FROM node:20-alpine AS web

WORKDIR /web

COPY frontend/package.json frontend/package-lock.json* ./
RUN npm install --no-audit --no-fund

COPY frontend/ ./
RUN npm run build

# ---------- Stage 2: backend runtime --------------------------------------
FROM python:3.11-slim AS app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential poppler-utils \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY backend/ ./backend/
COPY main.py ./
COPY pyproject.toml ./
COPY ops/ ./ops/
COPY data/ ./data/

COPY --from=web /web/dist /web/dist

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=3).status == 200 else 1)"

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
