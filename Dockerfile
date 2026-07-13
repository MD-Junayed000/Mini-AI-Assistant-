# Multi-stage build for Mini AI Assistant

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

# System dependencies
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential poppler-utils \
 && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt --no-cache-dir

# Application code
COPY backend/ ./backend/
COPY main.py ./
COPY pyproject.toml ./
COPY ops/ ./ops/
COPY data/ ./data/

# Frontend build
COPY --from=web /web/dist /web/dist

EXPOSE 8000

# Health check (dynamic port)
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
  CMD python -c '
import os, sys, urllib.request
port = os.getenv("PORT", 8000)
try:
    urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz", timeout=5)
    sys.exit(0)
except:
    sys.exit(1)
'

# Use shell form so $PORT is expanded by Railway
CMD sh -c "uvicorn main:app --host 0.0.0.0 --port \${PORT:-8000}"
