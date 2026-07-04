# =============================================================================
# Mini AI Assistant — production-ish image
# Multi-stage: builder (compiles wheels) → runtime (slim, non-root)
# =============================================================================

# ---- Stage 1: builder ----------------------------------------------------
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Build tooling for chromadb, docling, onnxruntime wheels.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    libffi-dev \
    libssl-dev \
    libxml2-dev \
    libxslt1-dev \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Layer 1: requirements only (cached when source changes)
COPY requirements.txt ./
# BuildKit cache mount keeps pip wheels between builds — much faster rebuilds.
# --no-build-isolation avoids pip spinning up an isolated build env that
# runs out of memory compiling grpcio/onnxruntime tokenizers.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip && \
    pip install --no-build-isolation --prefer-binary -r requirements.txt


# ---- Stage 2: runtime ----------------------------------------------------
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PATH="/app/.local/bin:${PATH}"

# Slim runtime deps only — no compilers.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    tini \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 mini

WORKDIR /app

# Copy installed packages from builder.
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy source.
COPY --chown=mini:mini . /app

# Persistent dirs.
RUN mkdir -p /app/.chroma /app/logs /app/data \
    && chown -R mini:mini /app

USER mini

EXPOSE 8000 8501

# Healthcheck hits the FastAPI endpoint (cached, so cheap).
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/healthz || exit 1

ENTRYPOINT ["/usr/bin/tini", "--"]

# Default = API; compose overrides to "streamlit run" for the UI service.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]