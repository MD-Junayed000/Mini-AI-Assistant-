# =============================================================================
# Mini AI Assistant — convenience targets
# Usage: make <target>
# =============================================================================

PY     ?= python
PIP    ?= $(PY) -m pip
PORT   ?= 8000

.PHONY: help install install-dev run api ui test test-offline lint fmt \
        recover-chroma \
        docker-build docker-up docker-down docker-logs clean

help:  ## Show this help.
	@$(MAKE) -p0 2>/dev/null | grep -E '^[a-zA-Z_-]+:.*?## .*$$' | sort | \
		awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install:  ## Install runtime dependencies into the active Python env.
	$(PIP) install -r requirements.txt

install-dev:  ## Install runtime + test dependencies.
	$(PIP) install -r requirements.txt
	$(PIP) install pytest pytest-asyncio httpx prometheus-client

run:  ## Boot API + UI (two foreground processes — use docker-up for prod).
	$(PY) -m uvicorn main:app --reload --port $(PORT) & \
	$(PY) -m streamlit run ui/streamlit_app.py

api:  ## Run only the FastAPI server.
	$(PY) -m uvicorn main:app --reload --port $(PORT)

ui:  ## Run only the Streamlit UI.
	$(PY) -m streamlit run ui/streamlit_app.py

test:  ## Run the full test suite (some cases need live API keys).
	$(PY) -m pytest -q

test-offline:  ## Run only the offline-safe tests.
	$(PY) -m pytest -q \
		tests/test_redactor.py \
		tests/test_health_cache.py \
		tests/test_logging_rotation.py \
		tests/test_injection.py \
		tests/test_tracing.py \
		tests/test_error_handling.py \
		tests/test_chunking.py \
		tests/test_tools.py \
		tests/test_locks.py \
		tests/test_eval.py

lint:  ## Compile-check every module.
	$(PY) -m compileall backend main.py ui

fmt:  ## Quick black-style format (no external deps).
	@echo "tip: pip install black ruff isort && ruff format ."

docker-build:  ## Build the container image.
	docker build -t mini-ai-assistant:latest .

docker-up:  ## Boot API + UI via docker compose.
	docker compose up -d --build
	@echo "API:  http://localhost:8000"
	@echo "UI:   http://localhost:8501"

docker-down:  ## Stop everything.
	docker compose down

docker-logs:  ## Tail compose logs.
	docker compose logs -f --tail=100

clean:  ## Remove caches, bytecode, and local artefacts.
	rm -rf .pytest_cache .ruff_cache .mypy_cache __pycache__ */__pycache__
	find . -name '*.pyc' -delete
recover-chroma:  ## Quarantine a corrupt .chroma/ (stops uvicorn, moves dir aside).
        powershell -ExecutionPolicy Bypass -File scripts\recover_chroma.ps1