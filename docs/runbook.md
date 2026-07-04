# Run-book — Mini AI Assistant (v2.2)

Each section corresponds to one rule in `ops/alerts.yaml`. Read the section
*before* going on-call.

## HighLatencyLLM

- **Symptom**: p99 of `request_stage_seconds{stage="llm"}` > 8s for 5m.
- **Likely causes**: Ollama Cloud degradation; under-sized Free-tier slot
  queuing; unusually long context flooded the GPU.
- **Triage**:
  1. Open Grafana → "p50 / p99 latency by stage" panel.
  2. Check Ollama Cloud status page.
  3. Inspect Mongo for sessions with abnormally long histories.
- **Mitigation**: lower `max_tokens` in `backend/llm/client.py`; switch
  primary to the fallback model; briefly widen the alert for the
  outage window.

## HighFallbackRate

- **Symptom**: `answerability_decisions_total{decision="fallback"} / total`
  > 30% for 10m.
- **Likely causes**: ingest drift (corpus changed but `confidentiality_gate`
  threshold wasn't recalibrated); embedding model swap; OCR errors on a
  new PDF batch.
- **Triage**:
  1. Re-run `tests/test_eval.py` against the current collection.
  2. Inspect `retrieval_topk_scores` distribution — if dense scores are
     flat, suspect embedding drift.
- **Mitigation**: re-ingest the affected corpus, recalibrate threshold,
  or tighten chunks.

## HighErrorRate

- **Symptom**: 5xx rate > 1% over 5m.
- **Triage**:
  1. Tail `logs/app.log` for `request_id` correlations.
  2. Check Ollama/HF/Mongo status dashboards.
- **Mitigation**: restart the FastAPI process; rotate `OLLAMA_CLOUD_API_KEY`
  if 401s are present.

## VectorStoreDown

- **Symptom**: Chroma scrape target down for 2m.
- **Triage**: `ls -la .chroma`; verify disk space; check that the
  process wasn't killed for OOM.
- **Mitigation**: clear `.chroma/` only if a full re-ingest is acceptable.

### Auto-recovery (preferred)

As of v2.3 the FastAPI worker self-heals on the next ingest attempt:

1. `ChromaStore.__init__` runs `auto_recover_if_corrupt(persist_dir)`,
   which probes chromadb in an **isolated subprocess** (a native crash
   inside chromadb's Rust HNSW code can never reach the worker).
2. If the probe fails, the corrupt `.chroma/` is renamed to
   `.chroma.bak-<UTC-stamp>` (move-aside, never wipe) and a fresh empty
   directory is created.
3. The very next `/ingest` call rebuilds the collection and BM25 cache
   from `data/` automatically — the Streamlit UI shows a clear
   "Click Upload again to rebuild the index" warning instead of a
   silent failure.
4. If the upsert still fails *after* the self-heal, the request is
   returned with `fallback_reason: "chroma_recovered_retry_ingest"` so
   the user can retry instead of seeing a 500.

### Manual recovery (when auto-recovery isn't enough)

If the worker can't even start — usually because a previous crash
left file handles locked — run from the project root:

```
make recover-chroma
```

or directly:

```
powershell -ExecutionPolicy Bypass -File scripts\recover_chroma.ps1
```

This script:

- stops any running `uvicorn` (so chromadb releases its mmap'd files),
- renames `.chroma/` → `.chroma.bak-<UTC-stamp>/` (kept on disk for
  forensics, never wiped),
- prints the size + file count of the quarantined directory,
- leaves a fresh empty `.chroma/` for the next ingest.

The next `/ingest` request (or `python -m backend.ingestion.pipeline`)
rebuilds the collection and BM25 cache from `data/`. No other action
is required.

## HealthCheckDegraded

- **Symptom**: `health_status{component="..."} == 0` for 2m.
- **Triage**: open `/healthz` JSON, see which component.
- **Mitigation**: per-component.

## PromptInjectionSpike

- **Symptom**: `prompt_injection_total` rate > 0.5/s for 10m.
- **Triage**: grep logs for `"signals"` keys; identify the surface
  (`user` vs `document`).
- **Mitigation**: temporarily raise the detector threshold; flag the
  offending uploader.
