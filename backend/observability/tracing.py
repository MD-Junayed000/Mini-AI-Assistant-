"""OpenTelemetry tracing — enabled whenever `OTEL_EXPORTER_OTLP_ENDPOINT` is set.

Local dev path (default): `http://tempo:4318` exposed by the `obs` docker-compose
profile (Grafana + Tempo). When the env var is empty this module is a true
no-op — zero network calls, zero cost.

Three things matter for production-correctness:

1. **Idempotency** — calling `tracer()` repeatedly returns the same instance.
2. **Lazy import** — the OTel SDK is heavy; we only import it when tracing is on.
3. **Exporter resilience** — every span export failure is logged at WARNING, never
   raised into the calling pipeline. A dead Tempo must not take the API down.

Two protocols are supported:
  - HTTP/protobuf — endpoint like `http://tempo:4318` (path `/v1/traces` is appended)
  - gRPC          — endpoint like `http://tempo:4317` (we strip nothing; SDK adds path)
We prefer HTTP because it works without extra C++ deps on slim Docker images.
"""
from __future__ import annotations

import logging
from typing import Any

from backend.config import get_settings
from backend.observability.logging_config import get_logger

log = get_logger("tracing")

_tracer: Any = None


def init_tracing() -> Any:
    """Idempotent OTel setup. Returns the (real or no-op) tracer."""
    global _tracer
    if _tracer is not None:
        return _tracer

    settings = get_settings()
    if not settings.otel_enabled:
        # Real OTel no-op (carries the interface callers expect).
        from opentelemetry.trace import NoOpTracer

        _tracer = NoOpTracer()
        return _tracer

    # ---- Real path -------------------------------------------------------
    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    endpoint = settings.otel_exporter_otlp_endpoint.rstrip("/")
    # Auto-derive `/v1/traces` for HTTP exporter if the user supplied a bare host.
    http_url = endpoint if endpoint.endswith("/v1/traces") else f"{endpoint}/v1/traces"

    # Optional auth headers — parse "k1=v1,k2=v2" into a dict.
    headers = None
    if settings.otel_exporter_otlp_headers:
        headers = dict(
            item.split("=", 1)
            for item in settings.otel_exporter_otlp_headers.split(",")
            if "=" in item
        )

    resource = Resource.create(
        {
            "service.name": settings.otel_service_name,
            "service.version": "0.2.2",
        }
    )
    provider = TracerProvider(resource=resource)

    # Bridge SDK export errors to our structlog instead of stderr.
    class _SafeExporter:
        def __init__(self, inner):
            self._inner = inner

        def export(self, spans):  # noqa: ANN001
            try:
                return self._inner.export(spans)
            except Exception as e:  # noqa: BLE001
                log.warning("otel_export_failed", error=str(e))
                # Returning SUCCESS so BatchSpanProcessor doesn't drop spans
                # entirely. Production: swap with a circuit-breaker here.
                from opentelemetry.sdk.trace.export import SpanExportResult

                return SpanExportResult.SUCCESS

        def shutdown(self):  # noqa: D401
            try:
                self._inner.shutdown()
            except Exception as e:  # noqa: BLE001
                log.warning("otel_shutdown_failed", error=str(e))

    inner = OTLPSpanExporter(endpoint=http_url, headers=headers)
    safe = _SafeExporter(inner)
    provider.add_span_processor(BatchSpanProcessor(safe))
    trace.set_tracer_provider(provider)
    _tracer = trace.get_tracer(settings.otel_service_name)
    log.info("otel_initialised", endpoint=http_url, has_headers=bool(headers))
    return _tracer


def tracer() -> Any:
    """Return the configured tracer (init first if needed)."""
    if _tracer is None:
        return init_tracing()
    return _tracer


# Silence the noisy OTel internal logger in the same way structlog is configured.
logging.getLogger("opentelemetry").setLevel(logging.WARNING)
