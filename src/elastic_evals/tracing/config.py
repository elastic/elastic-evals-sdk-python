"""Tracing configuration for elastic-evals."""

from __future__ import annotations

from urllib.parse import urlparse

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import NoOpTracerProvider
from pydantic import BaseModel

from .spans import set_run_id


class TracingConfig(BaseModel):
    """Configuration for tracing with a single OTLP exporter."""

    enabled: bool = True
    endpoint: str = "http://localhost:4318"
    api_key: str | None = None
    service_name: str = "elastic-evals"
    run_id: str | None = None


def _normalize_otlp_http_endpoint(endpoint: str) -> str:
    """Normalize OTLP HTTP endpoint to include /v1/traces path."""
    parsed = urlparse(endpoint)
    if parsed.path in ("", "/"):
        return endpoint.rstrip("/") + "/v1/traces"
    return endpoint


def init_tracing(config: TracingConfig) -> None:
    """Initialize OpenTelemetry tracing with a single OTLP exporter."""
    set_run_id(config.run_id)

    if not config.enabled:
        trace.set_tracer_provider(NoOpTracerProvider())
        return

    resource = Resource.create({"service.name": config.service_name})
    provider = TracerProvider(resource=resource)
    headers = {"Authorization": f"ApiKey {config.api_key}"} if config.api_key else {}
    exporter = OTLPSpanExporter(
        endpoint=_normalize_otlp_http_endpoint(config.endpoint),
        headers=headers,
    )
    provider.add_span_processor(BatchSpanProcessor(exporter))

    trace.set_tracer_provider(provider)
